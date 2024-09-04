import logging
import os

from fastapi import APIRouter

from app.services import open_metadata_service
from common.config import Config

from app.models.response_model import BaseCommonModel, ErrorModel, MessageModel

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/training"),
    tags=['ML training service'],
)


def __init_clustering():
    """
    기존 데이터를 이용하여 HDBScan 기법 적용하여 추천 항목을 만드는 함수
    """
    import pickle
    import hdbscan
    import pandas as pd

    from pickle import PickleError
    from sklearn.feature_extraction.text import TfidfVectorizer
    from app.services.open_metadata_service import get_documents

    from common.config import Config

    documents = get_documents(get_documents({}, True), False)

    try:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(list(documents.values()))
    except (ValueError, TypeError) as e:
        logger.error(f"Vectorization error: {e}")
        raise

    try:
        hdbscan_clusters = hdbscan.HDBSCAN(min_cluster_size=Config.open_metadata.min_cluster_size, metric='cosine')
        labels = hdbscan_clusters.fit_predict(x)
    except ValueError as e:
        logger.error(f"HDBSCAN error: {e}")
        raise

    try:
        df = pd.DataFrame({'id': list(documents.keys()), 'labels': labels})

        if not os.path.exists(Config.clustering.trained_model_path):
            os.makedirs(Config.clustering.trained_model_path)

        df.to_csv(Config.clustering.trained_model_path + '/hdbscan_clusters.csv', index=False)

        with open(Config.clustering.trained_model_path + '/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    except (FileNotFoundError, IOError, PickleError) as e:
        logger.error(f"File write Error: {e}")
        raise


def __get_embedding(text: str, tokenizer, model):
    """
    text를 받아서 embedding하는 함수

    :param text: Embedding을 하고자 하는 text
    :return: Embedding한 결과값
    """
    import torch
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1)


def __save_recommended(recommender: dict):
    """
    recommend를 db에 넣는 함수

    :param recommender: 정렬되어 있는 n가지 추천
    """
    logger.debug("__save_recommended start")

    from sqlalchemy.orm import class_mapper
    from app.models.recommend_classes import RecommenderWithDB, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(Config.database.get_database_url(), echo=False)

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    bulk_recommender = []

    for key, value in recommender.items():
        list_ = []
        while not value.empty():
            list_.append(value.get().target_id)
        list_.reverse()
        bulk_recommender.append(RecommenderWithDB(key, list_))

    session.query(RecommenderWithDB).filter(RecommenderWithDB.id.in_([
        recommender.id for recommender in bulk_recommender
    ])).delete(synchronize_session=False)

    session.bulk_insert_mappings(class_mapper(RecommenderWithDB), [{
        'id': recommender.id,
        'recommend_id': recommender.recommend_id}
        for recommender in bulk_recommender])
    session.commit()


def __init_embedding():
    """
    기존 데이터를 이용하여 Embedding 기법을 적용하여 추천 항목을 만드는 함수
    """

    from itertools import combinations, product
    from app.models.recommend_classes import OverrodePriorityQueue, RecommendEntity
    from app.models.dictionary_enum import DictionaryKeys
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    model = BertModel.from_pretrained('klue/bert-base')

    samples = open_metadata_service.get_samples()

    try:
        for key, value in samples.items():
            for column_name, column_data in value[DictionaryKeys.COLUMNS.value].items():
                values_str = ' '.join(map(str, column_data[DictionaryKeys.VALUES.value]))
                column_data[DictionaryKeys.EMBEDDINGS_RESULT.value] = __get_embedding(values_str, tokenizer, model)
    except KeyError as e:
        logger.error(f"Key exception: {e}")
        raise

    ids = samples.keys()
    ids_combinations = list(combinations(ids, 2))
    recommended_dict = {}
    top_similarities_average = 0
    for source_id, comparison_id in ids_combinations:
        queue_max_size = Config.recommend_settings.max_recommended_count
        source_recommender = recommended_dict[source_id] if source_id in recommended_dict \
            else OverrodePriorityQueue(maxsize=queue_max_size)

        comparison_recommender = recommended_dict[comparison_id] if comparison_id in recommended_dict \
            else OverrodePriorityQueue(maxsize=queue_max_size)
        top_similarity = []
        inclusion_column_count = 0

        source_columns = samples[source_id][DictionaryKeys.COLUMNS.value]
        comparison_columns = samples[comparison_id][DictionaryKeys.COLUMNS.value]

        for source_column, comparison_column in list(product(source_columns, comparison_columns)):
            source_column_dict = source_columns[source_column]
            comparison_column_dict = comparison_columns[comparison_column]

            if (source_column_dict[DictionaryKeys.DATA_TYPE.value] !=
                    comparison_column_dict[DictionaryKeys.DATA_TYPE.value]):
                continue

            source_value_dict = source_column_dict[DictionaryKeys.VALUES.value]
            comparison_value_dict = comparison_column_dict[DictionaryKeys.VALUES.value]

            if source_value_dict != [] and comparison_value_dict != [] and (
                    set(source_value_dict).issubset(set(comparison_value_dict))) or (
                    set(comparison_value_dict).issubset(set(source_value_dict))):
                inclusion_column_count += 1

            top_similarity.append(
                cosine_similarity(source_column_dict[DictionaryKeys.EMBEDDINGS_RESULT.value].numpy(),
                                  comparison_column_dict[DictionaryKeys.EMBEDDINGS_RESULT.value].numpy()))

        if top_similarity:
            top_similarities_average = float(
                np.mean(np.concatenate(sorted(top_similarity[:Config.recommend_settings.max_recommended_count]))))
        source_recommender.put(RecommendEntity(comparison_id, inclusion_column_count, top_similarities_average))
        comparison_recommender.put(RecommendEntity(source_id, inclusion_column_count, top_similarities_average))
        recommended_dict[source_id] = source_recommender
        recommended_dict[comparison_id] = comparison_recommender

    __save_recommended(recommended_dict)


@router.get(path='/clustering',
            response_model=BaseCommonModel,
            summary='Train a model using existing data with clustering',
            description='This API manually executes the training of a machine learning model using existing data with clustering.'
                        'It allows the model to learn patterns and make predictions based on the provided data',
            tags=['ML Training'],
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
async def init_clustering():
    logger.debug("__init_clustering start")

    try:
        __init_clustering()
        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))


@router.get(path='/embedding',
            response_model=BaseCommonModel,
            summary='Train a model using existing data with embedding',
            description='This API manually executes the training of a machine learning model using existing data with embedding.'
                        'It allows the model to learn patterns and make predictions based on the provided data',
            tags=['ML Training'],
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
async def init_embedding():
    logger.debug("__init_embedding start")

    try:
        __init_embedding()
        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))


if __name__ == "__main__":
    # embedding train test

    # 1.1. embedding training
    __init_embedding()

    # 1.2. use embedding
    from app.routers.embedding_router import embedding_recommend

    document_id = '66e39180-b276-484c-aa19-34efa560f32b'

    recommended = embedding_recommend(document_id)

    if ["9e075fd8-af50-4e20-bb58-a88c6c4155d0", "1449562c-a3f5-4a91-9c93-604731174e3d",
        "66e39180-b276-484c-aa19-34efa560f32b", "0672a85d-440a-4a30-b19e-ffad8b3b92f5",
        "259aedab-a68e-4ca9-b6d0-0b5cd6b21169"] != recommended.data.recommended_id:
        print('Error Occurred')
    else:
        print('Successful')

    # 2.1. clustering training
    __init_clustering()

    # 2.2. use clustering
    from app.routers.clustering_router import clustering_recommend

    document_id = '66e39180-b276-484c-aa19-34efa560f32b'

    recommended = clustering_recommend(document_id)
