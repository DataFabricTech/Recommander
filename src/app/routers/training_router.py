import logging
import os

import numpy as np
from fastapi import APIRouter

from app.services import open_metadata_service
from common.async_loop import loop_with_function
from common.config import Config

from app.models.response_model import BaseCommonModel, ErrorModel, MessageModel

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/training"),
    tags=['ML training service'],
)


def __get_representative_value_by_similarity(target_label: int, vectors, labels, documents):
    from sklearn.metrics import pairwise_distances
    cluster_points = vectors[labels == target_label]

    if cluster_points.shape[0] == 0:
        return None

    # 클러스터 내의 점들 간의 거리 계산
    distances = pairwise_distances(cluster_points)

    avg_distances = distances.mean(axis=1)
    representative_idx = avg_distances.argmin()
    doc_id = list(documents.keys())[representative_idx]
    return documents[doc_id]


def __init_clustering():
    """
    기존 데이터를 이용하여 HDBScan 기법 적용하여 추천 항목을 만드는 함수

    대상 데이터: 데이터 모델의 항목(컬럼, 이름, 설명, Tag 등)
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
        x = vectorizer.fit_transform([document[1] for document in documents.values()])
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
        df = pd.DataFrame(
            {'id': list(documents.keys()), 'name': [document[0] for document in documents.values()], 'labels': labels})
        model_path = Config.recommend_settings.clustering.trained_model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        df.to_csv(model_path + '/hdbscan_clusters.csv', index=False)

        with open(model_path + '/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        with open(model_path + '/hdbscan_clusters.pkl', 'wb') as f:
            pickle.dump(hdbscan_clusters, f)
    except (FileNotFoundError, IOError, PickleError) as e:
        logger.error(f"Clustering File write Error: {e}")
        raise

    '''
    데이터 모델에서 카탈로그를 자동적으로 설정하고자 한다.
    가지고 있는 데이터의 이름, 설명, 컬럼 이름 등(extract_text_from_table_json)을 이용하여 클러스터링한 후, 군집들의 대표를 뽑은 후,
    대표의 테이블 이름을 카탈로그라고 한다.
    
    3. 새로운 데이터 입력(router를 만들어서 해야지)
    4. 새로운 데이터 포함 클러스터링
    5. 새로운 데이터의 클러스터링 번호를 통한 대표값 출력
    '''
    # 클러스터링 된 값에서 대표값 추출 - 클러스터 내에서 데이터 포인터의 평균 거리가 가장 작은 점 선택
    try:
        rep_df = pd.DataFrame(columns=['cluster_label', 'representative_name'])
        for cluster in np.unique(labels):
            if cluster != -1:
                document = __get_representative_value_by_similarity(cluster, x, labels, documents)
                if document[0] is not None:
                    new_row = pd.DataFrame({'cluster_label': [cluster], 'representative_name': [document[0]],
                                            'document_text': [document[1]]})

                    rep_df = pd.concat([rep_df, new_row], ignore_index=True)

        rep_df.to_csv(model_path + '/representative_values.csv', index=False)

    except (FileNotFoundError, IOError, PickleError) as e:
        logger.error(f"Representative File Write Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unknown Error: {e}")
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

    대상 데이터: Sample Column Value 대상
    1. Sample id combination
    2. 각 컬럼 별 일치(포함) 관계 확인
    3. 각 컬럼 별 유사도 확인
    4. 저장 우선 순위 일치(포함) 컬럼 개수 > 유사도
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
            description='This API manually executes the training of '
                        'a machine learning model using existing data with clustering.'
                        'It allows the model to learn patterns and make predictions based on the provided data',
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
async def init_clustering():
    logger.debug("__init_clustering start")

    try:
        await loop_with_function(__init_clustering)

        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))


@router.get(path='/embedding',
            response_model=BaseCommonModel,
            summary='Train a model using existing data with embedding',
            description='Training will only occur when sample data is available.'
                        'This API manually executes the training of '
                        'a machine learning model using existing data with embedding.'
                        'It allows the model to learn patterns and make predictions based on the provided data',
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
async def init_embedding():
    logger.debug("__init_embedding start")

    try:
        await loop_with_function(__init_embedding)

        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))


if __name__ == "__main__":
    # embedding train test

    # 1.1. embedding training
    # __init_embedding()

    # 1.2. use embedding
    # from app.routers.embedding_router import embedding_recommend

    # document_id = '66e39180-b276-484c-aa19-34efa560f32b'

    # recommended = embedding_recommend(document_id)

    # if ["9e075fd8-af50-4e20-bb58-a88c6c4155d0", "1449562c-a3f5-4a91-9c93-604731174e3d",
    #     "66e39180-b276-484c-aa19-34efa560f32b", "0672a85d-440a-4a30-b19e-ffad8b3b92f5",
    #     "259aedab-a68e-4ca9-b6d0-0b5cd6b21169"] != recommended.data.recommended_id:
    #     print('Error Occurred')
    # else:
    #     print('Successful')

    # 2.1. clustering training
    __init_clustering()

    # 2.2. use clustering
    # from app.routers.clustering_router import clustering_recommend

    # document_id = '66e39180-b276-484c-aa19-34efa560f32b'

    # recommended = clustering_recommend(document_id)

    # 3.1. cataloging training -> clustering training 대체 가능
    # from app.routers.catalog_router import catalog_recommend
