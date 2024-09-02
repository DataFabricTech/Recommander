import itertools
import logging
import os
from pickle import PickleError

import torch
from fastapi import APIRouter
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from app.models.recommend_classes import OverrodePriorityQueue, RecommendEntity, RecommenderWithDB, Base
from common.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.dictionary_enum import DictionaryKeys
import numpy as np
from itertools import combinations

from app.models.response_model import BaseCommonModel, ErrorModel, MessageModel

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/training"),
    tags=['ML training service'],
)


# todo 주기적으로 하는 로직 추가 요청
def __init_clustering():
    '''
    기존 데이터를 이용한 clustering 생성
    '''
    import pickle
    import hdbscan
    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from app.services.open_metadata_service import get_documents

    from common.config import Config

    documents, fqns = get_documents()

    try:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(documents)
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
        df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})

        if not os.path.exists(Config.open_metadata.trained_model_path):
            os.makedirs(Config.open_metadata.trained_model_path)

        df.to_csv(Config.open_metadata.trained_model_path + '/hdbscan_clusters.csv', index=False)

        with open(Config.open_metadata.trained_model_path + '/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    except (FileNotFoundError, IOError, PickleError) as e:
        logger.error(f"File write Error: {e}")
        raise


# todo 주기적으로 하는 로직 추가 요청
def __get_embedding(text: str):
    """
    :param text: Embedding Text
    :return: Embedding result
    """
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    model = BertModel.from_pretrained('klue/bert-base')
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def __get_bert_embeddings(tables_sample: dict):
    """

    :param tables_sample:
    :return:
    """
    '''

    :param samples:
    :return: samples
    {
        'fqn': {
            'columns': {
                'column1': {'values':['row_value'], 'dataType':'string', 'embedding': tensor([[]]),
                'column2': {'values':['row_value'], 'dataType':'string', 'embedding': tensor([[]]),
            }},
        'fqn2': ...
    }
    '''

    for key, value in tables_sample.items():
        for column_name, column_data in value[DictionaryKeys.COLUMNS].items():
            values_str = ' '.join(map(str, column_data[DictionaryKeys.VALUES]))
            column_data[DictionaryKeys.EMBEDDINGS_RESULT] = __get_embedding(values_str)

    return tables_sample


def process_recommend(sample_with_embeddings: dict):
    ids = sample_with_embeddings.keys()
    ids_combinations = list(combinations(ids, 2))
    recommend_dict = {}
    top_similarities_average = 0
    for source_id, comparison_id in ids_combinations:
        queue_max_size = Config.embedding.queue_max_size
        source_recommender = recommend_dict[source_id] if source_id in recommend_dict \
            else OverrodePriorityQueue(maxsize=queue_max_size)

        comparison_recommender = recommend_dict[comparison_id] if comparison_id in recommend_dict \
            else OverrodePriorityQueue(maxsize=queue_max_size)
        top_similarity = []
        inclusion_column_count = 0

        source_columns = sample_with_embeddings[source_id][DictionaryKeys.COLUMNS]
        comparison_columns = sample_with_embeddings[comparison_id][DictionaryKeys.COLUMNS]

        for source_column, comparison_column in list(itertools.product(source_columns, comparison_columns)):
            source_column_dict = source_columns[source_column]
            comparison_column_dict = comparison_columns[comparison_column]

            if source_column_dict[DictionaryKeys.DATA_TYPE] != comparison_column_dict[DictionaryKeys.DATA_TYPE]:
                continue

            source_value_dict = source_column_dict[DictionaryKeys.VALUES]
            comparison_value_dict = comparison_column_dict[DictionaryKeys.VALUES]

            if source_value_dict != [] and comparison_value_dict != [] and (
                    set(source_value_dict).issubset(set(comparison_value_dict))) or (
                    set(comparison_value_dict).issubset(set(source_value_dict))):
                inclusion_column_count += 1

            top_similarity.append(
                cosine_similarity(source_column_dict[DictionaryKeys.EMBEDDINGS_RESULT].numpy(),
                                  comparison_column_dict[DictionaryKeys.EMBEDDINGS_RESULT].numpy()))

        if top_similarity:
            top_similarities_average = float(
                np.mean(np.concatenate(sorted(top_similarity[:Config.embedding.top_similarity_threshold]))))
        source_recommender.put(RecommendEntity(comparison_id, inclusion_column_count, top_similarities_average))
        comparison_recommender.put(RecommendEntity(source_id, inclusion_column_count, top_similarities_average))
        recommend_dict[source_id] = source_recommender
        recommend_dict[comparison_id] = comparison_recommender

    return recommend_dict


def save_recommend_to_db(recommender: dict):
    '''
    정리된 recommender를 db에 넣는 함수

    :param recommender: 5가지의 오름차순으로 정렬되어 있는 추천
    :return:
    '''
    from sqlalchemy.orm import class_mapper

    engine = create_engine('sqlite:///recommender.db', echo=False)

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    bulk_recommender = []

    for key, value in recommender.items():
        list_ = []
        while not value.empty():
            list_.append(value.get().target_fqn)
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
    '''
    embedding 알고리즘을 응용한 유사도 측정 알고리즘을 활용한 추천
    '''


@router.get(path='/clustering',
            response_model=BaseCommonModel,
            summary='Train a model using existing data',
            description='This API manually executes the training of a machine learning model using existing data. '
                        'It allows the model to learn patterns and make predictions based on the provided data',
            tags=['ML Training'],
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
def init_clustering():
    try:
        __init_clustering()
        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))


# todo 설명 바꾸기
@router.get(path='/embedding',
            response_model=BaseCommonModel,
            summary='Train a model using existing data',
            description='This API manually executes the training of a machine learning model using existing data. '
                        'It allows the model to learn patterns and make predictions based on the provided data',
            tags=['ML Training'],
            responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
            )
def init_embedding():
    try:
        print()
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))
