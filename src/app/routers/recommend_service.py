import logging
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from sklearn.metrics.pairwise import cosine_similarity

from app.models.response_model import BaseCommonModel, ErrorModel, RecommendationModel
from app.services.open_metadata_service import extract_text_from_table_json, get_data
from common.config import Config

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/predict"),
    tags=['ML Predict'],
)


def __find_similar_data(data):
    '''
    Clustering된 데이터를 이용한 데이터 추천 기능
    :returns existring_document 클러스터링된 데이터들
             existring_fqn 클러스터링된 데이터들의 fqn
             new_document 추천시스템을 사용하고자 하는 데이터
             vectorizer 클러스터링된 데이터들의 vectorizer
             x_new vectorizer를 이용한 값
    '''
    try:
        with open(Config.open_metadata.trained_model_path + '/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        df_existing = pd.read_csv(Config.open_metadata.trained_model_path + '/hdbscan_clusters.csv')
        existing_documents = df_existing['document'].tolist()
        existing_fqn = df_existing['fqn'].tolist()

        new_document = [extract_text_from_table_json(data)[0]]

        x_new = vectorizer.transform(new_document)

        return existing_documents, existing_fqn, new_document, vectorizer, x_new
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except (IOError, OSError) as e:
        logging.error(f"File read/write error: {e}")
        raise
    except KeyError as e:
        logging.error(f"CSV format error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data transformation error: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Vectorization error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def __find_most_similar_data(fqn, table_type):
    most_similar_idx = -1
    data = get_data(fqn, table_type)

    existing_documents, existing_fqn, new_document, vectorizer, x_new = __find_similar_data(data)

    try:
        for i, new_doc in enumerate(new_document):
            similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))

            if similarities.size == 0:
                raise ValueError("No similarities")
            most_similar_idx = np.argmax(similarities)

        return existing_fqn[most_similar_idx] if most_similar_idx != -1 else None
    except ValueError as e:
        logger.error(f"Vectorization error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error {e}")
        raise

# todo table_Type 추가 및 data -> fqn으로 변경 필요
# todo 작동 Test 및 bug fix
def __find_top_n_similar_documents(data):
    existing_documents, existing_fqn, new_document, vectorizer, x_new = __find_similar_data(data)
    top_n_fqn = []
    for i, new_doc in enumerate(new_document):
        similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))
        top_indices = np.argsort(similarities.flatten())[-Config.open_metadata.top_n:]
        for idx in list(map(int, reversed(top_indices))):
            top_n_fqn.append(existing_fqn[idx])

    return top_n_fqn


# todo 비동기 API로의 변경 필요
@router.get(path="/cluster_recommend",
            response_model=BaseCommonModel,
            summary='Find the most similar',
            description='This API uses machine learning results to find the most similar data to the currently '
                        'provided value',
            tags=['recommendation'],
            responses={
                404: {"description": "No data found", "model": BaseCommonModel}
            })
def cluster_recommend(
        fqn: str = Query(..., description='유사한 데이터를 찾기 위한 데이터의 fqn 값'),
        table_type: bool = Query(True, description='찾으려 하는 data의 type (table - True, storage - false)')):
    logger.debug('cluster_recommendation received request')
    try:
        # todo found_fqn을 top_n으로 바꿔어야한다.
        found_fqn = __find_most_similar_data(fqn, table_type)

        if found_fqn == -1:
            return BaseCommonModel(status=404, error=ErrorModel(detail=f'No data found for {fqn}'))
        # todo 이거 list로 return 하게끔 바꿔야겠지?
        return BaseCommonModel(status=200, data=RecommendationModel(fqn=found_fqn))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=f'Error Occurred by {e}'))
