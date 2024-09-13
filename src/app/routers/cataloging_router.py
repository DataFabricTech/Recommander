import logging
import os

from fastapi import APIRouter, Query

from app.models.response_model import BaseCommonModel, ErrorModel, RecommendationModel
from common.config import Config

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommend"),
    tags=['Recommend'],
)


def __get_catalog_recommend_id(target_id: str, table_type: bool) -> str:
    """
    Clustering ML의 결과값의 대표값을 이용한 추천

    :param target_id:  카탈로그 추천을 받고자 하는 데이터 모델의 id
    :return str: 카탈로그 이름
    """
    from app.services.open_metadata_service import get_document
    logger.debug('__get_catalog_recommend_id start')

    model_path = Config.recommend_settings.clustering.trained_model_path
    if not os.path.exists(model_path):
        logger.info("not exist catalog model")
        return ""

    try:
        import pickle
        from hdbscan import prediction
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import pandas as pd

        with open(model_path + '/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        df = pd.read_csv(model_path + '/representative_values.csv')
        exist_document_text = df['document_text'].tolist()

        new_document = get_document(target_id, table_type)

        X_new = vectorizer.transform([document[1] for document in new_document.values()])

        rep_name: str = ''

        for i, new_doc in enumerate(new_document):
            similarities = cosine_similarity(X_new[i], vectorizer.transform(exist_document_text))

            most_similar_idx = np.argmax(similarities)
            rep_name = df['representative_name'][most_similar_idx]

        return rep_name
    except ValueError as e:
        logger.error(f'Value exception: {e}')
        raise
    except Exception as e:
        logger.error(f'Unknown exception: {e}')
        raise


@router.get(path="/cataloging",
            response_model=BaseCommonModel,
            summary='Find the best catalog',
            description='This API uses machine learning results to find the catalog to the currently '
                        'provided value',
            responses={
                200: {'description': 'Recommend name', 'model': RecommendationModel},
                404: {"description": "No data found", "model": BaseCommonModel}
            })
async def catalog_recommendation(target_id: str = Query(..., description='카탈로그를 찾기위한 데이터의 id 값')
                                 ,
                                 table_type: bool = Query(True, description='데이터의 타입 (table- True, storage - False)')):
    logger.debug('catalog_recommendation received request')

    try:
        from common.async_loop import loop_with_function
        found = await loop_with_function(__get_catalog_recommend_id, target_id, table_type)

        if found == '':
            logger.info("cataloging result is nothing")
            return BaseCommonModel(status=404, error=ErrorModel(detail=f'No data found for {target_id}'))
        return BaseCommonModel(status=200, data=RecommendationModel(recommended=found))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=f'Error Occurred by: {e}'))
