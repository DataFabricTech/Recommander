import logging
import os

import pandas as pd
from fastapi import APIRouter, Query

from app.models.response_model import BaseCommonModel, ErrorModel, RecommendationModel
from common.config import Config

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommend"),
    tags=['Recommend'],
)


def __get_recommended_id(target_id: str) -> list:
    """
    Clustering ML의 결과값을 이용한 추천

    :param target_id: 추천을 받고자 하는 데이터 모델의 id
    :return list: 추천 id
    """
    logger.debug('__get_recommended_id start')

    if not os.path.exists(Config.clustering.trained_model_path):
        logger.info("not exist clustering model")
        return []

    try:
        df = pd.read_csv(Config.clustering.trained_model_path + '/hdbscan_clusters.csv')
        target_label = df.loc[df['id'] == target_id, 'labels'].values[0]

        df = df[(df['id'] != target_id) & (df['labels'] == target_label)]
        return df.sample(n=min(Config.recommend_settings.max_recommended_count, len(df)))['id'].tolist()
    except IndexError as e:
        logger.error(f'Index exception: {e}')
        raise
    except KeyError as e:
        logger.error(f'Key exception: {e}')
        raise
    except ValueError as e:
        logger.error(f'Value exception: {e}')
        raise
    except Exception as e:
        logger.error(f'Unknown exception: {e}')
        raise


@router.get(path="/clustering",
            response_model=BaseCommonModel,
            summary='Find the most similar id using clustering algorithm',
            description='This API uses machine learning results to find the most similar data to the currently '
                        'provided value',
            tags=['recommendation'],
            responses={
                200: {'description': 'Recommend id list returned', 'model': RecommendationModel},
                404: {"description": "No data found", "model": BaseCommonModel}
            })
def clustering_recommend(
        target_id: str = Query(..., description='유사한 데이터를 찾기 위한 데이터의 id 값')):
    logger.debug('cluster_recommendation received request')
    try:
        found_id_list = __get_recommended_id(target_id)

        if found_id_list == -1:
            logger.info("clustering result is nothing")
            return BaseCommonModel(status=404, error=ErrorModel(detail=f'No data found for {target_id}'))
        return BaseCommonModel(status=200, data=RecommendationModel(recommended_id_list=found_id_list))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=f'Error Occurred by: {e}'))
