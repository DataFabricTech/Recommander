import logging
import os

from fastapi import APIRouter, Query

from app.models.response_model import BaseCommonModel, ErrorModel, RecommendationModel
from common.config import Config

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommend"),
    tags=['Recommend']
)


def __get_embedding_recommended_id(target_id: str) -> list:
    """
    Embedding ML의 결과값을 이용한 추천

    :param target_id: 추천을 받고자 하는 데이터 모델의 id
    :return list: 추천 id
    """
    logger.debug('__get_recommended_id start')

    from app.models.recommend_classes import RecommenderWithDB
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(Config.database.get_database_url(), echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session.query(RecommenderWithDB).filter(RecommenderWithDB.id == target_id).first().recommend_id


@router.get(path="/embedding",
            response_model=BaseCommonModel,
            summary='Find the most similar id using embedding algorithm',
            description='** You can only retrieve results that were trained using available sample data **'
                        'This API uses machine learning results to find the most similar data to the currently '
                        'provided value',
            responses={
                200: {'description': 'Recommend id list returned', 'model': RecommendationModel},
                404: {"description": "No data found", "model": BaseCommonModel}
            })
async def embedding_recommend(target_id: str = Query(..., description='유사한 데이터를 찾기 위한 데이터의 id 값')):
    from common.async_loop import loop_with_function
    logger.debug("embedding_recommend received request")

    try:
        found_id_list = await loop_with_function(__get_embedding_recommended_id, target_id)

        if len(found_id_list) == 0:
            logger.info("embedding result is nothing")
            return BaseCommonModel(status=404, error=ErrorModel(detail=f'No data found for {target_id}'))
        return BaseCommonModel(status=200, data=RecommendationModel(recommended=found_id_list))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=f'Error Occurred by {e}'))


if __name__ == "__main__":
    print()
