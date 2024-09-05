import logging
import os

from fastapi import APIRouter

from app.models.response_model import BaseCommonModel, MessageModel

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/status"),
)


@router.get(path='',
            response_model=BaseCommonModel,
            summary='Check the status of the recommender',
            description='Check the status of the recommender',
            tags=['Status'])
async def get_status():
    logger.debug("Check the status of the recommender")
    return BaseCommonModel(status=200,
                           data=MessageModel(message='Status OK'))
