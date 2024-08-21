import os

from fastapi import APIRouter

from app.models.response_model import BaseCommonModel, MessageModel

router = APIRouter(
    prefix=os.path.join("/api/recommender/status"),
    tags=['Check'],
)


@router.get(path='',
            response_model=BaseCommonModel,
            summary='Check the status of the recommender',
            description='Check the status of the recommender',
            tags=['Status'])
def get_status():
    return BaseCommonModel(status=200,
                           data=MessageModel(message='Status OK'))

