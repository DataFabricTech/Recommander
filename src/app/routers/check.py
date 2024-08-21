import os

from fastapi import APIRouter

from app.models.response_model import BaseCommonModel

router = APIRouter(
    prefix=os.path.join("/recommender/check"),
    tags=['Check'],
)


@router.get(path='/api/status',
            response_model=BaseCommonModel,
            summary='Check the status of the recommender',
            description='Check the status of the recommender',
            tags=['Status'])
def get_status():
    return BaseCommonModel(status=200, message="status OK")
