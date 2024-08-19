import os

from fastapi import APIRouter
from typing import List

router = APIRouter(
    prefix=os.path.join("/recommender/check"),
    tags=['Check'],
)

class ResponseModel():
    # todo - APIRouter의 문서화를 위한 임시 구조체
    pass


# todo - router에 대한 자세한 설명 필요 -> fastapi + swagger를 이용한 문서화 작업
@router.get(path='/api/status',
            response_model=List[ResponseModel], # todo
            summary='Check the status of the recommender',
            description='Check the status of the recommender',
            tags=['Status'],
            responses={404: {"description": "Not found"}}) # todo 설명 말고도, "model": ErrorResponseModel을 줄 수도 있다.
def get_status(): # todo - do not change to async, 간단한 작업까지 비동기로 할 필요는 없다.
    # todo - message 구조체 생성 필요
    return {'message': 'Hello'}
