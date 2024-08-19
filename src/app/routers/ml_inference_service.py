import logging
import os

from fastapi import APIRouter

logging = logging.getLogger()

router = APIRouter(
    # todo - prefix 필요
    prefix=os.path.join("/todo"),
    tags=['ML Inference'],
)

# todo - router에 대한 자세한 설명 필요 -> fastapi + swagger를 이용한 문서화 작업
# todo - get -> post
@router.get(path="/api/recommend", tags=['ML Inference'], responses={400: {"message": "error message"}})
def recommendation_data_model():
    return {"status": "success"}