import logging
import os
from fastapi import APIRouter, Query
from transformers import BertTokenizer, BertModel

from app.models.recommend_classes import RecommenderWithDB
from app.models.response_model import BaseCommonModel, ErrorModel, RecommendationModel
from src.app.services import open_metadata_service
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/app/recommender/embedding"),
    tags=['Embedding Service']
)


def __get_recommended_fqns(target_fqn: str) -> list:
    '''
    이미 정리되어 DB에 저장된 Recommender를 가져오기 위한 함수

    :param target_fqn: 가져오고자 하는 fqn
    :return list: 추천하는 id
    '''
    engine = create_engine('sqlite:///recommender.db', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    result = session.query(RecommenderWithDB).filter(RecommenderWithDB.id == target_fqn).first()

    # todo test
    return result


'''
# todo
1. 정기적으로 logic을 돌려야한다.
2. 정기적 logic을 통한 recommend
'''


@router.get(path="/embedding_recommend",
            response_model=BaseCommonModel,
            summary='Find the most similar',
            description='This API uses machine learning results to find the most similar data to the currently '
                        'provided value',
            tags=['recommendation'],
            responses={
                404: {"description": "No data found", "model": BaseCommonModel}
            })
def embedding_recommend(fqn: str = Query(..., description='유사한 데이터를 찾기 위한 데이터의 fqn 값')):
    logger.debug("embedding_recommend received request")
    try:
        found_fqns = __get_recommended_fqns(fqn)
        if len(found_fqns) == 0:
            return BaseCommonModel(status=404, error=ErrorModel(detail=f'No data found for {fqn}'))
        return BaseCommonModel(status=200, data=RecommendationModel(fqns=found_fqns))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=f'Error Occurred by {e}'))


if __name__ == "__main__":
    # KLUE-BERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    model = BertModel.from_pretrained('klue/bert-base')

    # sample_data 불러오기
    tables_sample = open_metadata_service.get_tables_sample()

    samples_include_embedding = get_bert_embeddings(tables_sample)

    # todo 주기적으로 돌려야하는 함수 - 비동기 작업 -> 수동 + 자동
    recommend_dict = process_recommend(samples_include_embedding)
    save_recommend_to_db(recommend_dict)

    # todo DB에 있는 Data를 활용한 추천 확인 -> api를 통한 검색 router 생성
    result = __get_recommended_fqns('094c5564-0995-49f2-8490-f969fcdaaa90')
