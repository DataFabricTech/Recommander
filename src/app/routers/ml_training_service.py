import logging
import os

from fastapi import APIRouter

from app.models.response_model import BaseCommonModel, ErrorModel

logging = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/recommender/training"),
    tags=['ML training service'],
)


def __init_clustering():
    '''
    기존 데이터를 이용한 clustering 생성
    '''
    import pickle
    import hdbscan
    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from app.services.open_metadata_service import get_documents

    from common.config import Config

    documents, fqns = get_documents()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=Config.open_metadata.min_cluster_size, metric='cosine')
    labels = hdbscan_clusterer.fit_predict(X)

    df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})
    df.to_csv('hdbscan_clusters.csv', index=False)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


@router.post(path='/api/predict_data',
             response_model=BaseCommonModel,
             summary='Train a model using existing data',
             description='This API manually executes the training of a machine learning model using existing data. '
                         'It allows the model to learn patterns and make predictions based on the provided data',
             tags=['ML Training'],
             responses={404: {"description": "Train a model is fail", "model": ErrorModel}}
             )
def recommendation_data_model():
    try:
        __init_clustering()
        return BaseCommonModel(status=200)
    except Exception as e:
        return ErrorModel(status=404, detail=str(e))
