import logging
import os
from pickle import PickleError

from fastapi import APIRouter

from app.models.response_model import BaseCommonModel, ErrorModel, MessageModel

logger = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/api/recommender/training"),
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

    try:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(documents)
    except (ValueError, TypeError) as e:
        logger.error(f"Vectorization error: {e}")
        raise

    try:
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=Config.open_metadata.min_cluster_size, metric='cosine')
        labels = hdbscan_clusterer.fit_predict(x)
    except ValueError as e:
        logger.error(f"HDBSCAN error: {e}")
        raise

    try:
        df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})

        if not os.path.exists(Config.open_metadata.trained_model_path):
            os.makedirs(Config.open_metadata.trained_model_path)

        df.to_csv(Config.open_metadata.trained_model_path + '/hdbscan_clusters.csv', index=False)

        with open(Config.open_metadata.trained_model_path + '/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    except (FileNotFoundError, IOError, PickleError) as e:
        logger.error(f"File write Error: {e}")
        raise


@router.get(path='',
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
        return BaseCommonModel(status=200, data=MessageModel(message='Successfully trained'))
    except Exception as e:
        return BaseCommonModel(status=404, error=ErrorModel(detail=str(e)))
