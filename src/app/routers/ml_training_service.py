import logging
import os

from fastapi import APIRouter

logging = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/recommender/training"),
    tags=['ML training service'],
)

def __init_clustering():
    '''
    주기적인 기존 데이터 Clustering
    :return:
    '''
    documents, fqns = get_document()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
    labels = hdbscan_clusterer.fit_predict(X)

    df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})
    df.to_csv('hdbscan_clusters.csv', index=False)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)



# todo - get -> post
@router.get(path="/api/predict_data", tags=['ML training service'], responses={400: {"message": "error message"}})
def recommendation_data_model():
    return {"status": "success"}



