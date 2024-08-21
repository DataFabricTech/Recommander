import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from sklearn.metrics.pairwise import cosine_similarity

from app.models.response_model import RecommendationModel, ErrorModel
from app.services.open_metadata_service import extract_text_from_table_json, get_data

logging = logging.getLogger()

router = APIRouter(
    prefix=os.path.join("/recommender/predict"),
    tags=['ML Predict'],
)


def __find_similar_data(data):
    '''
    Clustering된 데이터를 이용한 데이터 추천 기능
    '''
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    df_existing = pd.read_csv('hdbscan_clusters.csv')
    existing_documents = df_existing['document'].tolist()
    existing_fqn = df_existing['fqn'].tolist()

    new_document = [extract_text_from_table_json(json.loads(data))[0]]

    x_new = vectorizer.transform(new_document)

    return existing_documents, existing_fqn, new_document, vectorizer, x_new


def __find_most_similar_data(fqn, table_type):
    most_similar_idx = -1
    data = get_data(fqn, table_type)

    existing_documents, existing_fqn, new_document, vectorizer, x_new = __find_similar_data(data)

    for i, new_doc in enumerate(new_document):
        similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))
        most_similar_idx = np.argmax(similarities)

    return existing_fqn[most_similar_idx] if most_similar_idx is not -1 else None


def __find_top_n_similar_documents(data):
    from common.config import Config
    existing_documents, existing_fqn, new_document, vectorizer, x_new = __find_similar_data(data)
    top_n_fqn = []
    for i, new_doc in enumerate(new_document):
        similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))
        top_indices = np.argsort(similarities.flatten())[-Config.open_metadata.top_n:]
        for idx in list(map(int, reversed(top_indices))):
            top_n_fqn.append(existing_fqn[idx])

    return top_n_fqn


@router.post(path="/api/recommend",
             response_model=RecommendationModel,
             summary='Find the most similar',
             description='This API uses machine learning results to find the most similar data to the currently '
                         'provided value',
             tags=['recommendation'],
             responses={
                 404: {"description": "No data found", "model": ErrorModel}
             })
def recommendation_data_model(
        fqn: str = Query(..., description='유사한 데이터를 찾기 위한 데이터의 fqn 값'),
        table_type: bool = Query(True, description='찾으려 하는 data의 type (table - True, storage - false)')):
    found_fqn = __find_most_similar_data(fqn, table_type)
    if found_fqn == -1:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f'No data found with {fqn}')
    return RecommendationModel(status=200, fqn=found_fqn)
