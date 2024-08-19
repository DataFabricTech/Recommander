import requests
import pickle

import numpy as np
import json
import hdbscan
import pandas as pd

from app.init.config import Config

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from open_metadata_service import get_document

'''
TfidfVectorizer().fit_transform() - `train data`에 사용해야하는 함수 (label(?)을 만드는 작업)
TfidfVectorizer().transform() - train data를 이용하여 `test data`에 적용하는 함수 (만들어진 label(?)을 사용하는 작업)

clustering은 memory가 많이 드는 작업으로, 주기적(새벽 시간대)를 이용한 Clustering 구축하는 것이 좋아보인다. 
'''


def extract_text_from_table_json(json_data):
    '''
    json 데이터에서 머신러닝 돌릴 때 필요한 데이터를 추출하는 함수
    :param json_data: 파싱할 json데이터
    :return: str, fqn
    '''
    texts = []
    texts.append(json_data['name'])
    texts.append(json_data['description']) if 'description' in json_data else None
    texts.append(json_data['service']['displayName'])

    for column in json_data['columns']:
        texts.append(column['name'])
        # texts.append(column['dataType'])
        texts.append(column['description']) if 'description' in column else None
        for tag in column.get('tags', []):
            texts.append(tag['tagFQN'])
            texts.append(tag['description'])

    for constraint in json_data.get('tableConstraints', []):
        texts.append(constraint['constraintType'])
        texts.extend(constraint['columns'])
        texts.extend(constraint.get('referredColumns', []))

    for tag in json_data.get('tags', []):
        texts.append(tag['tagFQN'])
        texts.append(tag['description'])

    return ' '.join(texts), json_data['fullyQualifiedName']


def get_token():
    '''
    오픈메타데이터의 토큰

    :return: token: string
    '''
    # todo open metadata에 대한 config 추가 (host, port, login_url, document_url, email, password, limit)
    url = f"http://{Config.OPEN_METADATA_HOST}:{Config.OPEN_METADATA_PORT}/{Config.OPEN_METADATA_LOGIN_URL}"

    payload = json.dumps({
        "email": Config.OPEN_METADATA_ID,
        "password": Config.OPEN_METADATA_PW
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)['accessToken']


def get_document():
    '''
    등록되어 있는 데이터 모델을 가져오는 함수

    :return: documents:List[str] - list of data model, fqn:str - list of data model's fqn
    '''
    url = (f"http://{Config.OPEN_METADATA_HOST}:{Config.OPEN_METADATA_PORT}/{Config.OPEN_METADATA_DOCUMENT_URL}"
           f"?limit={Config.OPEN_METADATA_LIMIT}")
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.request("GET", url, headers=header, data={})
    tables = json.loads(response.text)
    documents = []
    fqns = []
    for json_data in tables['data']:
        document, fqn = extract_text_from_table_json(json_data)
        documents.append(document)
        fqns.append(fqn)

    return documents, fqns


# todo scheduler 필요
def init_clustering():
    '''
    기존 데이터를 이용한 clustering 생성
    :return:
    '''
    documents, fqns = get_document()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # todo min_cluster_size를 config로 받아 드리게끔 변경 필요
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
    labels = hdbscan_clusterer.fit_predict(X)

    df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})
    df.to_csv('hdbscan_clusters.csv', index=False)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


# todo 새로운 데이터를 매개변수로 받아야한다. new_data -> arg
# todo fqn에 대한 return이 들어가야한다.
def find_similar_data():
    '''
    Clustering된 데이터를 이용한 데이터 추천 기능
    '''
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    df_existing = pd.read_csv('hdbscan_clusters.csv')
    existing_documents = df_existing['document'].tolist()
    existing_fqn = df_existing['fqn'].tolist()

    new_document = [extract_text_from_table_json(json.loads(new_data))[0]]

    x_new = vectorizer.transform(new_document)
    top_n = 5

    for i, new_doc in enumerate(new_document):
        similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))
        most_similar_idx = np.argmax(similarities)
        top_indices = np.argsort(similarities.flatten())[-top_n:]

        print(f"new Document\n{new_doc}")
        print(f"\nMost Similar Existing Document\n{existing_documents[most_similar_idx]}")
        print(f"\nMost Similar Found Document's FQN\n{existing_fqn[most_similar_idx]}")
        print(f"\nTop Similar Exist Document's index: {list(map(int, reversed(top_indices)))}")
        print()
