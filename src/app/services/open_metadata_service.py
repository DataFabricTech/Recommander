import requests
import pickle

import json
import hdbscan
import pandas as pd

from app.init.config import Config

from sklearn.feature_extraction.text import TfidfVectorizer

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
    url = Config.open_metadata.login_url

    payload = json.dumps({
        "email": Config.open_metadata.id,
        "password": Config.open_metadata.pw
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)['accessToken']


def get_documents():
    '''
    등록되어 있는 데이터 모델을 가져오는 함수

    :return: documents:List[str] - list of data model, fqn:str - list of data model's fqn
    '''
    url = Config.open_metadata.document_url
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


def get_data(fqn: str, table_type: bool):
    '''
    새로운 데이터를 가져오는 함수
    :return: data: json
    '''
    url = (f"{(Config.open_metadata.table_url if table_type else Config.open_metadata.storage_url)}"
           f"{fqn.strip('\"') if table_type else '\"' + fqn.strip('\"') + '\"'}")
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.request("GET", url, headers=header, data={})
    return json.loads(response.text)


def init_clustering():
    '''
    기존 데이터를 이용한 clustering 생성
    :return:
    '''

    documents, fqns = get_documents()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=Config.open_metadata.min_cluster_size, metric='cosine')
    labels = hdbscan_clusterer.fit_predict(X)

    df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})
    df.to_csv('hdbscan_clusters.csv', index=False)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
