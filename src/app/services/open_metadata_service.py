import logging

import requests

import json

from app.init.config import Config

'''
TfidfVectorizer().fit_transform() - `train data`에 사용해야하는 함수 (label(?)을 만드는 작업)
TfidfVectorizer().transform() - train data를 이용하여 `test data`에 적용하는 함수 (만들어진 label(?)을 사용하는 작업)

clustering은 memory가 많이 드는 작업으로, 주기적(새벽 시간대)를 이용한 Clustering 구축하는 것이 좋아보인다. 
'''

logger = logging.getLogger()


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
    logger.debug("get open metadata token")

    import base64

    url = Config.open_metadata.get_login_url()
    try:
        payload = json.dumps({
            "email": Config.open_metadata.id,
            "password": base64.b64encode(Config.open_metadata.pw.encode('utf-8')).decode('utf-8')
        })
    except (AttributeError, TypeError) as e:
        logger.error(f"base64 encode error: {e}")
        raise

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        return json.loads(response.text)['accessToken']
    except requests.exceptions.RequestException as e:
        logger.error(f"request exception: {e}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Json load error: {e}")
        raise


def get_documents():
    '''
    등록되어 있는 데이터 모델을 가져오는 함수

    :return: documents:List[str] - list of data model, fqn:str - list of data model's fqn
    '''
    logger.debug("get open metadata document")

    url = Config.open_metadata.get_document_url()
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    documents = []
    fqns = []

    try:
        response = requests.request("GET", url, headers=header, data={})
        tables = json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException error: {e}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Json load error: {e}")
        raise

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
    logger.debug("get open metadata data")

    url = (f"{(Config.open_metadata.get_table_url() if table_type else Config.open_metadata.get_storage_url())}"
           f"{fqn.strip('\"') if table_type else '\"' + fqn.strip('\"') + '\"'}")
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.request("GET", url, headers=header, data={})
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"request exception: {e}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Json load error: {e}")
        raise
