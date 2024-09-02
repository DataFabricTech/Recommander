import logging

import pandas as pd
import requests

import json

from app.init.config import Config
from app.models.dictionary_enum import DictionaryKeys

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
    texts.append(json_data[DictionaryKeys.NAME])
    texts.append(json_data[DictionaryKeys.DESCRIPTION]) if DictionaryKeys.DESCRIPTION in json_data else None
    texts.append(json_data[DictionaryKeys.SERVICE][DictionaryKeys.DISPLAY_NAME])

    for column in json_data[DictionaryKeys.COLUMNS]:
        texts.append(column[DictionaryKeys.NAME])
        # texts.append(column['dataType'])
        texts.append(column[DictionaryKeys.DESCRIPTION]) if DictionaryKeys.DESCRIPTION in column else None
        for tag in column.get(DictionaryKeys.TAGS, []):
            texts.append(tag[DictionaryKeys.TAG_FQN])
            texts.append(tag[DictionaryKeys.DESCRIPTION])

    for constraint in json_data.get(DictionaryKeys.TABLE_CONSTRAINTS, []):
        texts.append(constraint[DictionaryKeys.CONSTRAINT_TYPE])
        texts.extend(constraint[DictionaryKeys.COLUMNS])
        texts.extend(constraint.get(DictionaryKeys.REFERRED_COLUMNS, []))

    for tag in json_data.get(DictionaryKeys.TAGS, []):
        texts.append(tag[DictionaryKeys.TAG_FQN])
        texts.append(tag[DictionaryKeys.DESCRIPTION])

    return ' '.join(texts), json_data[DictionaryKeys.FULLY_QUALIFIED_NAME]


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


def get_sample(fqn: str, table_type: bool):
    logger.debug("get open metadata sample")
    url = (f"{(Config.open_metadata.get_tables_sample_url(fqn)
               if table_type else Config.open_metadata.get_storages_sample_url(fqn))}"
           f"{fqn.strip('\"') if table_type else '\"' + fqn.strip('\"') + '\"'}")
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.request("GET", url, headers=header, data={})
    response_tables = json.loads(response.text)
    sample = {}
    for json_data in response_tables['data']:
        json_ = json.loads(
            requests.request("GET", url,
                             headers=header, data={}).text)

        if DictionaryKeys.SAMPLE_DATA not in json_:
            continue

        rows = json_[DictionaryKeys.SAMPLE_DATA][DictionaryKeys.ROWS]

        columns = json_[DictionaryKeys.SAMPLE_DATA][DictionaryKeys.COLUMNS]
        sample = {}
        df = pd.DataFrame(rows, columns=columns)

        for idx, column in enumerate(columns):
            sample[column] = {DictionaryKeys.VALUES: sorted(df[column].values, key=lambda x: (x is None, x)),
                              DictionaryKeys.DATA_TYPE: json_[DictionaryKeys.COLUMNS][idx][DictionaryKeys.DATA_TYPE]}

        sample[json_[DictionaryKeys.ID]] = {DictionaryKeys.COLUMNS: sample}

    return sample
