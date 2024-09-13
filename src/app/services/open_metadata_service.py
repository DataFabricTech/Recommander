import logging

import pandas as pd
import requests

import json

from app.init.config import Config
from app.models.dictionary_enum import DictionaryKeys

logger = logging.getLogger()


def extract_text_from_table_json(json_data: dict) -> (str, str):
    """
    원천 데이터 전처리 함수

    :param json_data: 원천 데이터
    :return: 전처리 데이터
    """
    texts = [json_data[DictionaryKeys.NAME.value]]
    texts.append(json_data[DictionaryKeys.DESCRIPTION.value]) if DictionaryKeys.DESCRIPTION.value in json_data else None
    texts.append(json_data[DictionaryKeys.SERVICE.value][DictionaryKeys.DISPLAY_NAME.value])

    for column in json_data.get(DictionaryKeys.COLUMNS.value, []):
        texts.append(column[DictionaryKeys.NAME.value])
        # texts.append(column['dataType'])
        texts.append(column[DictionaryKeys.DESCRIPTION.value]) if DictionaryKeys.DESCRIPTION.value in column else None
        for tag in column.get(DictionaryKeys.TAGS.value, []):
            texts.append(tag[DictionaryKeys.TAG_FQN.value])
            texts.append(tag[DictionaryKeys.DESCRIPTION.value])

    for constraint in json_data.get(DictionaryKeys.TABLE_CONSTRAINTS.value, []):
        texts.append(constraint[DictionaryKeys.CONSTRAINT_TYPE.value])
        texts.extend(constraint[DictionaryKeys.COLUMNS.value])
        texts.extend(constraint.get(DictionaryKeys.REFERRED_COLUMNS.value, []))

    for tag in json_data.get(DictionaryKeys.TAGS.value, []):
        texts.append(tag[DictionaryKeys.TAG_FQN.value])
        texts.append(tag[DictionaryKeys.DESCRIPTION.value])

    return ' '.join(texts), json_data[DictionaryKeys.ID.value]


def get_token():
    """
    오픈메타데이터의 토큰

    :return: token: string
    """
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


def get_documents(documents: dict, table_type: bool) -> dict:
    """
    등록되어 있는 데이터 모델을 가져오는 함수

    :return: documents:List[str] - list of data model, fqn:str - list of data model's fqn
    """
    logger.debug("get open metadata document")

    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    document_url = (
        f"{(Config.open_metadata.get_table_document_url()
            if table_type else Config.open_metadata.get_storage_document_url())}"
    )

    try:
        response = requests.request("GET", document_url, headers=header, data={})
        tables = json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException error: {e}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Json load error: {e}")
        raise

    for json_data in tables['data']:
        document, document_id = extract_text_from_table_json(json_data)
        documents[document_id] = (json_data[DictionaryKeys.NAME.value], document)

    return documents


def get_document(target_id: str, table_type: bool):
    """
    table/storage의 데이터를 가져오는 함수
    :return: data: json
    """
    logger.debug("get open metadata data")

    url = (f"{(Config.open_metadata.get_table_url() if table_type else Config.open_metadata.get_storage_url())}"
           f"{target_id.strip('\"') if table_type else '\"' + target_id.strip('\"') + '\"'}")
    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.request("GET", url, headers=header, data={})
        table = json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"request exception: {e}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Json load error: {e}")
        raise

    documents = {}

    if DictionaryKeys.CODE.value in table and table[DictionaryKeys.CODE.value] == 404:
        logger.error(f"No document")
        raise ValueError("No document found")

    document, document_id = extract_text_from_table_json(table)
    documents[document_id] = (table[DictionaryKeys.NAME.value], document)

    return documents



def __get_samples(samples: dict, table_type: bool) -> dict:
    """
    table/storage의 sample 데이터를 가져오는 함수

    :param samples: 이미 존재하는 sample의 dictionary
    :param table_type: table/storage를 구분하는 bool 값

    :return: 추가 sample
    """
    logger.debug('__get_sample start')

    token = get_token()
    header = {
        "Authorization": f"Bearer {token}"
    }

    document_url = (
        f"{(Config.open_metadata.get_table_document_url()
            if table_type else Config.open_metadata.get_storage_document_url())}"
    )

    sample_url = (
        f"{(Config.open_metadata.get_tables_sample_url()
            if table_type else Config.open_metadata.get_storages_sample_url())}"
    )

    response = requests.request("GET", document_url, headers=header, data={})
    try:
        response_tables = json.loads(response.text)

        for json_data in response_tables['data']:

            json_ = json.loads(
                requests.request("GET", sample_url.format(json_data['id'].strip('\"')), headers=header, data={}).text)

            if DictionaryKeys.SAMPLE_DATA.value not in json_:
                continue

            rows = json_[DictionaryKeys.SAMPLE_DATA.value][DictionaryKeys.ROWS.value]

            columns = json_[DictionaryKeys.SAMPLE_DATA.value][DictionaryKeys.COLUMNS.value]
            sample = {}
            df = pd.DataFrame(rows, columns=columns)

            for idx, column in enumerate(columns):
                sample[column] = {
                    DictionaryKeys.VALUES.value: sorted(df[column].values, key=lambda x: (x is None, str(x))),
                    DictionaryKeys.DATA_TYPE.value: json_[DictionaryKeys.COLUMNS.value][idx][
                        DictionaryKeys.DATA_TYPE.value] if table_type else
                    json_[DictionaryKeys.DATA_MODEL.value][DictionaryKeys.COLUMNS.value][idx][
                        DictionaryKeys.DATA_TYPE.value]}

            samples[json_[DictionaryKeys.ID.value]] = {DictionaryKeys.COLUMNS.value: sample}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        raise
    except KeyError as e:
        logger.error(f"Key exception: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value exception: {e}")
        raise
    except TypeError or AttributeError as e:
        logger.error(f"Type/Attribute exception: {e}")
        raise
    except IndexError as e:
        logger.error(f"Index exception: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")
        raise

    return samples


def get_samples() -> dict:
    return __get_samples(__get_samples({}, True), False)
