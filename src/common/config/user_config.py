import logging
import os
import re
import socket

from common.config.base_config import BaseConfig
from common.errors import ConfigException


class _Server:
    host = "127.0.0.1"
    port = 9999
    workers = 1
    process_timeout = 60
    download_encode = "utf-8"
    uvicorn = {}


class _Log:
    level = "INFO"
    logfile = "logs/jaguar.log"

    def validation_check(self):
        self.level = self.level.upper()
        if self.level.lower() not in ['info', 'debug', 'error']:
            raise ConfigException('log.level', self.level, 'not useful value')


class _Session:
    host = "0.0.0.0"
    port = 6379
    timeout = 10
    password = None


class _ExecuteForkServer:
    host = "0.0.0.0"
    port = 9009


class _ExecutePreForkServer:
    host = "0.0.0.0"
    port = 29009
    worker = 3
    reuse_count = 100


class _LogServer:
    host = "0.0.0.0"
    port = 19009


class _JobManagement:
    host = "0.0.0.0"
    port = 9002
    grpc_port = 3301
    required = False
    debug = False


class _MetaStoreServer:
    host = "0.0.0.0"
    port = 3306
    user = ""
    password = ""
    dbname = ""


class _DataModelManagement:
    host = "0.0.0.0"
    grpc_port = 2226


class _FileManager:
    host = "http://0.0.0.0"
    port = 33016
    fms_check_file_cmd = "/iris-file-management/v1/file-meta/"


class _Minio:
    host = "0.0.0.0"
    port = 30333
    user = "test"
    password = "test"
    bucket = "test"


class _Join:
    maximum_size = "2GB"
    _maximum_size_int = 0

    def get_maximum_size(self):
        self.maximum_size = self.maximum_size.upper()

        pattern = re.compile(r'^(-?\d+)(GB|MB|KB|B)?$', flags=re.IGNORECASE)
        matched = pattern.match(self.maximum_size)
        num, unit = matched.groups()
        num = int(num)
        if unit.upper() == 'GB':
            self._maximum_size_int = num * 1024 * 1024 * 1024
        elif unit.upper() == 'MB':
            self._maximum_size_int = num * 1024 * 1024
        elif unit.upper() == 'KB':
            self._maximum_size_int = num * 1024
        else:
            self._maximum_size_int = num

        return self._maximum_size_int


class _Sentry:
    enable = False
    dsn = ""  # DSN include token of Sentry and url
    level = logging.INFO  # Capture info and above as breadcrumbs. Ex: level=logging.INFO(20)
    event_level = logging.ERROR  # Send errors as events. Ex: event_level=logging.ERROR(40)
    traces_sample_rate = 0.1  # 0 to not send this trace to Sentry. Between 0 and 1 to send this trace to Sentry.
    traces_exclude_urls = []  # exclude url list, to do not send the trace of urls to Sentry
    debug = False
    environment = ""


class _ProcessInfo:
    pod_id = socket.gethostname()
    app_id = os.getpid()


class _RSAFile:
    path = "/app/ssl"


class _IMA:  # 현재 ima 에만 적용된 것으로 나중에 삭제 가능성 있음
    limit = 10000  # resource.service_name=ima 인 경우, sql 쿼리에 항상 limit 조건을 주어 db 에서 가져오는 데이터량 제한


class _Internal:
    # DB 에서 한 번에 가져오는 row 개수
    # 한 번에 모든 데이터를 로드하는 것을 방지하여 시스템이 memory overflow 가 되는 경우를 막는다. (default: 5000)
    batch_rows = 5000

    # DB 에서 가져오는 데이터의 최대 사이즈, (GB, MB, KB, B) 로 사용 가능 (각 단위 1024 배)
    # 제한 없음은 -1 로 표기 (default: 200MB)
    # note: python object size 가 추가되기 때문에 실제 데이터 사이즈 보다 적은 사이즈를 가져 올 수 있다.
    maximum_size = '200MB'
    _maximum_size_bytes = 200 * 1024 * 1024  # 200MB
    _system_maximum_size = 10 * 1024 * 1024 * 1024  # 10GB, 내부 최대 설정 가능 사이즈

    analyze_max_rows = 10000  # kmenas, outlier 명령어의 최대 row 개수, 넘어가면 임의로 limit 제한 한다

    def get_maximum_size(self):
        return self._maximum_size_bytes

    def validation_check(self):
        if isinstance(self.maximum_size, int):  # -1 세팅 시, 최대 사이즈로 세팅
            self._maximum_size_bytes = self._system_maximum_size
        else:
            self.maximum_size = self.maximum_size.upper()

            pattern = re.compile(r'^(-?\d+)(GB|MB|KB|B)?$', flags=re.IGNORECASE)
            matched = pattern.match(self.maximum_size)
            if not matched:
                raise ConfigException('internal.maximum_size', self.maximum_size, 'is not useful value')

            num, unit = matched.groups()
            num = int(num)
            if num < 0:  # -1(unit) 세팅 시, 최대 사이즈로 세팅
                self._maximum_size_bytes = self._system_maximum_size
            else:
                if unit.upper() == 'GB':
                    _maximum_size_bytes = num * 1024 * 1024 * 1024
                elif unit.upper() == 'MB':
                    _maximum_size_bytes = num * 1024 * 1024
                elif unit.upper() == 'KB':
                    _maximum_size_bytes = num * 1024
                else:
                    _maximum_size_bytes = num

                self._maximum_size_bytes = min(_maximum_size_bytes, self._system_maximum_size)


class _External:
    """
    외부에 공개 되어 사용 되는 폴더의 path
    주로 udf 명령어에서 사용된다
    path의 마지막은 udf_file 로 고정 : 그렇지 않으면 python 이 실행될 때 path 를 제대로 찾을수 없음
    """
    path = "/mount/tmp/jaguar/udf_file"


class Config(BaseConfig):
    # TODO Recommender에 맞는 default Config로의 설정 필요
    log = _Log()
    server = _Server()
    session = _Session()
    execute_fork_server = _ExecuteForkServer()
    execute_prefork_server = _ExecutePreForkServer()
    log_server = _LogServer()
    job_management = _JobManagement()
    metastore_server = _MetaStoreServer()
    datamodel_management = _DataModelManagement()
    join = _Join()
    sentry = _Sentry()
    process_info = _ProcessInfo()
    rsa_file = _RSAFile()
    ima = _IMA()
    internal = _Internal()
    minio = _Minio()
    file_manager = _FileManager()


if __name__ == '__main__':
    configPath = "/Users/koseungbeom/git/Recommender/config_templates/config.yml"
    Config.init("/Users/koseungbeom/git/Recommender/config_templates/config.yml")

    # Config.init('/home/ubuntu1/mobigen/IRIS-Data-Source-Executor/conf/jaguar.yml')
    # print(Config.server.host)
    # print(type(Config.server.uvicorn), Config.server.uvicorn)
    # print(Config.log.level)
    # print(Config.internal.maximum_size)
    # print(Config.internal.batch_rows)
    # print(Config.internal.get_maximum_size())
    # print(Config.join.maximum_size)
    # print(Config.join.get_maximum_size())