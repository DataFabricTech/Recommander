import logging

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
    logfile = "logs/recommender.log"

    def validation_check(self):
        self.level = self.level.upper()
        if self.level.lower() not in ['info', 'debug', 'error']:
            raise ConfigException('log.level', self.level, 'not useful value')


class _LogServer:
    host = "0.0.0.0"
    port = 19009


class _Sentry:
    enable = False
    dsn = ""  # DSN include token of Sentry and url
    level = logging.INFO  # Capture info and above as breadcrumbs. Ex: level=logging.INFO(20)
    event_level = logging.ERROR  # Send errors as events. Ex: event_level=logging.ERROR(40)
    traces_sample_rate = 0.1  # 0 to not send this trace to Sentry. Between 0 and 1 to send this trace to Sentry.
    traces_exclude_urls = []  # exclude url list, to do not send the trace of urls to Sentry
    debug = False
    environment = ""


class _Cron:
    hour = '2'
    minute = '0'


class _OpenMetadata:
    host = 'localhost'
    port = '8080'
    id = 'root'
    pw = 'PASSWORD'
    limit = '1000000'
    min_cluster_size = 5
    top_n = 5

    def get_table_url(self):
        return 'http://{}:{}/api/v1/tables/name/'.format(self.host, self.port)

    def get_storage_url(self):
        return 'http://{}:{}/api/v1/containers/name/'.format(self.host, self.port)

    def get_table_document_url(self):
        return 'http://{}:{}/api/v1/tables?limit={}'.format(self.host, self.port, self.limit)

    def get_storage_document_url(self):
        return 'http://{}:{}/api/v1/containers?limit={}'.format(self.host, self.port, self.limit)

    def get_login_url(self):
        return 'http://{}:{}/api/v1/users/login'.format(self.host, self.port)

    def get_tables_url(self):
        return ('http://{}:{}/api/v1/tables?limit={}&include=non-deleted'
                .format(self.host, self.port, self.limit))

    def get_tables_sample_url(self):
        return 'http://{}:{}/api/v1/tables/{}'.format(self.host, self.port, '{}/sampleData')

    def get_storages_sample_url(self):
        return 'http://{}:{}/api/v1/containers/{}'.format(self.host, self.port, '{}/sampleData')


class _DataBase:
    dialects = 'postgresql'
    host = 'localhost'
    port = '8080'
    id = '<ID>'
    pw = '<PASSWORD>'
    db = '<DATABASE>'

    def get_database_url(self):
        return '{}://{}:{}@{}:{}/{}'.format(self.dialects, self.id, self.pw, self.host, self.port, self.db)


class _Clustering:
    trained_model_path = './trained_models/'


class _RecommendSettings:
    max_recommended_count = 5
    clustering = _Clustering()


class Config(BaseConfig):
    log = _Log()
    server = _Server()
    log_server = _LogServer()  # todo check
    sentry = _Sentry()  # todo check

    cron = _Cron()
    open_metadata = _OpenMetadata()
    database = _DataBase()
    recommend_settings = _RecommendSettings()
    clustering = _Clustering()


if __name__ == '__main__':
    pass
