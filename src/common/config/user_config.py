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
    document_limit = '1000000'
    table_sample_limit = '1000000'
    min_cluster_size = 2
    top_n = 5
    trained_model_path = './trained_models/'

    def get_table_url(self):
        return 'http://{}:{}/api/v1/tables/name/'.format(self.host, self.port)

    def get_storage_url(self):
        return 'http://{}:{}/api/v1/containers/name/'.format(self.host, self.port)

    def get_document_url(self):
        return 'http://{}:{}/api/v1/tables?limit={}'.format(self.host, self.port, self.document_limit)

    def get_login_url(self):
        return 'http://{}:{}/api/v1/users/login'.format(self.host, self.port)

    def get_tables_url(self):
        return ('http://{}:{}/api/v1/tables?limit={}&include=non-deleted'
                .format(self.host, self.port, self.table_sample_limit))

    def get_tables_sample_url(self, fqn: str):
        return 'http://{}:{}/api/v1/tables/{}/sampleData'.format(self.host, self.port, fqn)

    def get_storages_sample_url(self, fqn: str):
        return 'http://{}:{}/api/v1/containers/{}/sampleData'.format(self.host, self.port, fqn)


class _Embedding:
    queue_max_size = 5
    top_similarity_threshold = 5



class Config(BaseConfig):
    log = _Log()
    server = _Server()
    log_server = _LogServer()  # todo check
    sentry = _Sentry()  # todo check

    cron = _Cron()
    open_metadata = _OpenMetadata()
    embedding = _Embedding()


if __name__ == '__main__':
    configPath = "/Users/koseungbeom/git/Recommender/config_templates/config.yml"
    Config.init("/Users/koseungbeom/git/Recommender/config_templates/config.yml")

    # print(Config.server.host)
    # print(type(Config.server.uvicorn), Config.server.uvicorn)
    # print(Config.log.level)
    # print(Config.internal.maximum_size)
    # print(Config.internal.batch_rows)
    # print(Config.internal.get_maximum_size())
    # print(Config.join.maximum_size)
    # print(Config.join.get_maximum_size())
