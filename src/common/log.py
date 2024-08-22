import logging
import logging.handlers

from common.config import Config

log_level = Config.log.level.upper()
log_format = "[%(asctime)s] %(levelname)s, %(process)s-%(thread)d, %(filename)s(%(lineno)d), %(name)s, %(message)s"
formatter = logging.Formatter(log_format)
logfile = Config.log.logfile

file_max_byte = 1024 * 1024 * 10

log_server_host = Config.log_server.host
log_server_port = Config.log_server.port

"""
uvicorn 을 위한 log config
uvicorn 에서 사용되는 구조를 이용
"""
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": log_format,
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "[%(asctime)s] %(levelname)s, %(process)s-%(thread)d, %(message)s",
            "use_colors": None,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "udp": {
            "formatter": "default",
            "class": "logging.handlers.DatagramHandler",
            "host": log_server_host,
            "port": log_server_port,
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["udp"], "level": log_level, "propagate": False},
        "uvicorn.error": {"handlers": ["udp"], "level": log_level, "propagate": False},
        "uvicorn.access": {"handlers": ["udp"], "level": log_level, "propagate": False},
    },
}


def setup_logger(handler_name=None):
    """
    default logger 설정
    코드상에서 사용될 로그
    """
    if handler_name is None:
        log = logging.getLogger()
    else:
        log = logging.getLogger(handler_name)

    if log_level.lower() == 'error':
        log.setLevel(logging.ERROR)
    elif log_level.lower() == 'debug':
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    log.propagate = False

    log.handlers.clear()

    log.addHandler(logging.StreamHandler())
    for handler in log.handlers:
        handler.setFormatter(formatter)


setup_logger()
