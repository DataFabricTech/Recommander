import logging
import types

import yaml


class Struct(object):
    def __init__(self, default_obj, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                default_struct = None
                try:
                    default_struct = getattr(default_obj, k)
                except AttributeError:
                    pass

                setattr(self, k, Struct(getattr(default_obj, k, v), v))

                if default_struct is not None:
                    new_struct = getattr(default_obj, k)
                    for default_k in dir(default_struct):
                        if default_k.startswith('_'):
                            continue
                        if default_k not in dir(new_struct):
                            setattr(new_struct, default_k, getattr(default_struct, default_k))
            else:
                setattr(self, k, v)

        for default_k in dir(default_obj):
            if default_k.startswith('_'):
                continue

            if default_k not in obj:
                v = getattr(default_obj, default_k)
                if isinstance(v, types.BuiltinMethodType):
                    continue
                setattr(self, default_k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


class BaseConfig(object):
    """
    기본 Configuration
    """
    CONFIG_FILE_PATH = None
    config = None

    @classmethod
    def validation_check(cls):
        """config 파일에 쓴 데이터가 사용가능한 형태인지 확인

        :return:
        """
        pass

    @classmethod
    def init(cls, config_file_path):
        """ config 파일에 내용을 객체로 읽어오는 init

        :param config_file_path:
        :return:
        """
        cls.CONFIG_FILE_PATH = config_file_path
        if cls.config is None:
            try:
                with open(cls.CONFIG_FILE_PATH) as fd:
                    for k, v in yaml.load(fd, Loader=yaml.FullLoader).items():
                        default_struct = None
                        try:
                            default_struct = getattr(cls, k)
                        except AttributeError:
                            pass

                        if isinstance(v, dict):
                            setattr(cls, k, Struct(getattr(cls, k, v), v))

                            # default 값을 매핑 시키기 위한 용도
                            if default_struct is not None:
                                new_struct = getattr(cls, k)
                                for default_k in dir(default_struct):
                                    if default_k.startswith('_'):
                                        continue
                                    if default_k not in dir(new_struct):
                                        setattr(new_struct, default_k, getattr(default_struct, default_k))
                        else:
                            setattr(cls, k, v)

                        # 정의된 validation_check 를 실행하기 위한 용도
                        if 'validation_check' in dir(default_struct):
                            checker = getattr(default_struct, 'validation_check')
                            _tmp = getattr(cls, k)
                            for kk in dir(_tmp):
                                if kk.startswith('_'):
                                    continue
                                setattr(default_struct, kk, getattr(_tmp, kk))
                            checker()

            except Exception as err:
                logging.fatal(f"config load error {err}")
                raise

            # cls.validation_check()