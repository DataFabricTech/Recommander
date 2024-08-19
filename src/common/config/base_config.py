import yaml


class Struct(object):
    def __init__(self, default_obj, obj):
        self.merge_objects(default_obj, obj)

    def merge_objects(self, default_obj, obj):
        # 기본 객체와 새로운 객체를 병합
        for k, v in obj.items():
            default_struct = getattr(default_obj, k, None)
            if isinstance(v, dict):
                setattr(self, k, Struct(default_struct, v))
                self.copy_default_attributes(default_struct, k)
            else:
                setattr(self, k, v)

        # 기본 객체의 속성을 새로운 객체에 없는 경우 추가
        self.add_missing_attributes(default_obj, obj)

    def copy_default_attributes(self, default_struct, k):
        # 기본 속성을 복사
        if default_struct is not None:
            new_struct = getattr(self, k)
            for default_k in dir(default_struct):
                if default_k.startswith('_'):
                    continue
                if default_k not in dir(new_struct):
                    setattr(new_struct, default_k, getattr(default_struct, default_k))

    def add_missing_attributes(self, default_obj, obj):
        for default_k in dir(default_obj):
            if default_k.startswith('_') or default_k in obj:
                continue
            v = getattr(default_obj, default_k)
            setattr(self, default_k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


class BaseConfig(object):
    CONFIG_FILE_PATH = "/app/config_templates/config.yml"
    config = None

    @classmethod
    def init(cls, config_file_path=None):
        config_file_path = config_file_path or cls.CONFIG_FILE_PATH
        if cls.config is None:
            try:
                with open(config_file_path) as fd:
                    loaded_config = yaml.safe_load(fd)
                    cls.apply_config(loaded_config)
            except FileNotFoundError:
                print(f"Config file {config_file_path} not found.")

    @classmethod
    def apply_config(cls, config_data):
        for k, v in config_data.items():
            default_struct = getattr(cls, k, None)

            if isinstance(v, dict):
                setattr(cls, k, Struct(default_struct, v))
                if default_struct:
                    cls.copy_default_to_new_struct(k, default_struct)
            else:
                setattr(cls, k, v)

            # 유효성 검사 실행
            cls.run_validation_check(default_struct, k)

    @classmethod
    def copy_default_to_new_struct(cls, k, default_struct):
        new_struct = getattr(cls, k)
        for default_k in dir(default_struct):
            if default_k.startswith('_'):
                continue
            if default_k not in dir(new_struct):
                setattr(new_struct, default_k, getattr(default_struct, default_k))

    @classmethod
    def run_validation_check(cls, default_struct, k):
        if default_struct and 'validation_check' in dir(default_struct):
            checker = getattr(default_struct, 'validation_check')
            _tmp = getattr(cls, k)
            for kk in dir(_tmp):
                if kk.startswith('_'):
                    continue
                setattr(default_struct, kk, getattr(_tmp, kk))
            checker()


if __name__ == '__main__':
    a = OtherClass.config
