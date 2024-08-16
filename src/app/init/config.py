import os
import argparse
from common.config import Config


if 'CONFIG_FILE_PATH' in os.environ:
    Config.init(os.environ['CONFIG_FILE_PATH'])
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", dest="config_file_path", help="config file path")
    args = parser.parse_args()
    Config.init(args.config_file_path)