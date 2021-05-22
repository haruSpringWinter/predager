import dacite, json
from dataclasses import dataclass

PATH_TO_CONF = '/app/conf/conf.json'

@dataclass
class SparkSetting:
    executor: dict
    driver: dict
    cores: dict


@dataclass
class Configuration:
    spark: SparkSetting


with open(PATH_TO_CONF) as f:
    try:
        raw_json_data = json.load(f)
        f.close()
    except Exception:
        raise IOError('Cannot load config json')

    try:
        predager_config = dacite.from_dict(data_class=Configuration, data=raw_json_data, config=dacite.Config())
        print('run with configuration: ' + str(predager_config))
    except Exception:
        raise ValueError('Cannot mapping json to config object')
