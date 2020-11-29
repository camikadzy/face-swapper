import json
from defenitions import ROOT_DIR
home = ROOT_DIR

f = open(f"{home}/config.json")
serialized_json = json.load(f)


def get_telegram_token():
    return str(serialized_json['telegram_token'])
