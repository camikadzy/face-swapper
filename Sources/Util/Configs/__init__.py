import json

f = open("config.json")
serialized_json = json.load(f)


def get_telegram_token():
    return str(serialized_json['telegram_token'])
