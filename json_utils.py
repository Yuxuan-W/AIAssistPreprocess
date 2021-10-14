import glob
import os
import json
from configs.preprocess_configs import ANNOTATION_ROOT


def load_json(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[0].strip("\n"))


def load_json_list(path):
    path = glob.glob(os.path.join(path, "*.json"))
    json_list = []
    for filename in path:
        item = load_json(filename)
        json_list.append(load_json(filename))
    return json_list


def save_json(data, filename):
    """data corresponds to a single file"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(data)]))


def load_annotation_list():
    return load_json_list(ANNOTATION_ROOT + '/json')


def load_subtitle_list():
    return load_json_list('data/segment/subtitle')


if __name__ == '__main__':
    path = 'data/annotation'
    path = glob.glob(os.path.join(path, "*.json"))

    annotation = []
    for filename in path:
        annotation.append(load_json(filename))