import glob
import os
import json
from configs.preprocess_configs import ANNOTATION_ROOT


def load_json(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[0].strip("\n"))


def load_json_dir(path):
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


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def load_annotation_list():
    return load_json_dir(ANNOTATION_ROOT + '/json')


def load_subtitle_list():
    root = 'data/segment/subtitle'
    video_root_list = glob.glob(os.path.join(root, '*'))
    subtitle_list = []
    for video_root in video_root_list:
        subtitle_list = subtitle_list + load_json_dir(video_root)
    return subtitle_list


if __name__ == '__main__':
    path = 'data/annotation'
    path = glob.glob(os.path.join(path, "*.json"))

    annotation = []
    for filename in path:
        annotation.append(load_json(filename))