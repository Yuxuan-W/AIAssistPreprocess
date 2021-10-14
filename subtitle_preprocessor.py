"""
Running basic pre-processing for the .srt subtitle files
"""
import json
import re
import os
import pysrt
import glob

from joblib import Parallel, delayed
from tqdm import tqdm
from json_utils import load_annotation_list
from configs.preprocess_configs import DOWNLOAD_ROOT, SUBTITLE_ROOT

def save_json(data, filename):
    """data corresponds to a single file"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(data)]))


def convert_sub_time_to_seconds(sub_time):
    """sub_time is a SubRipTime object defined by pysrt"""
    return 60 * sub_time.minutes + sub_time.seconds + 0.001 * sub_time.milliseconds


def clean_single_sub_sentence(sub_sentence):
    """sub_sentence: str, """
    sub_sentence = sub_sentence.replace("\n", " ")
    sub_sentence = sub_sentence.replace("(", " ")
    sub_sentence = sub_sentence.replace(")", " ")
    sub_sentence = sub_sentence.replace(":", " : ")
    sub_sentence = re.sub(r"\s{2,}", " ",
                          sub_sentence)  # 这里是正则表达式，单空格替换双空格 https://www.runoob.com/python/python-reg-expressions.html
    return sub_sentence


def split_multi_lines(cur_sub):
    """
    return: subtitle list that split all multi-line items
    """
    start_time = []
    end_time = []
    duration = cur_sub.duration
    text_length = len(cur_sub.text)
    text = cur_sub.text.split('\n')

    start_time.append(cur_sub.start)
    for i in range(len(text) - 1):
        middle_time = start_time[i].__add__(duration.__mul__(len(text[i]) / text_length))
        start_time.append(middle_time)
        end_time.append(middle_time)
    end_time.append(cur_sub.end)

    splited_item = dict(
        text_list=text,
        start_time_list=start_time,
        end_time_list=end_time
    )
    return splited_item


def preprocess_subtitles_single_video(video_path, save_path, segment_list):
    """
    return: A python dict, the keys are the video names, the entries are lists,
            each contains all the text from a .srt file
    sub_times are the start time of the sentences.
    """
    video_id = os.path.basename(video_path).split('.')[0]
    subs = pysrt.open(video_path, encoding="iso-8859-1")  # 打开字幕.srt文件到subs
    if len(subs) == 0:
        subs = pysrt.open(video_path)

    # 转成单行
    sub_single_line = []
    for cur_sub in subs:
        # 对读取到subs中的每一个srt item进行处理
        text = cur_sub.text
        if text == '':
            continue

        if '\n' in text:
            splited_item = split_multi_lines(cur_sub)
            text_list = splited_item['text_list']
            start_time_list = splited_item['start_time_list']
            end_time_list = splited_item['end_time_list']
            for sentence_index in range(len(text_list)):
                sub_single_line.append(dict(
                    text=text_list[sentence_index],
                    start=start_time_list[sentence_index],
                    end=end_time_list[sentence_index]
                ))
        else:
            sub_single_line.append(dict(
                text=cur_sub.text,
                start=cur_sub.start,
                end=cur_sub.end
            ))

    # 去重
    sub_data = []
    prev = sub_single_line[0]
    for sentence_index in range(1, len(sub_single_line)):
        cur_sub = sub_single_line[sentence_index]
        if cur_sub['text'] != prev['text']:
            sub_data.append(dict(
                text=clean_single_sub_sentence(prev['text']),
                start=convert_sub_time_to_seconds(prev['start']),
                end=convert_sub_time_to_seconds(cur_sub['start'])
            ))
            prev = cur_sub

        if sentence_index == len(sub_single_line) - 1:
            sub_data.append(dict(
                text=clean_single_sub_sentence(prev['text']),
                start=convert_sub_time_to_seconds(prev['start']),
                end=convert_sub_time_to_seconds(cur_sub['end'])
            ))

    # 划分入segment，并在segment分界点拆开
    seg_sub_data = [[] for i in range(len(segment_list))]
    seg_index = 0
    sentence_index = 0
    while sentence_index < len(sub_data) and seg_index < len(segment_list):
        split_point = segment_list[seg_index][1]
        if sub_data[sentence_index]['start'] < split_point < sub_data[sentence_index]['end']:
            text = sub_data[sentence_index]['text']
            start = sub_data[sentence_index]['start']
            end = sub_data[sentence_index]['end']

            # split the sentence into two
            n_characters = len(text) * ((split_point - start) / (end - start))
            text = text.split(' ')
            text_before = ''
            text_after = ''
            for word in text:
                if len(text_before) + len(word) <= n_characters:
                    text_before = text_before + word + ' '
                else:
                    text_after = text_after + word + ' '

            # put the split item back to list
            sub_data[sentence_index]['end'] = split_point
            sub_data[sentence_index]['text'] = text_before[:-1]
            sub_data.insert(sentence_index + 1, dict(
                text=text_after[:-1],
                start=split_point,
                end=end
            ))

            seg_sub_data[seg_index].append(sub_data[sentence_index])
            seg_index += 1

        else:
            seg_sub_data[seg_index].append(sub_data[sentence_index])

        sentence_index += 1

    # write to json
    for seg_index in range(len(segment_list)):
        srt_data = dict(
            seg_id=video_id + '_' + str(seg_index),
            sub=seg_sub_data[seg_index]
        )
        save_json(srt_data, save_path + srt_data['seg_id'] + '.json')


def preprocess_subtitles(segment, srt_dir=DOWNLOAD_ROOT, save_path=SUBTITLE_ROOT):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Start preprocessing srt files from %s ..." % srt_dir)
    srt_paths = glob.glob(os.path.join(srt_dir, "*.srt"))
    Parallel(n_jobs=32)(delayed(preprocess_subtitles_single_video)
                        (srt, save_path, segment[os.path.basename(srt).split('.')[0]])
                        for srt in tqdm(srt_paths, desc="Loop over subtitle files"))


if __name__ == '__main__':
    # Get segment info
    seg_info = dict()
    annotation_list = load_annotation_list()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']

    preprocess_subtitles(seg_info)
