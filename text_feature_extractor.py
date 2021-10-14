import os
import re
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from json_utils import load_subtitle_list, load_annotation_list
from tqdm import tqdm
from joblib import Parallel, delayed
from configs.preprocess_configs import NUM_JOBS, TEXT_FEATURE_ROOT_QUERY, TEXT_FEATURE_ROOT_SUBTITLE


def get_time_list(sub, tokenizer):
    time_list = []
    for sentence in sub:
        token = tokenizer.tokenize(sentence['text'])
        if not len(token) == 0:
            time_per_token = (sentence['end'] - sentence['start']) / len(token)
            for t in range(len(token)):
                time_list.append(sentence['start'] + t * time_per_token)
    return time_list


def extract_from_subtitle(tokenizer, model, sub, save_path, max_length=256):
    # Get all text
    seg_id = sub['seg_name']
    vid = seg_id[0:10]
    sub = sub['sub']
    text = sub[0]['text']
    for i in range(1, len(sub)):
        text = text + ' ' + sub[i]['text']

    # Get token and ids
    with torch.no_grad():
        feature = torch.Tensor()
        encoded_input = tokenizer(text, return_tensors='pt')
        ids = encoded_input.data['input_ids'][:, 1: -1]
        attention_mask = encoded_input.data['attention_mask'][:, 1: -1]

        # Get output and time list
        i = 0
        while i * max_length < ids.size(1):
            encoded_input.data['input_ids'] = ids[:, i*max_length: (i + 1)*max_length]
            encoded_input.data['attention_mask'] = attention_mask[:, i*max_length: (i + 1)*max_length]
            output = model(**encoded_input)
            feature = torch.cat((feature, output['last_hidden_state']), dim=-2)
            i += 1

    # Get time list
    time_list = get_time_list(sub, tokenizer)

    # Write into file
    save_path = save_path + vid + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + seg_id + '_feature.npy', feature.numpy())
    np.save(save_path + seg_id + '_time.npy', time_list)


def extract_from_query(tokenizer, model, anno, save_path):
    vid = anno[0]['videoID']
    query_list = anno[2:]
    embedded_position_dict = dict()
    for i in range(len(query_list)):
        query = query_list[i]
        qid = query['ID']
        text = query['Question']
        embedded_word_list = (re.findall(re.compile(r"[<](.*?)[>]", re.S), text))

        # remove 'bbox1'
        for embedded_word in embedded_word_list:
            if '/' in embedded_word:
                kept_word = embedded_word.split('/')[-1]
                text = str.replace(text, '<' + embedded_word + '>', '<' + kept_word + '>')
            else:
                text = str.replace(text, '<', '')
                text = str.replace(text, '>', '')

        # generate embedded position, include the first [cls] and the last [sep]
        token = tokenizer.tokenize(text)
        embedded_position = [[0]]
        embedded_idx = 0
        row_idx = 0
        position_idx = 1
        for j in range(len(token)):
            if token[j] == '<':
                embedded_position.append([])
                embedded_idx += 1
                row_idx = embedded_idx
                continue;
            if token[j] == '>':
                row_idx = 0
                continue;
            embedded_position[row_idx].append(position_idx)
            position_idx += 1
        embedded_position[0].append(position_idx)
        embedded_position_dict[qid] = embedded_position

        # remove '<' and '>'
        text = str.replace(text, '<', '')
        text = str.replace(text, '>', '')
        query_list[i]['Question'] = text

    with torch.no_grad():
        query_feature_dict = dict()
        for query in query_list:
            encoded_input = tokenizer(query['Question'], return_tensors='pt')
            output = model(**encoded_input)
            query_feature_dict[query['ID']] = output['last_hidden_state'].numpy()
        np.save(save_path + vid + '_feature.npy', query_feature_dict)
        np.save(save_path + vid + '_embedding.npy', embedded_position_dict)


def extract_text_feature(sub_save_path=TEXT_FEATURE_ROOT_SUBTITLE, query_save_path=TEXT_FEATURE_ROOT_QUERY):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    print('Start extracting text features...')

    # Extract from subtitle
    if not os.path.exists(sub_save_path):
        os.makedirs(sub_save_path)
    sub_path = load_subtitle_list()
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_from_subtitle)
                       (tokenizer, model, sub, sub_save_path)
                       for sub in tqdm(sub_path, desc='Loop from subtitles'))
    for sub in sub_path:
        extract_from_subtitle(tokenizer, model, sub, sub_save_path)

    # Extract from query
    if not os.path.exists(query_save_path):
        os.makedirs(query_save_path)
    anno_path = load_annotation_list()
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_from_query)
                       (tokenizer, model, anno, query_save_path)
                       for anno in tqdm(anno_path, desc='Loop from queries'))


if __name__ == "__main__":
    extract_text_feature()