import glob
import os
import re
import torch
import h5py
from transformers import DistilBertTokenizer, DistilBertModel
from json_utils import load_annotation_list, load_jsonl
from tqdm import tqdm
from joblib import Parallel, delayed
from configs.preprocess_configs import (NUM_JOBS, TEXT_FEATURE_ROOT_QUERY,
                                        TEXT_FEATURE_ROOT_SUBTITLE, SUBTITLE_ROOT)


def get_timeline(sub, tokenizer):
    timeline = []
    for sentence in sub:
        token = tokenizer.tokenize(sentence['text'])
        if not len(token) == 0:
            time_per_token = (sentence['end'] - sentence['start']) / len(token)
            for t in range(len(token)):
                timeline.append(sentence['start'] + t * time_per_token)
    return timeline


def extract_from_subtitle(tokenizer, model, sub_path, save_path, max_length=256):
    sub_list = load_jsonl(sub_path)
    vid = sub_path[-17:-6]
    f = h5py.File(save_path + '/' + vid + '.hdf5', 'w')
    for sub in sub_list:
        # Get all text
        seg_id = sub['seg_id']
        sub = sub['sub']
        text = sub[0]['text']
        for i in range(1, len(sub)):
            text = text + ' ' + sub[i]['text']

        # Get feature, timeline and token
        with torch.no_grad():
            i = 0
            token = []
            feature = torch.Tensor()
            text_word_list = text.split(' ')
            while i * max_length < len(text_word_list):
                # Get sub_text <= max_length
                sub_text_word_list = text_word_list[i * max_length: (i + 1) * max_length]
                sub_text = ''
                for word in sub_text_word_list:
                    sub_text = sub_text + word + ' '

                # Get feature and timeline
                encoded_input = tokenizer(sub_text, return_tensors='pt')
                encoded_input.data['input_ids'] = encoded_input.data['input_ids'][:, 1: -1]
                encoded_input.data['attention_mask'] = encoded_input.data['attention_mask'][:, 1: -1]
                output = model(**encoded_input)
                feature = torch.cat((feature, output['last_hidden_state']), dim=-2)

                # Get token
                sub_token = tokenizer.tokenize(sub_text)
                token = token + sub_token
                i += 1

            # Get timeline
            timeline = get_timeline(sub, tokenizer)

            # Write into h5
            group = f.create_group(seg_id)
            group['token'] = token
            group['feature'] = feature
            group['timeline'] = timeline
    f.close()


def extract_from_query(tokenizer, model, anno, save_path):
    vid = anno[0]['videoID']
    query_list = anno[2:]
    token_dict = dict()
    image_alignment_dict = dict()
    for i in range(len(query_list)):
        query = query_list[i]
        qid = query['ID']
        text = query['Question']
        embedded_word_list = (re.findall(re.compile(r"[<](.*?)[>]", re.S), text))

        # for queries without bbox, remove '<' and '>'
        # for queries with bbox, remove bboxX
        for embedded_word in embedded_word_list:
            if '/' in embedded_word:
                kept_word = embedded_word.split('/')[-1]
                text = str.replace(text, '<' + embedded_word + '>', '<' + kept_word + '>')
            else:
                text = str.replace(text, '<', '')
                text = str.replace(text, '>', '')

        # generate embedded position, include the first [cls] and the last [sep]
        token = tokenizer.tokenize(text)
        alignment_position = [[0]]
        embedded_idx = 0
        row_idx = 0
        position_idx = 1
        for j in range(len(token)):
            if token[j] == '<':
                alignment_position.append([])
                embedded_idx += 1
                row_idx = embedded_idx
                continue;
            if token[j] == '>':
                row_idx = 0
                continue;
            alignment_position[row_idx].append(position_idx)
            position_idx += 1

        alignment_position[0].append(position_idx)
        image_alignment_dict[qid] = alignment_position

        # remove '<' and '>' for queries with bbox
        text = str.replace(text, '<', '')
        text = str.replace(text, '>', '')
        query_list[i]['Question'] = text
        token_dict[qid] = tokenizer.tokenize(text)

    with torch.no_grad():
        f = h5py.File(save_path + '/' + vid + '.hdf5', 'w')
        for query in query_list:
            # generate feature
            encoded_input = tokenizer(query['Question'], return_tensors='pt')
            output = model(**encoded_input)
            feature = output['last_hidden_state'].numpy()

            # save to hdf5
            qid = query['ID']
            group = f.create_group(qid)
            group['feature'] = feature
            group['token'] = token_dict[qid]
            sub_group = group.create_group('img_alignment')
            for i in range(len(image_alignment_dict[qid])):
                sub_group.create_dataset(str(i), data=image_alignment_dict[qid][i])
        f.close()


def extract_text_feature(sub_input_path=SUBTITLE_ROOT, sub_save_path=TEXT_FEATURE_ROOT_SUBTITLE, query_save_path=TEXT_FEATURE_ROOT_QUERY):
    print('Start extracting text features...')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Extract from query
    if not os.path.exists(query_save_path):
        os.makedirs(query_save_path)
    anno_path = load_annotation_list()
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_from_query)
                              (tokenizer, model, anno, query_save_path)
                              for anno in tqdm(anno_path, desc='Loop from queries'))

    # Extract from subtitle
    if not os.path.exists(sub_save_path):
        os.makedirs(sub_save_path)
    sub_path_list = glob.glob(os.path.join(sub_input_path, '*'))
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_from_subtitle)
                       (tokenizer, model, sub_path, sub_save_path)
                       for sub_path in tqdm(sub_path_list, desc='Loop from subtitles'))


if __name__ == "__main__":
    extract_text_feature()