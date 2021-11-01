import os
import re

from configs.preprocess_configs import ID_FILE_ROOT, ANNOTATION_ROOT, ANNOTATION_PACKAGE_ROOT
from json_utils import load_annotation_list, save_json, save_jsonl, load_json


def remove_tag(sentence):
    embedded_word_list = (re.findall(re.compile(r"[<](.*?)[>]", re.S), sentence))

    for embedded_word in embedded_word_list:
        if '/' in embedded_word:
            kept_word = embedded_word.split('/')[-1]
            sentence = str.replace(sentence, '<' + embedded_word + '>', '<' + kept_word + '>')
    sentence = str.replace(sentence, '<', '')
    sentence = str.replace(sentence, '>', '')

    return sentence


def preprocess_annotation(idfile_save_path=ID_FILE_ROOT, release_save_path=ANNOTATION_ROOT):
    annotation_list = load_annotation_list()
    id2seg_dict = dict()
    seg2id_dict = dict()
    sid = 0
    id2query_dict = dict()
    query2id_dict = dict()
    qid = 0
    for i in range(len(annotation_list)):
        annotation = annotation_list[i]
        vid = annotation[0]['videoID']
        # segment id
        for seg_idx in range(len(annotation[1]['segInfo'])):
            id2seg_dict[sid] = vid + '_' + str(seg_idx)
            seg2id_dict[vid + '_' + str(seg_idx)] = sid
            sid += 1
        for j in range(2, len(annotation)):
            query = annotation[j]
            # query id
            id2query_dict[qid] = query['ID']
            query2id_dict[query['ID']] = qid
            qid += 1
            # process query
            del query['Reason']
            query['Question'] = remove_tag(query['Question'])
            annotation[j] = query
        annotation_list[i] = annotation
    id_dict = dict(
        id2seg=id2seg_dict,
        seg2id=seg2id_dict,
        id2query=id2query_dict,
        query2id=query2id_dict
    )
    save_json(id_dict, os.path.join(idfile_save_path, 'id.json'))
    save_jsonl(annotation_list, os.path.join(release_save_path, 'annotation_release.jsonl'))


def package_annotation(idfile_root=ID_FILE_ROOT, test_list_path='test.txt',
                       annotation_root=ANNOTATION_ROOT, save_path=ANNOTATION_PACKAGE_ROOT):
    '''
    "meta": {
                "query_id": int,--in id.json
                "text_query": str,--in annotation_release                                  # purely text query
                "original_query": str,--in load_annotation()
                "query_image_path": str,--config + q_name
                "vid_name": str,--in json list                                    # youtube_id (11)
                "answer_segment_name": list[str],--in load_annotation                  # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],--in segment_name + id.json             # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]],--in load_annotation   # start_time, end_time of coresponding segment

                "sample_seg_id_for_training": int,              # sample one segment for training
                #####
            }
    '''
    # load all required file
    query2id = load_json(os.path.join(idfile_root, 'id.json'))['query2id']
    id2query = load_json(os.path.join(idfile_root, 'id.json'))['id2query']
    seg2id = load_json(os.path.join(idfile_root, 'id.json'))['seg2id']
    annotation = load_annotation_list()
    test_set = set()
    with open(test_list_path) as f:
        for line in f:
            vid = line.split('\n')[0]
            test_set.add(vid)

    # generate query dict, the key is query_name
    query_dict_by_name = dict()
    for anno in annotation:
        vid = anno[0]['videoID']
        seg_info = anno[1]['segInfo']
        query_list = anno[2:]
        for q in query_list:
            segment_name_list = [vid + '_' + str(seg_idx - 1) for seg_idx in q['Segment']]
            query_dict_by_name[q['ID']] = dict(
                query_id=query2id[q['ID']],
                query_name=q['ID'],
                text_query=remove_tag(q['Question']),
                original_query=q['Question'],
                query_img_path=os.path.join(annotation_root + '/image/' + vid, q['Filename']),
                vid_name=vid,
                query_type=q['QueryType'],
                answer_segment_name=segment_name_list,
                answer_segment_id=[seg2id[seg_name] for seg_name in segment_name_list],
                answer_segment_info=[seg_info[seg_idx - 1] for seg_idx in q['Segment']],
            )

    # package to iterable list for dataloader
    train_package = []
    test_package = []
    for _, query_name in id2query.items():
        vid = query_name[:11]
        query_item = query_dict_by_name[query_name]
        if vid in test_set:
            test_package.append(query_item)
        else:
            train_package.append(query_item)
    save_jsonl(train_package, os.path.join(save_path, 'trainset.jsonl'))
    save_jsonl(test_package, os.path.join(save_path, 'testset.jsonl'))


if __name__ == '__main__':
    preprocess_annotation()
    package_annotation()