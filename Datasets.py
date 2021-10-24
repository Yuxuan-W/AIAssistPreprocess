"""
Dataset for clip model
"""
import copy
import os
import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from easydict import EasyDict as edict

from json_utils import load_json, load_jsonl
from feature_packager import load_from_feature_package
from configs.preprocess_configs import ANNOTATION_PACKAGE_ROOT, FEATURE_PACKAGE_ROOT, ID_FILE_ROOT


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_image_text_alignment(sparse_alignment: list, max_img_feat: int, padding_value: int):
    """
    sparse_alignment:
    max_img_feat:
    return: N_img x max_img_feat x max_alignment_length
            B     x max_img_feat x max_alignment_length

    sparse_alignment:
    [
        [   # image 1
            [1,2,3],    # whole image feature to the position of text feature embedding
            [4,5,6,7],  # bbox1 feature to the position of text feature embedding
            [8,9,10],   # bbox2 feature to the position of text feature embedding
        ],
         ...
         [  #image 2
            [1,2,3,4],
            [3,4,5,6,7],
            [8,9,10],
         ],
    ]
    ##
    Giving a sparse alignment matrix, return a dense one with padding;
    """

    max_alignment_length = max([len(region_i) for image_i in sparse_alignment for region_i in image_i])
    bs = len(sparse_alignment)
    padded_image_text_alignment = \
        np.ones((bs, max_img_feat, max_alignment_length), dtype=np.int32) * padding_value
    for i, img_ in enumerate(sparse_alignment):
        for j, feat_ in enumerate(img_):
            padded_image_text_alignment[i, j, :len(feat_)] = feat_

    return padded_image_text_alignment


def collate_for_concat_fusion(batch):
    # collate function for concat text embedding and vis embedding
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]

    pad_query_text_feat, query_text_mask = pad_sequences_1d(batch_collect.query_text_feat, dtype=torch.float)
    pad_query_vis_feat, query_vis_mask = pad_sequences_1d(batch_collect.query_vis_feat, dtype=torch.float)
    max_len_img_feat = pad_query_vis_feat.shape[1]
    pad_img_text_alignment = torch.from_numpy(
        pad_image_text_alignment(batch_collect.image_2_text_alignment, max_len_img_feat, padding_value=-1)
    )
    pad_ctx_text_feat, ctx_text_mask = pad_sequences_1d(batch_collect.ctx_text_feat, dtype=torch.float)
    pad_ctx_vis_feat, ctx_vis_mask = pad_sequences_1d(batch_collect.ctx_vis_feat, dtype=torch.float)
    return edict(
        meta=batch_collect.meta,
        pad_query_text_feat=pad_query_text_feat,
        query_text_mask=query_text_mask,
        pad_query_vis_feat=pad_query_vis_feat,
        query_vis_mask=query_vis_mask,
        image_2_text_alignment=pad_img_text_alignment,
        pad_ctx_text_feat=pad_ctx_text_feat,
        pad_ctx_vis_feat=pad_ctx_vis_feat,
        ctx_text_mask=ctx_text_mask,
        ctx_vis_mask=ctx_vis_mask
    )


def collate_for_adding_fusion(batch):
    # collate function for adding text embedding and vis embedding for fusion
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]
    pad_query_text_feat, query_text_mask = pad_sequences_1d(batch_collect.query_text_feat, dtype=torch.float)
    pad_query_vis_feat = torch.zeros(
        pad_query_text_feat.size()[:2] + (batch_collect.query_vis_feat[0].shape[-1],),
        dtype=pad_query_text_feat.dtype
    )
    query_vis_mask = copy.deepcopy(query_text_mask)
    query_token_type_ids = torch.ones(
        pad_query_text_feat.shape[:2], dtype=torch.long
    )
    for bidx, (vis_feat, i2t) in enumerate(zip(batch_collect.query_vis_feat, batch_collect.image_2_text_alignment)):
        for idx, region2pos in enumerate(i2t):
            pad_query_vis_feat[bidx][region2pos] = vis_feat[idx]
            if idx == 0:  # 0 stands for the whole image
                query_token_type_ids[bidx][region2pos] = 0

    pad_ctx_text_feat, ctx_text_mask = pad_sequences_1d(batch_collect.ctx_text_feat, dtype=torch.float)
    pad_ctx_vis_feat, ctx_vis_mask = pad_sequences_1d(batch_collect.ctx_vis_feat, dtype=torch.float)

    return edict(
        meta=batch_collect.meta,
        pad_query_text_feat=pad_query_text_feat,
        query_text_mask=query_text_mask,
        pad_query_vis_feat=pad_query_vis_feat,
        query_vis_mask=query_vis_mask,
        query_token_type_ids=query_token_type_ids,
        image_2_text_alignment=batch_collect.image_2_text_alignment,
        pad_ctx_text_feat=pad_ctx_text_feat,
        pad_ctx_vis_feat=pad_ctx_vis_feat,
        ctx_text_mask=ctx_text_mask,
        ctx_vis_mask=ctx_vis_mask
    )


class VQASR_query(Dataset):
    """
    Args:
        dset_name, str, ["VQASR"]
        avg_pooling, boolean, default = False, True for avg_pooling, False for max_pooling

    Return:
        a dict: {
            "meta": {
                "query_id": int,
                "text_query": str,                                  # purely text query
                "original_query": str,
                "query_image_path": str,
                "vid_name": str,                                    # youtube_id (11)
                "answer_segment_name": list[str],                   # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],              # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]],  # start_time, end_time of coresponding segment

                "sample_seg_id_for_training": int,                  # sample one segment for training
                #####
            }

            "query_text_feat": torch.tensor, (L, D_q)                       # query feature
            "query_vis_feat": torch.tensor,  (n_region, 2048)               # image feature&region feature
            "image_2_text_alignment": list[list]                              # image to token alignment
            "ctx_vis_feat": torch.tensor, (n_clip_in_segment, dim_video)     # video feature
            "ctx_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)      # sub feature

        }
    """

    def __init__(self, dset_name="train", query_bert_path_or_handler="", sub_feat_path_or_handler="",
                 vid_feat_path_or_handler="", normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):
        assert dset_name in ['train', 'test'], "dset_name should be whether 'train' or 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        else:
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        self.query_bert_path_or_handler = query_bert_path_or_handler
        self.sub_feat_path_or_handler = sub_feat_path_or_handler
        self.vid_fear_path_or_handler = vid_feat_path_or_handler
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.query_text_feat = load_from_feature_package(f['query_text_feature'])
            self.query_img_feat = load_from_feature_package(f['query_grid_feature'])
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sample_seg_idx = random.sample(range(len(item['answer_segment_id'])), 1)[0]
        meta = edict(
            query_id=item['query_id'],
            query_name=item['query_name'],
            text_query=item['text_query'],
            original_query=item['original_query'],
            query_img_path=item['query_img_path'],
            vid_name=item['vid_name'],
            answer_segment_name=item['answer_segment_name'],
            answer_segment_id=item['answer_segment_id'],
            answer_segment_info=item['answer_segment_info'],
            sample_seg_id_for_training=item['answer_segment_id'][sample_seg_idx],
            sample_seg_name_for_training=item['answer_segment_name'][sample_seg_idx]
        )

        query_text_feat = self.query_text_feat[item['vid_name']][item['query_name']]['feature'][0]
        img_2_text_alignment = self.query_text_feat[item['vid_name']][item['query_name']]['img_alignment']

        query_vis_feat = self.query_img_feat[item['vid_name']][item['query_img_path'].split('/')[-1]]

        ctx_vis_feat = self.video_vis_feat[item['vid_name']][item['answer_segment_name'][sample_seg_idx]][self.pooling]
        ctx_text_feat = self.sub_text_feat[item['vid_name']][item['answer_segment_name'][sample_seg_idx]][self.pooling]

        if self.normalize_tfeat:
            query_text_feat = l2_normalize_np_array(query_text_feat)
            ctx_text_feat = l2_normalize_np_array(ctx_text_feat)

        if self.normalize_vfeat:
            query_vis_feat = l2_normalize_np_array(query_vis_feat)
            ctx_vis_feat = l2_normalize_np_array(ctx_vis_feat)

        return edict(
            meta=meta,
            query_text_feat=torch.from_numpy(query_text_feat),
            query_vis_feat=torch.from_numpy(query_vis_feat),
            image_2_text_alignment=img_2_text_alignment,
            ctx_vis_feat=torch.from_numpy(ctx_vis_feat),
            ctx_text_feat=torch.from_numpy(ctx_text_feat)
        )


class VQASR_segment(Dataset):
    def __init__(self, dset_name="train", query_bert_path_or_handler="", sub_feat_path_or_handler="",
                 vid_feat_path_or_handler="", normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):
        assert dset_name in ['train', 'test'], "dset_name should be whether 'train' or 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        else:
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        self.query_bert_path_or_handler = query_bert_path_or_handler
        self.sub_feat_path_or_handler = sub_feat_path_or_handler
        self.vid_fear_path_or_handler = vid_feat_path_or_handler
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Generate iterable segment list
        self.segment_list = []
        vid_set = set()
        for query in self.data:
            vid = query['query_name'][:11]
            vid_set.add(vid)

        seg2id = load_json(os.path.join(ID_FILE_ROOT, 'id.json'))['seg2id']
        for seg_name, seg_id in seg2id.items():
            vid = seg_name[:11]
            if vid in vid_set:
                self.segment_list.append([seg_id, seg_name, vid])

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, index):
        seg = self.segment_list[index]
        seg_id = seg[0]
        seg_name = seg[1]
        vid = seg[2]

        ctx_vis_feat = self.video_vis_feat[vid][seg_name][self.pooling]
        ctx_text_feat = self.sub_text_feat[vid][seg_name][self.pooling]

        if self.normalize_tfeat:
            ctx_text_feat = l2_normalize_np_array(ctx_text_feat)

        if self.normalize_vfeat:
            ctx_vis_feat = l2_normalize_np_array(ctx_vis_feat)

        return edict(
            seg_id=seg_id,
            seg_name=seg_name,
            vid_name=vid,
            ctx_vis_feat=torch.from_numpy(ctx_vis_feat),
            ctx_text_feat=torch.from_numpy(ctx_text_feat)
        )


if __name__ == "__main__":
    train_set = VQASR_query()
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True, collate_fn=collate_for_concat_fusion)
    for batch in train_loader:
        b = batch

    test_set = VQASR_segment(dset_name='test')
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
    for batch in test_loader:
        b = batch