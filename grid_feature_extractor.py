#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script.
"""
import argparse
import csv
import glob
import os
import torch
import h5py
import numpy as np
import torchvision
from functools import partial

from joblib import Parallel, delayed
from tqdm import tqdm
from configs.preprocess_configs import NUM_JOBS, GRID_FEATURE_DIM, ANNOTATION_ROOT, FRAME_ROOT, \
    GRID_FEATURE_ROOT_QUERY, GRID_FEATURE_ROOT_FRAME, GRID_FEATURE_R50_PATH

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import detection_utils as utils

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {'coco_2014_train': 'train2014', 'coco_2014_val': 'val2014', 'coco_2015_test': 'test2015'}


def extract_grid_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="configs/R-50-grid.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test'])
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def extract_grid_feature_single_dir(model, roi_pooler, out_path, img_root, csv_path):
    '''
    compute feature from single image folder: img_root
    csv_root is optional
    outputs the hdf5 file, whose name is defined in out_path
    '''

    # start output
    with torch.no_grad():
        # get image list
        img_list = glob.glob(os.path.join(img_root, '*.jpg'))

        # csv process: build a dictionary for all bbox in a video
        # for each question, there is a single dictionary including all the bbox info.
        if csv_path != '':
            bbox_list = dict()
            csv_reader = csv.reader(open(csv_path))
            for row in csv_reader:
                bbox_name = row[0]
                coordinate = [int(row[1]), int(row[2]), int(row[1]) + int(row[3]), int(row[2]) + int(row[4])]
                img_name = row[5]
                if img_name not in bbox_list:
                    bbox_list[img_name] = dict()
                bbox_list[img_name][bbox_name] = coordinate

        # extract feature from every image
        f = h5py.File(out_path + '.hdf5', "w")
        for img_path in img_list:
            # get the image
            img = utils.read_image(img_path, format='BGR')
            img_name = img_path.split('/')[-1]
            dta_dict = dict()
            dta_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            inp = model.preprocess_image([dta_dict])
            features = model.backbone(inp.tensor)

            # to save: conv5_feat for video_frame / whole image
            conv5_feat = model.roi_heads.get_conv5_features(features)

            # for every image, extract feature for the full image
            group = f.create_group(img_name)
            h, w, _ = img.shape
            image_feature = roi_pooler(
                input=conv5_feat,
                boxes=torch.FloatTensor([[0, 0, 0, w, h]]).to(model.device)
            )
            group['image'] = torch.Tensor.cpu(image_feature)

            # if bbox exists, extract region feature for bbox
            if '_bbox_' in img_path:
                # extract feature for every bbox
                for name, coordinate in bbox_list[img_name].items():
                    bbox_region_feature = roi_pooler(
                        input=conv5_feat,
                        boxes=torch.FloatTensor([[0] + coordinate]).to(model.device)
                    )
                    group[name] = torch.Tensor.cpu(bbox_region_feature)

        f.close()


def extract_grid_feature(query_input_root=ANNOTATION_ROOT,
                         query_output_root=GRID_FEATURE_ROOT_QUERY,
                         frame_input_root=FRAME_ROOT,
                         frame_output_root=GRID_FEATURE_ROOT_FRAME):
    print('Start extracting grid features...')
    # build & load pretrained models
    args = extract_grid_feature_argument_parser().parse_args()
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        GRID_FEATURE_R50_PATH, resume=True
    )
    model.eval()

    # output_size here should be configurable: 1x1, 3x3, 7x7, etc;
    # 1/32 corresponding to 1/32, no need to modify
    roi_pooler = partial(torchvision.ops.roi_pool, output_size=(GRID_FEATURE_DIM, GRID_FEATURE_DIM),
                         spatial_scale=1 / 32)

    # extract feature from query image
    if not os.path.exists(query_output_root):
        os.makedirs(query_output_root)
    img_root_list = glob.glob(os.path.join(query_input_root + '/image', '*'))
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_grid_feature_single_dir)
                              (model, roi_pooler,
                               out_path=query_output_root + '/' + img_root.split('/')[-1],
                               img_root=img_root,
                               csv_path=query_input_root + '/csv/' + img_root.split('/')[-1] + '.csv'
                               if os.path.exists(query_input_root + '/csv/' + img_root.split('/')[-1] + '.csv') else '')
                              for img_root in tqdm(img_root_list, desc='Extracting feature from query images'))

    # extract feature from video frame
    img_root_list = []
    video_root_list = glob.glob(os.path.join(frame_input_root, '*'))
    for video_root in video_root_list:
        img_root_list = img_root_list + glob.glob(os.path.join(video_root, '*'))
    for img_root in img_root_list:
        if not os.path.exists(frame_output_root + '/' + img_root[-25:-14]):
            os.makedirs(frame_output_root + '/' + img_root[-25:-14])
    Parallel(n_jobs=NUM_JOBS)(delayed(extract_grid_feature_single_dir)
                              (model, roi_pooler,
                               out_path=frame_output_root + '/' + img_root[-25:],
                               img_root=img_root, csv_path='')
                              for img_root in tqdm(img_root_list, desc='Extracting feature from video frames'))


if __name__ == "__main__":
    extract_grid_feature()
