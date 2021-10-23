import glob
import os
import math
import h5py
import numpy as np
from json_utils import load_annotation_list
from configs.preprocess_configs import (CLIP_LENGTH, GRID_FEATURE_ROOT_FRAME, GRID_FEATURE_ROOT_CLIP,
                                        TEXT_FEATURE_ROOT_SUBTITLE, TEXT_FEATURE_ROOT_CLIP)


def frame_feature_pool(seg_h5, clip_length=CLIP_LENGTH):
    time_list = sorted([float(x) for x in seg_h5.keys()])
    seg_start_time = time_list[0]
    clip_idx = 0
    feature_list = []
    clip_start_time_list = []
    clip_feature_max_pooling = []
    clip_feature_avg_pooling = []
    for time in time_list:
        if time >= (clip_idx + 1) * clip_length + seg_start_time:
            feature_arr = np.array(feature_list)
            clip_start_time_list.append(clip_idx * clip_length + seg_start_time)
            clip_feature_max_pooling.append(np.max(feature_arr, axis=0))
            clip_feature_avg_pooling.append(np.average(feature_arr, axis=0))
            clip_idx += 1
            feature_list = []
        feature_list.append(seg_h5[str(time)][0][:][:][:])
    clip_feature = dict(
        start_time=clip_start_time_list,
        max_pooling=np.array(clip_feature_max_pooling),
        avg_pooling=np.array(clip_feature_avg_pooling)
    )
    return clip_feature


def sub_feature_pool(seg_group, seg_time, clip_length=CLIP_LENGTH):
    time_list = seg_group['timeline'][:].tolist()
    seg_start_time = seg_time[0]
    seg_end_time = seg_time[1]
    n_clip = math.ceil((seg_end_time - seg_start_time)/clip_length)
    clip_start_time_list = [seg_start_time + clip_length*j for j in range(n_clip)]
    feature_list = [[] for j in range(n_clip)]
    for i in range(len(time_list)):
        time = time_list[i]
        clip_idx = int((time - seg_start_time)/clip_length)
        feature = seg_group['feature'][0, i]
        feature_list[clip_idx].append(feature)

    # padding with all zero numpy array, pooling
    clip_feature_max_pooling = []
    clip_feature_avg_pooling = []
    for i in range(len(feature_list)):
        if not feature_list[i]:
            feature_list[i].append(np.zeros(np.size(feature)))
        feature_list[i] = np.array(feature_list[i])
        clip_feature_max_pooling.append(np.max(feature_list[i], axis=0))
        clip_feature_avg_pooling.append(np.average(feature_list[i], axis=0))
    clip_feature = dict(
        start_time=clip_start_time_list,
        max_pooling=clip_feature_max_pooling,
        avg_pooling=clip_feature_avg_pooling
    )
    return clip_feature


def separate_into_clip(seg_info, frame_root=GRID_FEATURE_ROOT_FRAME, sub_root=TEXT_FEATURE_ROOT_SUBTITLE,
                       clip_grid_root=GRID_FEATURE_ROOT_CLIP, clip_text_root=TEXT_FEATURE_ROOT_CLIP):

    # process grid feature
    if not os.path.exists(clip_grid_root):
        os.makedirs(clip_grid_root)
    frame_grid_dir_list = glob.glob(os.path.join(frame_root, '*'))
    for frame_grid_dir in frame_grid_dir_list:
        vid = frame_grid_dir.split('/')[-1]
        new_video_h5 = h5py.File(os.path.join(clip_grid_root, vid + '.hdf5'), 'w')
        frame_grid_h5_list = glob.glob(os.path.join(frame_grid_dir, '*.hdf5'))
        for frame_grid_h5 in frame_grid_h5_list:
            seg_id = frame_grid_h5.split('/')[-1][:-5]
            with h5py.File(frame_grid_h5, 'r') as seg_h5:
                clip_feature = frame_feature_pool(seg_h5)
            seg_group = new_video_h5.create_group(seg_id)
            seg_group.create_dataset('start_time', data=clip_feature['start_time'])
            seg_group.create_dataset('max_pooling', data=clip_feature['max_pooling'])
            seg_group.create_dataset('avg_pooling', data=clip_feature['avg_pooling'])
        new_video_h5.close()

    # process text feature
    if not os.path.exists(clip_text_root):
        os.makedirs(clip_text_root)
    sub_h5_list = glob.glob(os.path.join(sub_root, '*'))
    for sub_h5 in sub_h5_list:
        vid = sub_h5.split('/')[-1][:-5]
        video_seg_info = seg_info[vid]
        new_video_h5 = h5py.File(os.path.join(clip_text_root, vid + '.hdf5'), 'w')
        with h5py.File(sub_h5, 'r') as video_h5:
            for key in video_h5.keys():
                clip_feature = sub_feature_pool(video_h5[key], video_seg_info[int(key.split('_')[-1])])
                seg_group = new_video_h5.create_group(key)
                seg_group.create_dataset('start_time', data=clip_feature['start_time'])
                seg_group.create_dataset('max_pooling', data=clip_feature['max_pooling'])
                seg_group.create_dataset('avg_pooling', data=clip_feature['avg_pooling'])
        new_video_h5.close()


if __name__ == '__main__':
    seg_info = dict()
    annotation_list = load_annotation_list()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']
    separate_into_clip(seg_info)