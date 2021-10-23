import glob
import os
import h5py
import numpy as np
from configs.preprocess_configs import (FEATURE_PACKAGE_ROOT,
                                        GRID_FEATURE_ROOT_QUERY, GRID_FEATURE_ROOT_CLIP,
                                        TEXT_FEATURE_ROOT_QUERY, TEXT_FEATURE_ROOT_CLIP)


def load_from_feature_package(group_handle):
    feature_dict = dict()
    vids = group_handle.keys()
    for vid in vids:
        feature_dict[vid] = dict()
        sub_groups = group_handle[vid].keys()
        for sub_group in sub_groups:
            if '.jpg' in sub_group:
                regions = group_handle[vid][sub_group].keys()
                region_feature_list = [[] for r in regions]
                for region in regions:
                    if region == 'image':
                        region_feature_list[0] = group_handle[vid][sub_group][region][0].squeeze()
                    elif region == 'bbox' or region == 'box':
                        region_feature_list[1] = group_handle[vid][sub_group][region][0].squeeze()
                    else:
                        bbox_idx = int(region[4:])
                        region_feature_list[bbox_idx] = group_handle[vid][sub_group][region][0].squeeze()
                feature_dict[vid][sub_group] = np.array(region_feature_list)
            else:
                feature_dict[vid][sub_group] = dict()
                datas = group_handle[vid][sub_group].keys()
                for data in datas:
                    if data == 'img_alignment':
                        img_alignment_rows = group_handle[vid][sub_group][data].keys()
                        feature_dict[vid][sub_group][data] = [[] for i in img_alignment_rows]
                        for img_alignment_row in img_alignment_rows:
                            int(img_alignment_row)
                            feature_dict[vid][sub_group][data][int(img_alignment_row)] = \
                                group_handle[vid][sub_group][data][img_alignment_row][:].tolist()
                    elif data == 'token':
                        token_list = group_handle[vid][sub_group][data][:].tolist()
                        feature_dict[vid][sub_group][data] = [str(token)[2:-1] for token in token_list]
                    else:
                        feature_dict[vid][sub_group][data] = group_handle[vid][sub_group][data][:]

    return feature_dict


def copy_all_group(root, feature_group):
    frame_grid_h5_list = glob.glob(os.path.join(root, '*.hdf5'))
    for frame_grid_h5 in frame_grid_h5_list:
        with h5py.File(frame_grid_h5, 'r') as sub_f:
            vid = frame_grid_h5.split('/')[-1][:11]
            feature_group.create_group(vid)
            for key in sub_f.keys():
                sub_f.copy(key, feature_group[vid])


def package_all_feature(save_path=FEATURE_PACKAGE_ROOT,
                        query_grid_root=GRID_FEATURE_ROOT_QUERY, frame_grid_root=GRID_FEATURE_ROOT_CLIP,
                        query_text_root=TEXT_FEATURE_ROOT_QUERY, subtitle_text_root=TEXT_FEATURE_ROOT_CLIP):
    f = h5py.File(os.path.join(save_path, 'feature.hdf5'), 'w')

    # query_grid_feature
    query_grid_feature = f.create_group('query_grid_feature')
    copy_all_group(query_grid_root, query_grid_feature)

    # frame_grid_feature
    frame_grid_feature = f.create_group('frame_grid_feature')
    copy_all_group(frame_grid_root, frame_grid_feature)

    # query_text_feature
    query_text_feature = f.create_group('query_text_feature')
    copy_all_group(query_text_root, query_text_feature)

    # subtitle_text_feature
    subtitle_text_feature = f.create_group('subtitle_text_feature')
    copy_all_group(subtitle_text_root, subtitle_text_feature)

    f.close()


if __name__ == '__main__':
    package_all_feature()