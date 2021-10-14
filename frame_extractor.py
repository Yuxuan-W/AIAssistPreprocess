from joblib import delayed, Parallel
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from json_utils import load_annotation_list
from configs.preprocess_configs import FRAME_WIDTH, FRAME_HEIGHT, DOWNLOAD_ROOT, FRAME_ROOT


def extract_video_opencv(v_path, f_root, segments, sample_time_list):
    '''v_path: single video path;
       f_root: root to store frames'''
    vid = os.path.basename(v_path)[0:-4]
    video_sample_time_list = sample_time_list[vid]
    video_segment = segments[vid]

    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = 1 / fps
    # float
    if (width == 0) or (height == 0):
        print(v_path, 'not successfully loaded, drop ..');
        return
    new_dim = (FRAME_WIDTH, FRAME_HEIGHT)

    success, image = vidcap.read()
    count = 1
    segment_index = 0
    interval_index = 0
    image_index = 0
    while success and segment_index < len(video_segment):
        time_second = count / fps

        if time_second >= interval_index * interval:
            out_dir = os.path.join(f_root, vid) + '_' + str(segment_index)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, 'image_%d.jpg' % image_index), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])  # quality from 0-100, 95 is default, high is good
            image_index += 1
            interval_index += 1

        if time_second >= video_segment[segment_index][1]:
            segment_index += 1
            image_index = 0

        success, image = vidcap.read()
        count += 1
    vidcap.release()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return int(target * w / h), int(target)
    else:
        return int(target), int(target * h / w)


def calculate_sample_time(video_seg_list):
    sample_time_list = dict()
    for video in video_seg_list.items():
        video_sample_time_list = []
        for seg in video[1]:
            video_sample_time_list.append((np.linspace(seg[0], seg[1], num=8, endpoint=False) + (seg[1] - seg[0]) / 16)
                                          .astype(int).tolist())
        sample_time_list[video[0]] = video_sample_time_list
    return sample_time_list


def sampling_frame(segment, v_root=DOWNLOAD_ROOT, f_root=FRAME_ROOT):
    print('Start extracting videos from %s, frame save to %s...' % (v_root, f_root))

    if not os.path.exists(f_root):
        os.makedirs(f_root)
    video_list = glob.glob(os.path.join(v_root, '*.mp4'))
    sample_time_list = calculate_sample_time(segment)

    # sampling from video
    Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root, segment, sample_time_list)
                        for p in tqdm(video_list, desc="Loop over videos"))


if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    seg_info = dict()
    annotation_list = load_annotation_list()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']

    sampling_frame(seg_info)
