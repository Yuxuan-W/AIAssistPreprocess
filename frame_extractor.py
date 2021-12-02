from joblib import delayed, Parallel
import os
import glob
import cv2
import shutil
from tqdm import tqdm
from configs.preprocess_configs import FRAME_WIDTH, FRAME_HEIGHT, DOWNLOAD_ROOT, FRAME_ROOT, FRAME_RATE
from json_utils import load_annotation_list


def extract_video_opencv(v_path, f_root):
    '''v_path: single video path;
       f_root: root to store frames'''
    vid = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, vid + '_temp')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir_final = os.path.join(f_root, vid)

    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    interval = 1 / FRAME_RATE
    # float
    if (width == 0) or (height == 0):
        print(v_path, 'not successfully loaded, drop ..');
        return
    new_dim = (FRAME_WIDTH, FRAME_HEIGHT)

    success, image = vidcap.read()
    count = 0
    interval_index = 0
    while success:
        time_second = count / FRAME_RATE

        if time_second >= interval_index * interval:
            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, '%.3f.jpg' % time_second), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])  # quality from 0-100, 95 is default, high is good
            interval_index += 1

        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(vid + " finished")
    os.rename(out_dir, out_dir_final)


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return int(target * w / h), int(target)
    else:
        return int(target), int(target * h / w)


def separate_frame(segment, f_root=FRAME_ROOT):
    video_list = glob.glob(os.path.join(f_root, '*/'))
    for video in video_list:
        vid = video[-12:-1]
        seg = segment[vid]
        frame_list = glob.glob(os.path.join(video, '*.jpg'))
        # make segment dirs
        seg_dir_list = []
        for seg_idx in range(len(seg)):
            seg_dir = video + vid + '_' + str(seg_idx)
            seg_dir_list.append(seg_dir)
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)
        # move images
        for frame in frame_list:
            time = float(frame.split('/')[-1][:-4])
            for seg_idx in range(len(seg)):
                if seg[seg_idx][0] <= time <= seg[seg_idx][1]:
                    os.replace(frame, os.path.join(seg_dir_list[seg_idx], frame.split('/')[-1]))


def sampling_frame(v_root=DOWNLOAD_ROOT, f_root=FRAME_ROOT):
    print('Start extracting videos from %s, frame save to %s...' % (v_root, f_root))

    if not os.path.exists(f_root):
        os.makedirs(f_root)
    video_list = glob.glob(os.path.join(v_root, '*.mp4'))
    target_list = []
    for video in video_list:
        if not os.path.exists(f_root + '/' + video[-15:-4]):
            target_list.append(video)

    temp_folder_list = glob.glob(os.path.join(f_root, '*_temp'))
    for temp_folder in temp_folder_list:
        shutil.rmtree(temp_folder)

    # sampling from video
    Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root)
                        for p in tqdm(target_list, desc="Loop over videos"))


if __name__ == '__main__':
    sampling_frame()

    seg_info = dict()
    annotation_list = load_annotation_list()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']
    separate_frame(seg_info)
