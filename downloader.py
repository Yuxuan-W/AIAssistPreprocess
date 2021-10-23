from __future__ import unicode_literals

import os

from joblib import Parallel, delayed
from tqdm import tqdm
from configs.preprocess_configs import NUM_JOBS
from json_utils import load_annotation_list
import youtube_dl


def download_single_video(vid, save_path='./data/download/'):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': save_path + '%(id)s.%(ext)s',
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        # 'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegSubtitlesConvertor',
            'format': 'srt'
        }]
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=' + vid])


def download_data():
    # Parallel(n_jobs=NUM_JOBS)(delayed(download_single_video)(annotation[0]['videoID'])
    #                     for annotation in tqdm(annotation_list, desc="Downloading from Youtube"))
    # for annotation in tqdm(annotation_list, desc="Downloading from Youtube"):
    #     download_single_video(annotation[0]['videoID'])
    with open('vidlist.txt') as f:
        vid_list = f.readlines()
    for vid in tqdm(vid_list, desc='Downloading from Youtube'):
        vid = vid[0:11]
        path = 'data/download/' + vid + '.mp4'
        if not os.path.exists(path):
            download_single_video(vid)


if __name__ == '__main__':
    annotation_list = load_annotation_list()
    download_data()
