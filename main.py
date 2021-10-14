from json_utils import load_annotation_list
from downloader import download_data
from subtitle_preprocessor import preprocess_subtitles
from frame_extractor import sampling_frame
from grid_feature_extractor import extract_grid_feature
from text_feature_extractor import extract_text_feature
from configs.preprocess_configs import ANNOTATION_ROOT, DOWNLOAD_ROOT, FRAME_ROOT, \
    SUBTITLE_ROOT, GRID_FEATURE_ROOT_FRAME, GRID_FEATURE_ROOT_QUERY

if __name__ == '__main__':

    # Load annotations
    print('Start loading annotations...')
    annotation_list = load_annotation_list()
    seg_info = dict()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']

    # Download
    download_data(annotation_list)

    # PreProcess subtitles
    preprocess_subtitles(seg_info)

    # Extract Frames
    sampling_frame(seg_info)

    # Extract Grid Features
    extract_grid_feature()

    # Extract Text Features
    extract_text_feature()
