from json_utils import load_annotation_list
from downloader import download_data
from subtitle_preprocessor import preprocess_subtitles
from frame_extractor import sampling_frame
from grid_feature_extractor import extract_grid_feature, separate_frame_grid_feature
from text_feature_extractor import extract_text_feature
from clip_separator import separate_into_clip
from feature_packager import package_all_feature
from annotation_preprocessor import preprocess_annotation, package_annotation


if __name__ == '__main__':

    # Download
    download_data() # Deprecated, better to use App "Downie" for Mac (search in "macwk.com"), ensure that the file type is .mp4 and .srt

    # Extract Frames
    sampling_frame() # Sampling frame from raw videos

    # Extract Grid Features
    extract_grid_feature() # Extract visual features from frames and query images

    # Load annotations
    print('Start loading annotations...')
    annotation_list = load_annotation_list()
    seg_info = dict()
    for anno in annotation_list:
        seg_info[anno[0]['videoID']] = anno[1]['segInfo']

    # Preprocess and package annotations
    preprocess_annotation() # Pre-process annotations for packaging
    package_annotation() # Packaging annotations, converting from raw to our final json version

    # Separate Grid Features
    separate_frame_grid_feature(seg_info) # Separate the extracted visual features by segments

    # PreProcess subtitles
    preprocess_subtitles(seg_info) # Process raw srt file

    # Extract Text Features
    extract_text_feature() # Extract textual features from subtitles and query text

    # Separate into Clips
    separate_into_clip(seg_info) # Re-allocate features by clips (per 1.5s)

    # Package all feature
    package_all_feature() # Package all the features


