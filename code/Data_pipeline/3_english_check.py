"""Run English-language checks on the metadata using Google's MediaPipe and optionally
Google Cloud language detection.

Note that Google Cloud language detection requires a Google Cloud account with the
Cloud Translation API enabled, installation of the gcloud CLI, and logging in via the CLI: 
https://cloud.google.com/python/docs/setup#installing_the_cloud_sdk"""

from mediapipe.tasks import python
from mediapipe.tasks.python import text
from google.cloud import translate_v2 as translate

import pandas as pd
from tqdm import tqdm
import tomllib

#-------------------------Work-performing Functions-------------------------------------
def row_to_str(row):
    """Convert a metadata dataframe row to a string for use in the language detectors."""
    row_filtered = row.drop(labels=['brand', 'item_weight', 'model_name', 'product_type',
                                    'main_image_id', 'other_image_id', 'country', 
                                    'marketplace', 'domain_name', 'model_year']).dropna()
    text = []
    for item in row_filtered:
        if isinstance(item, list):
            text.extend(item)
        else:
            text.append(item)
    
    return ' '.join(text).replace('\n', ' ').replace('^', ' ').replace(',', ', ')

def mediapipe_detection(text_for_detection):
    """Run MediaPipe to detect the language of a row of data. Returns a dataframe with
    the language and confidence of the row by index."""
    base_options = python.BaseOptions(model_asset_path="../../assets/language_detector.tflite")
    options = text.LanguageDetectorOptions(base_options=base_options)
    mediapipe_detector = text.LanguageDetector.create_from_options(options)
    
    mediapipe_detection_results = {'languages': [], 'confidences': []}

    for item in tqdm(text_for_detection):
        mediapipe_result = mediapipe_detector.detect(item).detections
        if mediapipe_result:
            mediapipe_detection_results['languages'].append(mediapipe_result[0].language_code)
            mediapipe_detection_results['confidences'].append(mediapipe_result[0].probability)
        else:
            mediapipe_detection_results['languages'].append(None)
            mediapipe_detection_results['confidences'].append(None)
            
    return pd.DataFrame(mediapipe_detection_results)

def detect_language_google_cloud(text_for_detection, indexes):
    """Run Google Cloud language detection on the indicated rows of data."""
    translate_client = translate.Client()
    cloud_detection_results = {'index': [], 'languages': [], 'confidences': []}

    for idx in tqdm(indexes):
        text_item = text_for_detection[idx]
        google_cloud_detection = translate_client.detect_language(text_item)
        cloud_detection_results['index'].append(idx)
        cloud_detection_results['languages'].append(google_cloud_detection['language'])
        cloud_detection_results['confidences'].append(google_cloud_detection['confidence'])
        
    return pd.DataFrame(cloud_detection_results).set_index('index')

#-------------------------------Operate on metadata-------------------------------------
if __name__ == "__main__":
    print('3. Using a model (MediaPipe from Google the optionaly Google Cloud language detection), '
          'verify that every row of metadata is in English and filter out those that aren\'t.')
    
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    working_dir = config['global']['working_dir']
    meta_save_prefix = config['global']['meta_save_prefix']
    cloud_language_detection = config['english_check']['cloud_language_detection']
    
    print('Loading (supposed) English-language metadata with spaces in product_types...')
    pdf = pd.read_pickle(working_dir + '/' + meta_save_prefix + "preprocess-2.pkl")
    
    print('Converting metadata rows to text...')
    text_for_detection = [row_to_str(pdf.loc[item_id]) for item_id in tqdm(pdf.index)]
    
    print('Running MediaPipe language detection...')
    mediapipe_languages = mediapipe_detection(text_for_detection)

    if cloud_language_detection:
        indexes_for_cloud_detection = mediapipe_languages[mediapipe_languages['languages'] != 'en'].index
        print(f'Running cloud detection on {len(indexes_for_cloud_detection)} non-English MediaPipe results...')
        cloud_detection_results_df = detect_language_google_cloud(text_for_detection, indexes_for_cloud_detection)
       
        print('Saving cloud detection results...')
        cloud_detection_results_df.to_pickle(working_dir + meta_save_prefix + "cloud-language-detections.pkl")
        non_eng_idxs = cloud_detection_results_df[cloud_detection_results_df['languages'] != 'en'].index
    else:
        non_eng_idxs = mediapipe_languages[mediapipe_languages['languages'] != 'en'].index

    print('Dropping rows detected as non-English...')
    pdf = pdf.drop(pdf.index[non_eng_idxs])
    
    print('Saving verified English metadata...')
    pdf.to_pickle(working_dir + '/' + meta_save_prefix + "preprocess-3.pkl")