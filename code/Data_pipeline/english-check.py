from mediapipe.tasks import python
from mediapipe.tasks.python import text
from google.cloud import translate_v2 as translate

import pandas as pd
from tqdm import tqdm

working_dir = "../../../ShopTalk-blobs/ABO_dataset/"
meta_save_prefix = "abo-listings-"
cloud_language_detection = True

#-------------------------Work-performing Functions-------------------------------------
def row_to_text(row):
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
    
    return ' '.join(text).replace('\n', ' ')

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

print('Loading English-language metadata...')
pdf = pd.read_pickle(working_dir + meta_save_prefix + "english-tags.pkl")
print('Converting metadata rows to text...')
text_for_detection = [row_to_text(pdf.loc[item_id]) for item_id in tqdm(pdf.index)]
print('Running MediaPipe language detection...')
mediapipe_languages = mediapipe_detection(text_for_detection)

if cloud_language_detection:
    print('Running cloud detection on non-English MediaPipe results...')
    indexes_for_cloud_detection = mediapipe_languages[mediapipe_languages['languages'] != 'en'].index
    cloud_detection_results_df = detect_language_google_cloud(text_for_detection, indexes_for_cloud_detection)
    print('Saving cloud detection results...')
    cloud_detection_results_df.to_pickle(working_dir + meta_save_prefix + "cloud-language-detections.pkl")
    non_eng_idxs = cloud_detection_results_df[cloud_detection_results_df['languages'] != 'en'].index
else:
    non_eng_idxs = mediapipe_languages[mediapipe_languages['languages'] != 'en'].index

print('Dropping rows detected as non-English...')    
pdf = pdf.drop(pdf.index[non_eng_idxs])
print('Saving verified English metadata...')
pdf.to_pickle("../../../Capstone Project - ShopTalk/ABO_dataset/abo-listings-verified-english.pkl")