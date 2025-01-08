from mediapipe.tasks import python
from mediapipe.tasks.python import text

import tarfile
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

listing_tarfile = "../../ABO_dataset/abo-listings.tar"
working_dir = "../../../ShopTalk-blobs/ABO_dataset/"
meta_save_prefix = "abo-listings"

def read_json_from_tar(tar_file):
    """Read all the metadata in from the listing tarfile and save in a Pandas dataframe."""
    dflist = []
    with tarfile.open(tar_file, 'r') as tar:
        for member in tqdm(tar.getmembers()):
            if member.name.endswith('json.gz'):
                #print(f"Reading {member.name}...")
                # Extract the gz file in memory
                f = tar.extractfile(member)
                if f is not None:
                  with gzip.open(f, 'rt') as f:
                    df = pd.read_json(f, lines=True)
                    dflist.append(df)
    pdf = pd.concat(dflist).set_index('item_id')
    return pdf

def has_english_tag(row):
    """Check if a row has an English tag in it."""
    for value in row:
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            for item in value:
                if 'language_tag' in item and item['language_tag'].startswith('en_'):
                    return True
    return False

def drop_non_eng_vals(value):
    """Only keep values with English language tags or no language tags."""
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        english_items = []
        for item in value:
            if 'language_tag' in item and item['language_tag'].startswith('en_') \
              or 'language_tag' not in item and 'value' in item:
                english_items.append(item['value'])
        if len(english_items) == 1:
            return english_items[0]
        return english_items
    return value


print('Reading in data from tarfile...')
pdf = read_json_from_tar(listing_tarfile)
pdf = pdf.drop(columns=['model_number', 'color_code', 'node', 'item_dimensions',
                        'spin_id', '3dmodel_id', 'item_shape'])

print('Filtering non-English values...')
pdf = pdf[[has_english_tag(pdf.loc[item_id]) for item_id in tqdm(pdf.index)]]
for col in tqdm(pdf.columns):
    for i in range(len(pdf[col])):
        pdf[col][i] = drop_non_eng_vals(pdf[col][i])

print('Saving intermediate metadata results...')
pdf.to_pickle(working_dir + meta_save_prefix + "-english-tags.pkl")

