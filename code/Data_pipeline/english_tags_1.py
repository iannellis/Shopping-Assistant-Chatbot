"""Read the metadata from .json.gz files into a Pandas dataframe, filter out non-English
tagged items, and save into a pickle file."""
import tomllib
import os
import gzip
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

#-------------------------Work-performing Functions-------------------------------------
def read_meta(abo_listing_dir):
    """Read all the metadata in from the listing metadata dir and save in a Pandas
    dataframe. Expects the json files to be somewhere in abo_listing_dir compressed in
    gzip format."""
    dflist = []
    for root, dirs, files in os.walk(abo_listing_dir):
        fnames = [fname for fname in files if fname.endswith('.json.gz')]
        for fname in tqdm(fnames):
            with gzip.open(os.path.join(root, fname), 'rt') as f:
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

#-------------------------------Operate on metadata-------------------------------------
if __name__ == "__main__":
    print('1. Filter non-English tags out of metadata.')   
    
    with open('pipeline_config.toml', 'rb') as f:
        config = tomllib.load(f)
    
    working_dir = config['global']['working_dir']
    meta_save_prefix = config['global']['meta_save_prefix']
    abo_dataset_dir = config['global']['abo_dataset_dir']
    
    print('Reading in metadata from listings...')
    listings_dir = abo_dataset_dir + '/listings'
    pdf = read_meta(listings_dir)
    pdf = pdf.drop(columns=['model_number', 'color_code', 'node', 'item_dimensions',
                            'spin_id', '3dmodel_id', 'item_shape'])

    print('Filtering out rows without English tags...')
    pdf = pdf[[has_english_tag(pdf.loc[item_id]) for item_id in tqdm(pdf.index)]]
    print('Filtering non-English tags from remaining rows (by column)...')
    for col in tqdm(pdf.columns):
        for i in range(len(pdf[col])):
            pdf[col][i] = drop_non_eng_vals(pdf[col][i])

    print('Saving intermediate metadata results...')

    pdf.to_pickle(working_dir + '/' + meta_save_prefix + "preprocess-1.pkl")

