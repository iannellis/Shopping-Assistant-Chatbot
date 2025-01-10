"""Run images and corresponding product_types through Llama-vision to check if the items
are categorized correctly. Remove the product_type if not.

Note that this file has a restart function to restart a run that crashed, but it is
not yet setup to use it if the file is run directly.

This code almost completely comes from the Llama_data_checks directory."""

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import tomllib
import gzip

import warnings
warnings.filterwarnings('ignore')

#-------------------------Work-performing Functions-------------------------------------
def run_dataset_check(model, processor, pdf_has_images, image_meta_df, iloc_start, iloc_end,
                      image_path_prefix, checkpoint_fpath):
    """Run the dataset through the model, checking if each image matches the product_type.

    Args:
        model: the LLM-vision model to use
        processor: the pre- and post-processor of the text and image for the model
        pdf_has_images: the dataframe with the metadata just for products with images
        image_meta_df: maps image_ids to file locations
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        checkpoint_name (str): the file name for saving the results of the run
    """
    image_categroy_match = {'image_id': [], 'item_id': [], 'product_type': [], 'match': []}
    for i, item_id in enumerate(tqdm(pdf_has_images.index[iloc_start:iloc_end])):
        row = pdf_has_images.loc[item_id]
        product_type = row['product_type']
        image_ids = get_all_image_ids(row)
        
        for image_id in image_ids:
            image_path = image_path_prefix + '/' + image_meta_df.loc[image_id, 'path']
            image = Image.open(image_path)
            match = llama_check_image_category(model, processor, product_type, image)
            image_categroy_match['image_id'].append(image_id)
            image_categroy_match['item_id'].append(item_id)
            image_categroy_match['product_type'].append(product_type)
            image_categroy_match['match'].append(match)
            
        if (i+1) % 1000 == 0:
            write_checkpoint(image_categroy_match, checkpoint_fpath)

    write_checkpoint(image_categroy_match, checkpoint_fpath)
    
    return pd.DataFrame(image_categroy_match).set_index('image_id')

def resume_dataset_check(model, processor, pdf_has_images, image_meta_df, iloc_start,
                         iloc_end, image_path_prefix, checkpoint_fpath):
    """Resume the run of the dataset through the model, checking if each image matches 
    the product_type.

    Args:
        model: the LLM-vision model to use
        processor: the pre- and post-processor of the text and image for the model
        pdf_has_images: the dataframe with the metadata just for products with images
        image_meta_df: maps image_ids to file locations
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        checkpoint_name (str): the file name for saving the results of the run
    """
    image_categroy_match_df = pd.read_pickle(checkpoint_fpath)
    
    image_categroy_match = image_categroy_match_df.to_dict(orient='list')
    image_categroy_match['image_id'] = image_categroy_match_df.to_dict(orient='split')['index']
    
    resume_iter = len(image_categroy_match_df['item_id'].unique())
    print(f'Resuming from iteration: {resume_iter}')
    for i, item_id in enumerate(tqdm(pdf_has_images.index[iloc_start:iloc_end])):
        if i<resume_iter:  # just for tqdm progress bar (time won't be correct though)
            continue
    
        row = pdf_has_images.loc[item_id]
        product_type = row['product_type']
        image_ids = get_all_image_ids(row)
        
        for image_id in image_ids:
            image_path = image_path_prefix + '/' + image_meta_df.loc[image_id, 'path']
            image = Image.open(image_path)
            match = llama_check_image_category(model, processor, product_type, image)
            image_categroy_match['image_id'].append(image_id)
            image_categroy_match['item_id'].append(item_id)
            image_categroy_match['product_type'].append(product_type)
            image_categroy_match['match'].append(match)
            
        if (i+1) % 1000 == 0:
            write_checkpoint(image_categroy_match, checkpoint_fpath)

    write_checkpoint(image_categroy_match, checkpoint_fpath)
    
    return pd.DataFrame(image_categroy_match).set_index('image_id')

def get_all_image_ids(row):
    """Given a metadata row, get all the images_ids from it."""
    image_ids = []
    if isinstance(row['main_image_id'], str):
        image_ids.append(row['main_image_id'])
    if isinstance(row['other_image_id'], list):
        image_ids.extend(row['other_image_id'])
    
    return image_ids

def llama_check_image_category(model, processor, product_type, image):
    """Given a product_type and a PIL image, run both through the model to check whether
    they match."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Answer yes or no without details: Is something "
                                     f"that can be categorized as {product_type} in this image?"}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_length=100)
    return processor.decode(output[0])[-14:-11].strip()

def write_checkpoint(image_category_match, checkpoint_fpath):
    """Save the current progress."""
    image_categroy_match_df = pd.DataFrame(image_category_match).set_index('image_id')
    image_categroy_match_df.to_pickle(checkpoint_fpath)

def get_mismatch_item_ids(image_category_match_df):
    """Declare an item to have an incorrect product type if Llama decided that the
    majority of the images don't match it"""
    mismatch_df = image_category_match_df[image_category_match_df['match']!='Yes']
    mismatch_item_ids = []
    for item_id in tqdm(mismatch_df['item_id'].unique()):
        item_match = image_category_match_df[image_category_match_df['item_id']==item_id]['match']
        yes_no_counts = item_match.groupby(item_match).size()
        if 'No' not in yes_no_counts.index:
            continue
        elif 'Yes' not in yes_no_counts.index or yes_no_counts['No'] >= yes_no_counts['Yes']:
            mismatch_item_ids.append(item_id)
    return mismatch_item_ids


#-------------------------------Run the operation---------------------------------------
if __name__ == "__main__":
    print('4. Verify product_type is correct for each item, and set to null if not.')
    
    with open('pipeline_config.toml', 'rb') as f:
        config = tomllib.load(f)
        
    working_dir = config['global']['working_dir']
    meta_save_prefix = config['global']['meta_save_prefix']
    abo_dataset_dir = config['global']['abo_dataset_dir']
    checkpoint_name = config['product_type_verification']['checkpoint_name']

    print('Loading LLM-Vision model...')
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        do_sample=True, # model can get inconsistent if we use defaults for last 3 options
        temperature=0.01, 
        top_p=0
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print('Loading verified English metadata...')
    pdf = pd.read_pickle(working_dir + '/' + meta_save_prefix + "preprocess-3.pkl")
    pdf_has_images = pdf[~pdf['main_image_id'].isna() | ~pdf['other_image_id'].isna()]
    n_products = len(pdf_has_images)
    
    print('Loading image metadata...')
    try:
        image_meta_path = abo_dataset_dir + '/images/metadata/images.csv'
        image_meta_df = pd.read_csv(image_meta_path).set_index('image_id')
    except FileNotFoundError:
        image_meta_path = abo_dataset_dir + '/images/metadata/images.csv.gz'
        image_meta_df = pd.read_csv(image_meta_path).set_index('image_id')
    
    print('Verifying product_type using an LLM-Vision model...')
    image_path_prefix = abo_dataset_dir + '/images/small/'
    checkpoint_fpath = working_dir + '/' + checkpoint_name
    image_category_match_df = run_dataset_check(model, processor, pdf_has_images,
                                                image_meta_df, iloc_start=0, 
                                                iloc_end=n_products, 
                                                image_path_prefix=image_path_prefix,
                                                checkpoint_fpath=checkpoint_fpath)
    
    mismatch_item_ids = get_mismatch_item_ids(image_category_match_df)
    print(f"Setting product_type to null for {len(mismatch_item_ids)} items where the LLM-Vision model declared a majority of images don't match...")
    pdf.loc[mismatch_item_ids, 'product_type'] = np.nan
    
    print("Saving final metadata results...")
    pdf.to_pickle(working_dir + '/' + meta_save_prefix + "preprocess-4.pkl")
