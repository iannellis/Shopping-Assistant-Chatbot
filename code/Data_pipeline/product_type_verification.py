"""Run images and corresponding product_types through Llama-vision to check if the items
are categorized correctly. Remove the product_type if not.

Note that this file has a restart function to restart a run that crashed.

This code almost completely comes from the Llama_data_checks directory."""

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

abo_dir = "../../../ShopTalk-blobs/ABO_dataset/"
working_dir = "../../../ShopTalk-blobs/ABO_dataset/"
meta_save_prefix = "abo-listings-"

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

def run_dataset_check(pdf_has_images, iloc_start, iloc_end, image_path_prefix, checkpoint_name):
    """Run the dataset through the model, checking if each image matches the product_type.

    Args:
        pdf_has_images: the dataframe with the metadata just for products with images
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        checkpoint_name (str): the file name for saving the results of the run
    """
    image_meta_df = load_image_meta()    

    image_categroy_match = {'image_id': [], 'item_id': [], 'product_type': [], 'match': []}
    for i, item_id in enumerate(tqdm(pdf_has_images.index[iloc_start:iloc_end])):
        row = pdf_has_images.loc[item_id]
        product_type = row['product_type']
        image_ids = get_all_image_ids(row)
        
        for image_id in image_ids:
            image_path = image_path_prefix + '/' + image_meta_df.loc[image_id, 'path']
            image = Image.open(image_path)
            match = llama_check_image_category(product_type, image)
            image_categroy_match['image_id'].append(image_id)
            image_categroy_match['item_id'].append(item_id)
            image_categroy_match['product_type'].append(product_type)
            image_categroy_match['match'].append(match)
            
        if i % 1000 == 0:
            write_checkpoint(image_categroy_match, checkpoint_name)

    write_checkpoint(image_categroy_match, checkpoint_name)
    
    return pd.DataFrame(image_categroy_match).set_index('image_id')

def resume_dataset_check(pdf_has_images, iloc_start, iloc_end, image_path_prefix, checkpoint_name):
    """Resume the run of the dataset through the model, checking if each image matches 
    the product_type.

    Args:
        pdf_has_images: the dataframe with the metadata just for products with images
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        checkpoint_name (str): the file name for saving the results of the run
    """
    image_meta_df = load_image_meta()    

    image_categroy_match_df = pd.read_pickle(working_dir + '/' + checkpoint_name)
    
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
            match = llama_check_image_category(product_type, image)
            image_categroy_match['image_id'].append(image_id)
            image_categroy_match['item_id'].append(item_id)
            image_categroy_match['product_type'].append(product_type)
            image_categroy_match['match'].append(match)
            
        if (i+1) % 1000 == 0:
            write_checkpoint(image_categroy_match, checkpoint_name)

    write_checkpoint(image_categroy_match, checkpoint_name)
    
    return pd.DataFrame(image_categroy_match).set_index('image_id')

def load_image_meta():
    """Load the image metadata. Tells us where each image is given its image_id."""
    image_meta_path = abo_dir + '/images/metadata/images.csv'
    image_meta_df = pd.read_csv(image_meta_path).set_index('image_id')
    return image_meta_df

def get_all_image_ids(row):
    """Given a metadata row, get all the images_ids from it."""
    image_ids = []
    if isinstance(row['main_image_id'], str):
        image_ids.append(row['main_image_id'])
    if isinstance(row['other_image_id'], list):
        image_ids.extend(row['other_image_id'])
    
    return image_ids

def llama_check_image_category(product_type, image):
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

def write_checkpoint(image_category_match, checkpoint_name):
    """Save the current progress."""
    image_categroy_match_df = pd.DataFrame(image_category_match).set_index('image_id')
    image_categroy_match_df.to_pickle(working_dir + '/' + checkpoint_name)

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

print('Loading verified English metadata...')
pdf = pd.read_pickle(working_dir + '/' + meta_save_prefix + "/preprocess-3.pkl")
pdf_has_images = pdf[~pdf['main_image_id'].isna() | ~pdf['other_image_id'].isna()]
n_products = len(pdf_has_images)
print('Verifying product_type using a Llama-vision model...')
image_category_match_df = run_dataset_check(pdf_has_images, 0, n_products, 'D:/images/small/', 'abo-category-check.pkl')
print("Dropping product_type for items where Llama-vision declared a majority of images don't match...")
mismatch_item_ids = get_mismatch_item_ids(image_category_match_df)
pdf.loc[mismatch_item_ids, 'product_type'] = np.nan
print("Saving intermediate metadata results...")
pdf.to_pickle(working_dir + '/' + meta_save_prefix + "/preprocess-4.pkl")
