"""Run images and corresponding product_types through Llama-vision to check if the items
are categorized correctly. Remove the product_type if not.

Note that this file has a restart function to restart a run that crashed.

Uses the 3.12 environment locally
"""

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
from PIL import Image

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16)

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    do_sample=True,
    temperature=0.01, # model can get inconsistent if we use defaults for last 2 options
    top_p=0
)
processor = AutoProcessor.from_pretrained(model_id)

shoptalk_blobs_dir = "../ShopTalk-blobs/"

def load_listings():
    """Load the metadata and fix the product_type so that the words are separated by sapces."""
    pdf = pd.read_pickle(shoptalk_blobs_dir + "ABO_dataset/abo-listings-english-tags.pkl")
    pdf['product_type'] = pdf['product_type'].str.replace('_', ' ')
    pdf.loc[pdf['product_type'] == 'FINERING', 'product_type'] = 'FINE RING'
    pdf.loc[pdf['product_type'] == 'FINENECKLACEBRACELETANKLET', 'product_type'] = 'FINE NECKLACE BRACELET ANKLET'
    pdf.loc[pdf['product_type'] == 'FINEEARRING', 'product_type'] = 'FINE EARRING'
    pdf.loc[pdf['product_type'] == 'FASHIONNECKLACEBRACELETANKLET', 'product_type'] = 'FASHION NECKLACE BRACELET ANKLET'
    pdf.loc[pdf['product_type'] == 'FINEOTHER', 'product_type'] = 'FINE OTHER'
    pdf.loc[pdf['product_type'] == 'FASHIONEARRING', 'product_type'] = 'FASHION EARRING'
    pdf.loc[pdf['product_type'] == 'SHOWERHEAD', 'product_type'] = 'SHOWER HEAD'
    pdf.loc[pdf['product_type'] == 'FASHIONOTHER', 'product_type'] = 'FASHION OTHER'
    pdf['product_type'] = pdf['product_type'].str.replace('ABIS ', '')
    return pdf

def run_dataset_check(pdf_has_images, iloc_start, iloc_end, image_path_prefix, save_name):
    """Run the dataset through the model, checking if each image matches the product_type.

    Args:
        pdf_has_images: the dataframe with the metadata just for products with images
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        save_name (str): the file name for saving the results of the run
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
            save_dataset(image_categroy_match, save_name)

    save_dataset(image_categroy_match, save_name)

def resume_dataset_check(pdf_has_images, iloc_start, iloc_end, image_path_prefix, save_name):
    """Resume the run of the dataset through the model, checking if each image matches 
    the product_type.

    Args:
        pdf_has_images: the dataframe with the metadata just for products with images
        iloc_start (int): the integer index in pdf_has_images to start the run at
        iloc_end (int): the integer index in pdf_has_images to end the run at
        image_path_prefix (str): the path to the directory with the images in it
        save_name (str): the file name for saving the results of the run
    """
    image_meta_df = load_image_meta()    

    image_categroy_match_df = pd.read_pickle(shoptalk_blobs_dir + 'ABO_dataset/' + save_name)
    
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
            
        if i % 1000 == 0:
            save_dataset(image_categroy_match, save_name)

    save_dataset(image_categroy_match, save_name)

def load_image_meta():
    """Load the image metadata. Tells us where each image is given its image_id."""
    image_meta_path = shoptalk_blobs_dir + 'ABO_dataset/images/metadata/images.csv'
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
            {"type": "text", "text": f"Answer yes or no without details: Is something that can be categorized as {product_type} in this image?"}
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

def save_dataset(image_category_match, save_name):
    """Save the current progress."""
    image_categroy_match_df = pd.DataFrame(image_category_match).set_index('image_id')
    image_categroy_match_df.to_pickle(shoptalk_blobs_dir + "ABO_dataset/" + save_name)