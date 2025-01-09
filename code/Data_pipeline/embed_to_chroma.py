"""Take the metadata and images, create multimodal (text + image) embeddings, and add
them to a chroma database. Then add the text-only strings to the database.

Note: must be run using the 3.11 Python environment
"""

from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

import os
from tqdm import tqdm
import pickle

import chromadb

def run_embeddings(abo_images_dir='/mnt/d/abo-dataset/images/small',
                   image_metadata_file='/mnt/d/abo-dataset/images/metadata/images.csv',
                   metadata_df_file='/mnt/d/abo-dataset/abo-listings-final-draft.pkl',
                   models_dir='/mnt/d/models/', model_selection='gs', device='cuda',
                   embeddings_save_path='/mnt/d/embeddings/', batch_size=64):
    """Run the data throught the multimodal (text + image) and text-only feature
    extractor of the BLIP-2 mmodel to create embeddings.

    Args:
        abo_images_dir (str, optional): The directory where the images of the ABO dataset are located
        image_metadata_file (str, optional): The path to the csv file mapping image_id to image file
        metadata_df_file (str, optional): The path to the pickle file containing the dataframe
                                          of the pre-processed metadata
        model_type (str, optional): Which version of the BLIP-2 model to use. Options are
                                    'coco,' 'pretrain,' 'gs,' and 'abo.' See documentation for
                                    details.
        device (str, optional): Device to load the model to. Usually 'cpu' or 'cuda'.
        embeddings_save_path (str, optional): The directory to dump the embeddings into.
        batch_size (int, optional): The batch size for calculating the embeddings.
    """
    if model_selection == 'coco':
        with open('blip-2-processors-coco.pkl', 'rb') as f:
            vis_processor, text_processor = pickle.load(f)
    else:
        with open('blip-2-processors-pretrain.pkl', 'rb') as f:
            vis_processor, text_processor = pickle.load(f)
    model = torch.load(models_dir + "/blip-2-" + model_selection + ".pt")
    model.to(device)

    dataloader_multimodal = build_abo_dataloader_multimodal(
        abo_images_dir, metadata_df_file, image_metadata_file, image_processor=vis_processor, 
        text_processor=text_processor, batch_size=batch_size, num_workers=2)
       
    embed_multimodal(model, dataloader_multimodal, device, embeddings_save_path, model_selection)
    return

def build_abo_dataloader_multimodal(images_dir: str, metadata_file: str, image_metadata_file: str, 
                         image_processor: callable, text_processor: callable, 
                         batch_size=64, num_workers=2) -> DataLoader:
    """Load the data to build the multimodal (image + text) dataloader for the ABO dataset
    and build it."""
    metadata = pd.read_pickle(metadata_file)
    image_metadata = pd.read_csv(image_metadata_file).set_index('image_id')
    image_item_pairs = abo_image_item_pairs(metadata)
    
    dataset_multimodal = ABODataset_multimodal(images_dir, metadata, image_metadata,
                                                   image_item_pairs, image_processor, text_processor)
    dataloader_multimodal = DataLoader(dataset=dataset_multimodal, batch_size=batch_size,
                                       num_workers=num_workers)
    
    return dataloader_multimodal

def abo_image_item_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    """Takes the ABO metadata extracts all the image_ids associated with each item_id,
    and produces image-item (image_id-item_id) pairs."""
    image_ids = []
    item_ids = []
    for item_id in metadata.index:
        main_image_id = metadata.loc[item_id, 'main_image_id']
        if not pd.isna(main_image_id):
            image_ids.append(main_image_id)
            item_ids.append(item_id)
        other_image_ids = metadata.loc[item_id, 'other_image_id']
        if isinstance(other_image_ids, list):
            for other_image_id in other_image_ids:
                image_ids.append(other_image_id)
                item_ids.append(item_id)
        elif not pd.isna(other_image_ids):
            image_ids.append(other_image_ids)
            item_ids.append(item_id)
    return pd.DataFrame({'image_id': image_ids, 'item_id': item_ids})

class ABODataset_multimodal(Dataset):
    """The dataloader for ABO multimodal data. Loads the multimodal data (image-item pairs)
    for creating the embedding. Takes an index in the image_item_pairs dataframe, then
    uses the corresponding image_id and item_id to load the image and produce a text string
    from the row of metadata.
    Note: modified from fine-tuning version"""
    def __init__(self, image_dir: str, metadata: pd.DataFrame,
                 image_metadata: pd.DataFrame, image_item_pairs: pd.DataFrame,
                 image_processor: callable, text_processor: callable):
        self.image_dir = image_dir
        self.metadata = metadata
        self.image_metadata = image_metadata
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.image_item_pairs = image_item_pairs
        
        self.metadata = reorg_metadata_columns(self.metadata)
        
    def __len__(self):
        return len(self.image_item_pairs)
    
    def __getitem__(self, idx: int):
        image_id = self.image_item_pairs.loc[idx, 'image_id']
        item_id = self.image_item_pairs.loc[idx, 'item_id']
        image_path = os.path.join(self.image_dir, self.image_metadata.loc[image_id, 'path'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(image)
        label = row_to_str(self.metadata.loc[item_id])
        label = self.text_processor(label)
        
        return image, label, image_id, item_id

def reorg_metadata_columns(metadata):
        """Put the metadata dataframe columns in an order we want them in for producing
        the text strings."""
        # drop item_keywords because there can be rediculously many
        metadata = metadata[['item_name', 'brand', 'model_name', 'model_year',
                            'product_description', 'product_type', 'color',
                            'fabric_type', 'style', 'material',
                            'pattern', 'finish_type', 'bullet_point']]
    
def row_to_str(row):
    """Convert a row of metadata to a string the model can consume."""
    row_filtered = row.dropna()
    text = []
    for row_item in row_filtered:
        if isinstance(row_item, list):
            for list_item in row_item:
                text.append(str(list_item) + ';')
        else:
            text.append(str(row_item) + ';')
    
    return ' '.join(text).replace('\n', ' ').replace('^', ' ').replace(',', ', ')

def embed_multimodal(model, dataloader, device, save_path, model_type):
    """Enumerate through the multimodal dataloader and create the embeddings. Create a
    Pandas dataframe from them and save to a pickle file.
    
    Note that saving more of the embedding (we save 4 out of 32 rows) requires more memory.
    Depending on the memory available, you may need to increase the frequency of saves.
    """
    embeddings_dict = {'image_id': [], 'item_id': [], 'embedding': []}
    
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        images, labels, image_ids, item_ids = data
        embeddings_dict['image_id'].extend(image_ids)
        embeddings_dict['item_id'].extend(item_ids)
        
        images = images.to(device)
        samples = {"image": images, "text_input": labels}
        
        output = model.extract_features(samples, mode='multimodal')
        embeddings_dict['embedding'].extend(output.multimodal_embeds[:,:4,:].detach().cpu().numpy())
        
        if (i+1) % 1000 == 0:
            embeddings_df = pd.DataFrame(embeddings_dict)
            embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_multimodal.pkl')

    embeddings_df = pd.DataFrame(embeddings_dict)
    embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_multimodal_' + str(i+1) + '.pkl')
    return

def load_multimodal_to_chroma(collection, embeddings_dir, model_selection):
    """A wrapper for the next function. For each embedding file, load it into the
    Chroma database. We have a start_id to keep track of the ids (ints) across those 
    multiple files."""
    start_id = 0
    embeddings_dir = 'D:/embeddings/'
    for file in os.listdir(embeddings_dir):
        if file.startswith('embeddings_'+model_selection+'_multimodal'):
            print(file)
            # start_id to keep track of the ids (ints) across files when adding to the database
            start_id = embed_multimodal(collection, embeddings_dir + file, start_id)

def load_multimodal_to_chroma_helper(collection, file, start_id):
    """Load the multimodal embeddings into the Chroma database. We have multiple files,
    so this function is called on a per-file basis. We have a start_id to keep track of 
    the ids (ints) across those multiple files.
    
    Chroma has a batch size limit due to the underlying sqlite database. Therefore, we 
    need to add the embeddings in batches. """
    batch_size = 4000
    embeddings_df = pd.read_pickle(file)
    n_rows = len(embeddings_df)
    n_batches = (n_rows-1)//batch_size + 1
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i+1) * batch_size, n_rows)
        embeddings = np.stack(embeddings_df.iloc[start:end]['embedding'])
        embeddings = embeddings[:,:4,:]
        embeddings = list(embeddings.reshape((len(embeddings), -1)))
        metadatas = []
        for i in range(start, end):
            image_id = embeddings_df.loc[i, 'image_id']
            item_id = embeddings_df.loc[i, 'item_id']
            metadatas.append({'image_id': image_id, 'item_id': item_id})
        collection.add(embeddings=embeddings, metadatas=metadatas, ids=[str(i + start_id) for i in range(start, end)])
    return end + start_id

def load_text_to_chroma(collection, metadata_df_file):
    """Takes all the metadata, converts each row to a string, and adds them all to the
    text-only database."""
    metadata_df = pd.read_pickle(metadata_df_file)
    metadata_df = reorg_metadata_columns(metadata_df)
    batch_size = 1000
    n_rows = len(metadata_df)
    n_batches = (n_rows-1)//batch_size + 1
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i+1) * batch_size, n_rows)
        rows_to_add = []
        for i in range(start, end):
            row_str = row_to_str(metadata_df.iloc[i])
            rows_to_add.append(row_str)
        collection.add(documents=rows_to_add, ids=list(metadata_df.index[start:end]))
        
embeddings_dir='/mnt/d/embeddings/'
model_selection='gs'
print('Running multimodal embeddings...')
run_embeddings(abo_images_dir='/mnt/d/abo-dataset/images/small',
                   image_metadata_file='/mnt/d/abo-dataset/images/metadata/images.csv',
                   metadata_df_file='/mnt/d/abo-dataset/abo-listings-final-draft.pkl',
                   models_dir='/mnt/d/models/', model_selection=model_selection, device='cuda',
                   embeddings_save_path=embeddings_dir, batch_size=64)
print('Loading the multimodal embeddings into the Chroma database...')
client=chromadb.PersistentClient(path="/mnt/d/chroma")
collection=client.get_or_create_collection(name='blip_2_'+model_selection+'_multimodal')
load_multimodal_to_chroma(collection, embeddings_dir, model_selection)
print('Loading the items into the text-only Chroma database...')
collection=client.get_or_create_collection(name="text_only")