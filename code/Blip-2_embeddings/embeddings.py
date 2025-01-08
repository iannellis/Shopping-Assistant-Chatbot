# Uses the 3.11 environment locally

from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

import os
from tqdm import tqdm

model_paths = {'gs': '/mnt/d/marqo-gs-10m/model-saves/pretrain_1epoch.pt',
               'abo': '/mnt/d/abo-dataset/model_saves/pretrain_2epochs.pt'}

def run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset/', model_type='pretrain', 
                   device='cuda', save_path='/mnt/d/embeddings/', batch_size=64):
    """Run the data throught the multimodal (text + image) and text-only feature
    extractor of the BLIP-2 mmodel to create embeddings.

    Args:
        abo_dataset_dir (str, optional): The directory where the ABO dataset has been extracted to
        model_type (str, optional): Which version of the BLIP-2 model to use. Options are
                                    'coco,' 'pretrain,' 'gs,' and 'abo.' See documentation for
                                    details.
        device (str, optional): Device to load the model to. Usually 'cpu' or 'cuda'.
        save_path (str, optional): The directory to dumpt the embeddings into.
        batch_size (int, optional): The batch size for calculating the embeddings.
    """  
    images_dir = abo_dataset_dir + '/images/small'
    metadata_file = abo_dataset_dir + '/abo-listings-final-draft.pkl'
    image_metadata_file = abo_dataset_dir + '/images/metadata/images.csv'
    
    if model_type in ["pretrain", "coco"]:
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type=model_type, is_eval=True, device='cpu')
    else:
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type='pretrain', is_eval=True, device='cpu')
        model = torch.load(model_paths[model_type], weights_only=False)
    model.to(device)

    dataloader_multimodal = build_abo_dataloader_multimodal(
        images_dir, metadata_file, image_metadata_file, image_processor=vis_processors['eval'], 
        text_processor=txt_processors['eval'], batch_size=batch_size, num_workers=2)
    dataloader_text = build_abo_dataloader_text(
        metadata_file, text_processor=txt_processors['eval'], batch_size=batch_size, num_workers=2)
    
    embed_multimodal(model, dataloader_multimodal, device, save_path, model_type)
    embed_text(model, dataloader_text, save_path, model_type)
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
        
        self._reorg_metadata_columns()
        
    def _reorg_metadata_columns(self):
        """Put the metadata dataframe columns in an order we want them in for producing
        the text strings."""
        # drop item_keywords because there can be rediculously many
        self.metadata = self.metadata[['item_name', 'brand', 'model_name', 'model_year',
                                       'product_description', 'product_type', 'color',
                                       'fabric_type', 'style', 'material',
                                       'pattern', 'finish_type', 'bullet_point']]

    def __len__(self):
        return len(self.image_item_pairs)
    
    def __getitem__(self, idx: int):
        image_id = self.image_item_pairs.loc[idx, 'image_id']
        item_id = self.image_item_pairs.loc[idx, 'item_id']
        image_path = os.path.join(self.image_dir, self.image_metadata.loc[image_id, 'path'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(image)
        label = self._row_to_str(self.metadata.loc[item_id])
        label = self.text_processor(label)
        
        return image, label, image_id, item_id
    
    def _row_to_str(self, row):
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
    
def build_abo_dataloader_text(metadata_file: str, text_processor: callable,
                              batch_size=64, num_workers=2) -> DataLoader:
    """Load the data to build the text-only dataloader for the ABO dataset and build it."""
    metadata = pd.read_pickle(metadata_file)
    dataset_text = ABODataset_text(metadata, text_processor)
    dataloader_text = DataLoader(dataset=dataset_text, batch_size=batch_size,
                                 num_workers=num_workers)
    return dataloader_text

class ABODataset_text(ABODataset_multimodal):
    """The dataloader for ABO text-only data. Loads the text data for creating the
    embedding. Takes an integer index in the metadata dataframe then turns the row into
    a text string for creating embeddings."""
    def __init__(self, metadata: pd.DataFrame, text_processor: callable):
        self.metadata = metadata
        self.text_processor = text_processor
        
        self._reorg_metadata_columns()
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        label = self._row_to_str(self.metadata.iloc[idx])
        label = self.text_processor(label)
        item_id = self.metadata.index[idx]
        return label, item_id
        
def embed_multimodal(model, dataloader, device, save_path, model_type):
    """Enumerate through the multimodal dataloader and create the embeddings. Create a
    Pandas dataframe from them and save to a pickle file."""
    embeddings_dict = {'image_id': [], 'item_id': [], 'embedding': []}
    
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        images, labels, image_ids, item_ids = data
        embeddings_dict['image_id'].extend(image_ids)
        embeddings_dict['item_id'].extend(item_ids)
        
        images = images.to(device)
        samples = {"image": images, "text_input": labels}
        
        output = model.extract_features(samples, mode='multimodal')
        embeddings_dict['embedding'].extend(output.multimodal_embeds.detach().cpu().numpy())
        
        if (i+1) % 1000 == 0:
            embeddings_df = pd.DataFrame(embeddings_dict)
            embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_multimodal_' + str(i+1) + '.pkl')
            embeddings_dict = {'image_id': [], 'item_id': [], 'embedding': []}

    embeddings_df = pd.DataFrame(embeddings_dict)
    embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_multimodal_' + str(i+1) + '.pkl')
    return

def embed_text(model, dataloader, save_path, model_type):
    """Enumerate through the text dataloader and create the embeddings. Create a
    Pandas dataframe from them and save to a pickle file."""
    embeddings_dict = {'item_id': [], 'embedding': []}
    
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        labels, item_ids = data
        embeddings_dict['item_id'].extend(item_ids)
        
        samples = {"text_input": labels}
        
        output = model.extract_features(samples, mode='text')
        embeddings_dict['embedding'].extend(output.text_embeds.detach().cpu().numpy())

        if (i+1) % 500 == 0:
            embeddings_df = pd.DataFrame(embeddings_dict)
            embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_text_' + str(i+1) + '.pkl')
            embeddings_dict = {'item_id': [], 'embedding': []}

    embeddings_df = pd.DataFrame(embeddings_dict)
    embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '_text_' + str(i+1) + '.pkl')
    return