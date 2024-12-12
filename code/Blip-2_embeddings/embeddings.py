from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

import os
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

model_paths = {'gs': '/mnt/d/marqo-gs-10m/model-saves/pretrain_1epoch.pt',
               'abo': '/mnt/d/abo-dataset/model_saves/pretrain_2epochs.pt'}

def run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset', model_type='pretrain', 
                   device='cuda', save_path='/mnt/d/embeddings'):
    logging.basicConfig(filename=save_path + '/' + model_type + '.log', level=logging.INFO)
    images_dir = abo_dataset_dir + '/images/small'
    metadata_file = abo_dataset_dir + '/abo-listings-final-draft.pkl'
    image_metadata_file = abo_dataset_dir + '/images/metadata/images.csv'
    
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device='cpu')
    if model_type != 'pretrain':
        model = torch.load(model_paths[model_type], weights_only=False)
    model.to(device)

    dataloader_multimodal = build_abo_dataloader_multimodal(
        images_dir, metadata_file, image_metadata_file, image_processor=vis_processors['train'], 
        text_processor=txt_processors['train'], batch_size=64, num_workers=2)
    dataloader_text = build_abo_dataloader_text(
        metadata_file, text_processor=txt_processors['train'], batch_size=64, num_workers=2)
    
    multimodal_embeddings_df = embed_multimodal(model, dataloader_multimodal, device)
    text_embeddings_df = embed_text(model, dataloader_text)
    embeddings_df = pd.concat([multimodal_embeddings_df, text_embeddings_df], ignore_index=True)
    embeddings_df.to_pickle(save_path + '/embeddings_' + model_type + '.pkl')
    
    return

def build_abo_dataloader_multimodal(images_dir: str, metadata_file: str, image_metadata_file: str, 
                         image_processor: callable, text_processor: callable, 
                         batch_size=64, num_workers=2) -> DataLoader:
    metadata = pd.read_pickle(metadata_file)
    image_metadata = pd.read_csv(image_metadata_file).set_index('image_id')
    image_item_pairs = abo_image_item_pairs(metadata)
    
    dataset_multimodal = ABODataset_multimodal(images_dir, metadata, image_metadata,
                                                   image_item_pairs, image_processor, text_processor)
    dataloader_multimodal = DataLoader(dataset=dataset_multimodal, batch_size=batch_size,
                                       num_workers=num_workers)
    
    return dataloader_multimodal

def abo_image_item_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
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
    # Note: modified from fine-tuning version
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
        # self.metadata = self.metadata.drop(columns=['item_weight', 'main_image_id',
        #                                             'other_image_id', 'country',
        #                                             'marketplace', 'domain_name'])    
        self.metadata = self.metadata[['item_name', 'brand', 'model_name', 'model_year',
                                       'product_description', 'product_type', 'color',
                                       'fabric_type', 'style', 'material', 'item_keywords',
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
        
        log_str = f'multimodal {idx}: image shape: {image.shape}; label length: {len(label)}'
        logger.info(log_str)
        
        return image, label, image_id, item_id
    
    def _row_to_str(self, row):
        row_filtered = row.dropna()
        # heading_data_pairs = list(zip(row_filtered.index, row_filtered))
        text = []
        for row_item in row_filtered:
            if isinstance(row_item, list):
                for list_item in row_item:
                    text.append(str(list_item) + ';')
            else:
                text.append(str(row_item) + ';')
        
        return ' '.join(text).replace('\n', ' ')
    
def build_abo_dataloader_text(metadata_file: str, text_processor: callable,
                              batch_size=64, num_workers=2) -> DataLoader:
    
    metadata = pd.read_pickle(metadata_file)
    dataset_text = ABODataset_text(metadata, text_processor)
    dataloader_text = DataLoader(dataset=dataset_text, batch_size=batch_size,
                                 num_workers=num_workers)
    return dataloader_text

class ABODataset_text(ABODataset_multimodal):
    def __init__(self, metadata: pd.DataFrame, text_processor: callable):
        self.metadata = metadata
        self.text_processor = text_processor
        
        self._drop_rows_with_images()
        self._reorg_metadata_columns()
        
    def _drop_rows_with_images(self):
        self.metadata = self.metadata[self.metadata['main_image_id'].isna() &
                                      self.metadata['other_image_id'].isna()]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        label = self._row_to_str(self.metadata.iloc[idx])
        label = self.text_processor(label)
        item_id = self.metadata.index[idx]
        
        log_str = f'text {idx}: label length: {len(label)}'
        logger.info(log_str)
        return label, item_id
        
def embed_multimodal(model, dataloader, device):
    embeddings_dict = {'image_id': [], 'item_id': [], 'embedding': []}
    
    model.eval()
    for data in tqdm(dataloader):
        images, labels, image_ids, item_ids = data
        embeddings_dict['image_id'].extend(image_ids)
        embeddings_dict['item_id'].extend(item_ids)
        
        images = images.to(device)
        samples = {"image": images, "text_input": labels}
        
        output = model.extract_features(samples, mode='multimodal')
        embeddings_dict['embedding'].extend(output.multimodal_embeds[:,0,:].detach().cpu().numpy())

    embeddings_df = pd.DataFrame(embeddings_dict)
    return embeddings_df

def embed_text(model, dataloader):
    embeddings_dict = {'item_id': [], 'embedding': []}
    
    model.eval()
    for data in tqdm(dataloader):
        labels, item_ids = data
        embeddings_dict['item_id'].extend(item_ids)
        
        samples = {"text_input": labels}
        
        output = model.extract_features(samples, mode='text')
        embeddings_dict['embedding'].extend(output.text_embeds[:,0,:].detach().cpu().numpy())

    embeddings_df = pd.DataFrame(embeddings_dict)
    return embeddings_df