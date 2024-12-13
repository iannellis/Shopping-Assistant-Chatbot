# Uses the 3.11 environment locally

from shared import train, validate, save_model_and_loss, save_validate_loss

from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from PIL import Image
import pandas as pd

import os
import random

def run_fine_tune(abo_dataset_dir='/mnt/d/abo-dataset', device='cuda',
                 save_dir='/mnt/d/abo-dataset/model_saves'):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=False, device=device)
    images_dir = abo_dataset_dir + '/images/small'
    metadata_file = abo_dataset_dir + '/abo-listings-english-fixed-product-type.pkl'
    image_metadata_file = abo_dataset_dir + '/images/metadata/images.csv'

    train_dataloader = build_abo_dataloader(images_dir, metadata_file, image_metadata_file,
                                        image_processor=vis_processors['train'],
                                        text_processor=txt_processors['train'], seed=42,
                                        batch_size=24, num_workers=2)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    
    losses_train = train(model=model, dataloader=train_dataloader, device=device,
                         optimizer=optimizer, epochs=2, save_dir=save_dir)
    save_model_and_loss(model, losses_train, save_dir, 'pretrain_2epochs')
    torch.distributed.destroy_process_group()
    return

def build_abo_dataloader(images_dir: str, metadata_file: str, image_metadata_file: str, 
                         image_processor: callable, text_processor: callable, seed=42, 
                         batch_size=64, num_workers=2) -> DataLoader:
    metadata = pd.read_pickle(metadata_file)
    image_metadata = pd.read_csv(image_metadata_file).set_index('image_id')
    image_item_pairs = abo_image_item_pairs(metadata)
    dataset = ABODataset(images_dir, metadata, image_metadata, image_item_pairs,
                         image_processor, text_processor)
    weights = get_sample_weights_abo(metadata, image_item_pairs)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader

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

class ABODataset(Dataset):
    def __init__(self, image_dir: str, metadata: pd.DataFrame,
                 image_metadata: pd.DataFrame, image_item_pairs: pd.DataFrame,
                 image_processor: callable, text_processor: callable):
        self.image_dir = image_dir
        self.metadata = metadata
        self.image_metadata = image_metadata
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.image_item_pairs = image_item_pairs
        
        self._drop_unwanted_metadata_columns()
        random.seed(42)
        
    def _drop_unwanted_metadata_columns(self):
        self.metadata = self.metadata.drop(columns=['item_weight', 'main_image_id',
                                                    'other_image_id', 'country',
                                                    'marketplace', 'domain_name'])    
     
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
        return image, label
    
    def _row_to_str(self, row):
        row_filtered = row.dropna()
        heading_data_pairs = list(zip(row_filtered.index, row_filtered))
        random.shuffle(heading_data_pairs)  # so that the model doesn't get used to an order
        text = []
        for heading, item in heading_data_pairs:
            text.append(heading.replace('_', ' ') + ':')
            if isinstance(item, list):
                text.extend(item)
            else:
                text.append(str(item))
        
        return ' '.join(text).replace('\n', ' ')
    
def get_sample_weights_abo(metadata: pd.DataFrame, image_item_pairs: pd.DataFrame) -> list:
    joined = image_item_pairs.join(metadata, on='item_id')
    product_type_counts = joined['product_type'].value_counts()
    product_type_counts_full = product_type_counts[joined['product_type']].to_numpy()
    weights = 1 / product_type_counts_full
    return weights