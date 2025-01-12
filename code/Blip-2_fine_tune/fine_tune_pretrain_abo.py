"""Fine-tunes a BLIP-2 model using a dataset consisting of image-text pairs.

Uses the 3.11 environment locally.

Note that this must be run in Linux, as LAVIS requres nccl distributed processing,
even if only running on one GPU.
"""

from shared import train, validate, save_model_and_loss, save_validate_loss

from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from PIL import Image
import pandas as pd

import os
import random

def run_fine_tune(abo_dataset_dir='/mnt/d/abo-dataset', device='cuda',
                 save_dir='/mnt/d/abo-dataset/model_saves', batch_size=24, epochs=2):
    """The orchechestrator function for the fine-tuning. It loads the model and
    processors, calls the function to build the dataloader, sets up the opimizer,
    sets up distributed processing group, calls the training function, saves the final
    model and loss, and destroys the distributed processing group."""    
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=False, device=device)
    images_dir = abo_dataset_dir + '/images/small'
    metadata_file = abo_dataset_dir + '/abo-listings-english-fixed-product-type.pkl'
    image_metadata_file = abo_dataset_dir + '/images/metadata/images.csv'

    train_dataloader = build_abo_dataloader(images_dir, metadata_file, image_metadata_file,
                                        image_processor=vis_processors['train'],
                                        text_processor=txt_processors['train'], seed=42,
                                        batch_size=batch_size, num_workers=2)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    
    losses_train = train(model=model, dataloader=train_dataloader, device=device,
                         optimizer=optimizer, epochs=epochs, save_dir=save_dir)
    save_model_and_loss(model, losses_train, save_dir, file_prefix='pretrain_2epochs')
    torch.distributed.destroy_process_group()
    return

def build_abo_dataloader(images_dir: str, metadata_file: str, image_metadata_file: str, 
                         image_processor: callable, text_processor: callable, seed: int=42, 
                         batch_size: int=64, num_workers: int=2) -> DataLoader:
    """Builds the dataloader for model training. Loads metadata, creates image-item
    pairs, instantiates the dataset class, gets the weigths for the sampler (as some 
    product types are represented more than others), sets up the sampler, and
    instantiates the dataloader.

    Args:
        images_dir (str): The directory where the images are located
        metadata_file (str): The file with the processed metadata
        image_metadata_file (str): The CSV file mapping image_ids to image files
        image_processor (callable): The processor BLIP-2 requires us to run PIL images
                                    through before passing them through the model
        text_processor (callable): The processor BLIP-2 requires us to run text through
                                    before passing it to the model
        seed (int, optional): The seed for random number generators
        batch_size (int, optional): How many image-text pairs to pass through the model at once
        num_workers (int, optional): How may workers to use to load image-text pairs

    Returns:
        DataLoader: The PyTorch dataloader used in the training loop
    """
    metadata = pd.read_pickle(metadata_file)
    image_metadata = pd.read_csv(image_metadata_file).set_index('image_id')
    image_item_pairs = abo_image_item_pairs(metadata)
    dataset = ABODataset(images_dir, metadata, image_metadata, image_item_pairs,
                         image_processor, text_processor, seed)
    weights = get_sample_weights_abo(metadata, image_item_pairs)
    generator = torch.Generator().manual_seed(seed)
    # Recommend using 2*len(weights) in future runs rather than 2 epochs
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader

def abo_image_item_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    """Buids a Pandas dataframe of image_id-item_id pairs for use by the dataset class."""
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
    """A typical PyTorch dataloader for the ABO dataset."""
    def __init__(self, image_dir: str, metadata: pd.DataFrame,
                 image_metadata: pd.DataFrame, image_item_pairs: pd.DataFrame,
                 image_processor: callable, text_processor: callable, seed: int):
        """
        Args:
        images_dir (str): The directory where the images are located
        metadata (pd.DataFrame): The processed metadata
        image_metadata (pd.DataFrame): Maps image_ids to image files
        image_item_pairs (pd.DataFrame): The image_id-item_id pairs corresponding to
                                         the images and metadata items to feed to the model
        image_processor (callable): The processor BLIP-2 requires us to run PIL images
                                    through before passing them through the model
        text_processor (callable): The processor BLIP-2 requires us to run text through
                                    before passing it to the model
        seed (int): The seed for random number generators
        """
        self.image_dir = image_dir
        self.metadata = metadata
        self.image_metadata = image_metadata
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.image_item_pairs = image_item_pairs
        
        self._drop_unwanted_metadata_columns()
        random.seed(seed)
        
    def _drop_unwanted_metadata_columns(self):
        """Drop metadata columns that are not useful for training."""
        self.metadata = self.metadata.drop(columns=['item_weight', 'main_image_id',
                                                    'other_image_id', 'country',
                                                    'marketplace', 'domain_name'])    
     
    def __len__(self):
        return len(self.image_item_pairs)
    
    def __getitem__(self, idx: int):
        """Gets the image_id and item_id, loads the respective image and item text,
        sends them through the processors, and returns the results."""
        image_id = self.image_item_pairs.loc[idx, 'image_id']
        item_id = self.image_item_pairs.loc[idx, 'item_id']
        image_path = os.path.join(self.image_dir, self.image_metadata.loc[image_id, 'path'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(image)
        label = self._row_to_str(self.metadata.loc[item_id])
        label = self.text_processor(label)
        return image, label
    
    def _row_to_str(self, row):
        """Convert a row of metadata to a text string consumable by the model."""
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
        
        return ' '.join(text).replace('\n', ' ').replace('^', ' ').replace(',', ', ')
    
def get_sample_weights_abo(metadata: pd.DataFrame, image_item_pairs: pd.DataFrame) -> list:
    """The sample weights for use by the WeightedRandomSampler. An image-item pair's
    weight is the inverse of the number of image-item pairs of that product_type."""
    joined = image_item_pairs.join(metadata, on='item_id')
    product_type_counts = joined['product_type'].value_counts()
    product_type_counts_full = product_type_counts[joined['product_type']].to_numpy()
    weights = 1 / product_type_counts_full
    return weights