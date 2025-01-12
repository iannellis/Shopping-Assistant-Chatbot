# Uses the 3.11 environment locally

from shared import train, validate, save_model_and_loss, save_validate_loss

from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from PIL import Image
import pandas as pd

import os
import random

def run_fine_tune(marqo_gs_data_dir='/mnt/d/marqo-gs-10m', device='cuda',
                 save_dir='/mnt/d/marqo-gs-10m/model-saves', batch_size=24, epochs=1):
    """The orchechestrator function for the fine-tuning. It loads the model and
    processors, calls the function to build the dataloader, sets up the opimizer,
    sets up distributed processing group, calls the training and validation functions,
    saves the final model, loss, and validation loss, and destroys the distributed
    processing group."""    
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=False, device=device)
    images_dir = marqo_gs_data_dir + '/images'
    train_annotations = marqo_gs_data_dir + '/marqo-gs-dataset/marqo_gs_full_10m/query_0_product_id_0.csv'
    val_annotations = marqo_gs_data_dir + '/marqo-gs-dataset/marqo_gs_full_10m/query_1_product_id_1.csv'

    train_dataloader = build_dataloader(images_dir=images_dir, annotations_file=train_annotations,
                                        image_processor=vis_processors['train'],
                                        text_processor=txt_processors['train'], seed=42,
                                        batch_size=batch_size, num_workers=2)
    val_dataloader = build_dataloader(images_dir=images_dir, annotations_file=val_annotations,
                                        image_processor=vis_processors['eval'],
                                        text_processor=txt_processors['eval'], seed=42,
                                        batch_size=batch_size, num_workers=2)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    losses_train = train(model=model, dataloader=train_dataloader, device=device,
                                   optimizer=optimizer, epochs=epochs, save_dir=save_dir)
    save_model_and_loss(model, losses_train, save_dir, 'pretrain_1epoch')
    losses_validate = validate(model=model, dataloader=val_dataloader, device=device, save_dir=save_dir)
    save_validate_loss(losses_validate, save_dir, 'validate_final')
    torch.distributed.destroy_process_group()
    return
    
def build_dataloader(images_dir: str, annotations_file: str, image_processor: callable, 
                     text_processor: callable, seed=42, batch_size=64, num_workers=2) -> DataLoader:
    """Builds the dataloader for model training. Loads metadata, creates image-item
    pairs, instantiates the dataset class, gets the weigths for the sampler (as some 
    product types are represented more than others), sets up the sampler, and
    instantiates the dataloader.

    Args:
        images_dir (str): The directory where the images are located
        annotations_file (str): The file with all the annotations for the dataset
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
    dataset = GoogleShoppingDataset(image_dir=images_dir, annotations_file=annotations_file,
                                      image_processor=image_processor, text_processor=text_processor,
                                      seed=seed)
    weights = get_sample_weights(annotations_file)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader

class GoogleShoppingDataset(Dataset):
    """A typical PyTorch dataloader for the Google Shopping dataset."""
    def __init__(self, image_dir: str, annotations_file: str, image_processor: object,
                 text_processor: object, seed: int):
        """
        Args:
        images_dir (str): The directory where the images are located
        annotations_file (str): The file with all the annotations for the dataset
        image_processor (callable): The processor BLIP-2 requires us to run PIL images
                                    through before passing them through the model
        text_processor (callable): The processor BLIP-2 requires us to run text through
                                    before passing it to the model
        seed (int): The seed for random number generators
        """
        self.annotations = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.text_processor = text_processor
        random.seed(seed)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx: int):
        """Loads the image and annotation, randomly choosing the order of query and
        title in the annotation, sends them through the processors, and returns the
        result."""
        image_path = os.path.join(self.image_dir, self.annotations.loc[idx, 'image_local'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(image)
        label_options = (self.annotations.loc[idx, 'query'] + ': ' + self.annotations.loc[idx, 'title'], 
                         self.annotations.loc[idx, 'title'] + ': ' + self.annotations.loc[idx, 'query'])
        label = random.choice(label_options)
        label = self.text_processor(label)
        return image, label

def get_sample_weights(annotations_file: str):
    """The sample weights for use by the WeightedRandomSampler. An item's weight is
    1/(<# items in a query> * <# queries an item appears in>)."""
    annotations_df = pd.read_csv(annotations_file)
    query_counts = annotations_df['query_id'].value_counts()
    query_counts_full = query_counts[annotations_df['query_id']].to_numpy()
    product_counts = annotations_df['product_id'].value_counts()
    product_counts_full = product_counts[annotations_df['product_id']].to_numpy()
    weights = 1 / ( query_counts_full * product_counts_full)
    return weights
