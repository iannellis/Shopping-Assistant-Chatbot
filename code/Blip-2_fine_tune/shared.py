"""Shared functions independent of the dataset for fine-tuning BLIP-2.

Uses the 3.11 environment locally"""

import torch

from tqdm import tqdm
import pickle
from pathlib import Path

def train(model, dataloader, device, optimizer, epochs, save_dir):
    """The main training loop. Checkpoints the model in save_dir and returns a list of 
    the average loss calculated every 1000 iterations. 
    
    Note that autocast is required, as that is how the model was pre-trained in LAVIS.
    That also means using the gradient scaler to mitiage exploding gradients.
    """
    losses = []
    running_loss = 0
    
    scaler = torch.GradScaler()
    model.train()
    for epoch in range(epochs): 
        for i, data in enumerate(tqdm(dataloader)):
            images, labels = data
            images = images.to(device)
            samples = {"image": images, "text_input": labels}
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                output = model(samples)
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += output.loss.item()
            
            iteration = epoch * len(dataloader) + i + 1
            if (iteration+1) % 1000 == 0:
                last_loss = running_loss / 1000
                losses.append(last_loss)
                print(f'  iteration {iteration} loss: {last_loss}')
                running_loss = 0
        
            if (iteration+1) % 5000 == 0:
                save_model_and_loss(model, losses, save_dir, file_prefix=f'pretrain+{iteration+1}')

    return losses

@torch.no_grad()
def validate(model, dataloader, device, save_dir):
    """The validation loop. Only returns the validation loss calculated every 1000
    iterations, but saved less frequently.
    
    Note the use of autocast, which probably isn't necessary as this is an evaulation
    run.
    """
    losses = []
    running_loss = 0
    
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        images, labels = data
        images = images.to(device)
        samples = {"image": images, "text_input": labels}
        with torch.autocast(device_type=device):
            output = model(samples)
        running_loss += output.loss.item()
        if (i+1) % 1000 == 0:
            last_loss = running_loss / 1000
            losses.append(last_loss)
            print(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0
        
        if (i+1) % 5000 == 0:
            save_validate_loss(losses, save_dir, file_prefix=f'validate_{i+1}')
            
    return losses

def save_model_and_loss(model, loss, save_dir, file_prefix):
    """For use in the training loop and after, when training is complete. Saves the 
    model and loss to the specified save_dir using the spcified file_prefix."""
    Path(save_dir).mkdir(exist_ok=True)
    torch.save(model, save_dir + '/' + file_prefix +'.pt')
    with open(save_dir + '/' + file_prefix + '_loss', 'wb') as f:            
        pickle.dump(loss, f)
        
def save_validate_loss(loss, save_dir, file_prefix):
    """For use in the validation loop and after, when validation is complete. Simply
    saves the given loss to the save_dir using the specified file_prefix."""
    with open(save_dir + '/' + file_prefix + '_loss', 'wb') as f:            
        pickle.dump(loss, f)