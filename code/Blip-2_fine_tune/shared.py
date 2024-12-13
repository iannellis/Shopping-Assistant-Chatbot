import torch

from tqdm import tqdm
import pickle
from pathlib import Path

def train(model, dataloader, device, optimizer, epochs, save_dir):
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
            if iteration % 1000 == 999:
                last_loss = running_loss / 1000
                losses.append(last_loss)
                print(f'  iteration {iteration} loss: {last_loss}')
                running_loss = 0
        
            if iteration % 5000 == 4999:
                save_model_and_loss(model, losses, save_dir, f'pretrain+{iteration}')

    return losses

@torch.no_grad()
def validate(model, dataloader, device, save_dir):
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
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            losses.append(last_loss)
            print(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0
        
        if i % 5000 == 4999:
            save_validate_loss(losses, save_dir, f'validate_{i+1}')
            
    return losses

def save_model_and_loss(model, loss, save_dir, file_prefix):
    Path(save_dir).mkdir(exist_ok=True)
    torch.save(model, save_dir + '/' + file_prefix +'.pt')
    with open(save_dir + '/' + file_prefix + '_loss', 'wb') as f:            
        pickle.dump(loss, f)
        
def save_validate_loss(loss, save_dir, file_prefix):
    with open(save_dir + '/' + file_prefix + '_loss', 'wb') as f:            
        pickle.dump(loss, f)