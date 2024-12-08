import torch
from fine_tune_pretrain import run_pretrain

device = 'cuda' if torch.cuda.is_available() else 'cpu'

run_pretrain(marqo_gs_data_dir = '/mnt/d/marqo-gs-10m', device='cuda', save_dir='../../saves')