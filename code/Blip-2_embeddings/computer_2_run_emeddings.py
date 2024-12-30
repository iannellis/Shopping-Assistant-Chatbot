# Uses the 3.11 environment locally

from embeddings import run_embeddings

import warnings
warnings.filterwarnings('ignore')

print('\n\nRunning GS embeddings\n\n')
run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset', model_type='gs', 
                   device='cuda', save_path='/mnt/d/embeddings', batch_size=64)

print('\n\nRunning ABO embeddings\n\n')
run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset', model_type='abo', 
                   device='cuda', save_path='/mnt/d/embeddings', batch_size=64)

print('\n\nRunning pretrained embeddings\n\n')
run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset', model_type='pretrain', 
                   device='cuda', save_path='/mnt/d/embeddings', batch_size=64)

print('\n\nRunning coco embeddings\n\n')
run_embeddings(abo_dataset_dir='/mnt/d/abo-dataset', model_type='coco', 
                   device='cuda', save_path='/mnt/d/embeddings', batch_size=32)