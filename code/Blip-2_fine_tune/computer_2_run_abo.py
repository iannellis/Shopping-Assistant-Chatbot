# Uses the 3.11 environment locally

from fine_tune_pretrain_abo import run_fine_tune

run_fine_tune(abo_dataset_dir='/mnt/d/abo-dataset', device='cuda',
                 save_dir='/mnt/d/abo-dataset/model_saves')