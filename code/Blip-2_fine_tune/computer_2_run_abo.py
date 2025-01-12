"""Just calls the main function in fine_tune_pretrain_abo.py

Uses the 3.11 environment locally
"""

from fine_tune_pretrain_abo import run_fine_tune

run_fine_tune(abo_dataset_dir='/mnt/d/abo-dataset', device='cuda',
                 save_dir='/mnt/d/abo-dataset/model_saves', batch_size=24, epochs=2)