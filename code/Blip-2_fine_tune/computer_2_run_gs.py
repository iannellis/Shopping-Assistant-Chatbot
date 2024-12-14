# Uses the 3.11 environment locally

from fine_tune_pretrain_gs import run_fine_tune

run_fine_tune(marqo_gs_data_dir='/mnt/d/marqo-gs-10m', device='cuda',
             save_dir='/mnt/d/marqo-gs-10m/model-saves')