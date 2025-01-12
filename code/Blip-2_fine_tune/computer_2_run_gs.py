"""Just calls the main function in fine_tune_pretrain_gs.py

Uses the 3.11 environment locally
"""

from fine_tune_pretrain_gs import run_fine_tune

run_fine_tune(marqo_gs_data_dir='/mnt/d/marqo-gs-10m', device='cuda',
             save_dir='/mnt/d/marqo-gs-10m/model-saves', batch_size=24, epochs=1)