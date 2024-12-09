from check_image_category import (load_listings, run_dataset_check, resume_dataset_check,
                                  run_missing_items)
import pickle

pdf = load_listings()
pdf_has_images = pdf[~pdf['main_image_id'].isna() | ~pdf['other_image_id'].isna()]
n_products = len(pdf_has_images)

# run_dataset_check(pdf_has_images, n_products//2, n_products//2 + 45000, 'D:/images/small', 'abo-category-check-comp-3-run2.pkl')
# resume_dataset_check(pdf_has_images, n_products//2, n_products//2 + 45054, 'D:/images/small', 'abo-category-check-comp-3-run2.pkl')

with open('code/Llama_data_checks/missing_items.pkl', 'rb') as f:
    missing_items = pickle.load(f)
        
run_missing_items(pdf_has_images, missing_items, 'D:/images/small', 'abo-category-check-missing-items.pkl')