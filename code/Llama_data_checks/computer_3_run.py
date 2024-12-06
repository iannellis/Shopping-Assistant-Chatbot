from check_image_category import load_listings, run_dataset_check, resume_dataset_check

pdf = load_listings()
pdf_has_images = pdf[~pdf['main_image_id'].isna() | ~pdf['other_image_id'].isna()]
n_products = len(pdf_has_images)

# run_dataset_check(pdf_has_images, n_products//2, n_products, 'D:/images/small', 'abo-category-check-comp-3.pkl')
resume_dataset_check(pdf_has_images, n_products//2, n_products, 'D:/images/small', 'abo-category-check-comp-3.pkl')