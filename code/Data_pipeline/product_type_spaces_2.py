import pandas as pd
import tomllib

import warnings
warnings.filterwarnings('ignore')

#-------------------------Work-performing Functions-------------------------------------
def fix_product_type(pdf):
    """The product_type often has underscores or no spaces at all between words. This
    function fixes that.
    
    Note: it will not fix unknown product_types without spaces between words."""
    
    pdf['product_type'] = pdf['product_type'].str.replace('_', ' ')
    pdf.loc[pdf['product_type'] == 'FINERING', 'product_type'] = 'FINE RING'
    pdf.loc[pdf['product_type'] == 'FINENECKLACEBRACELETANKLET', 'product_type'] = 'FINE NECKLACE BRACELET ANKLET'
    pdf.loc[pdf['product_type'] == 'FINEEARRING', 'product_type'] = 'FINE EARRING'
    pdf.loc[pdf['product_type'] == 'FASHIONNECKLACEBRACELETANKLET', 'product_type'] = 'FASHION NECKLACE BRACELET ANKLET'
    pdf.loc[pdf['product_type'] == 'FINEOTHER', 'product_type'] = 'FINE OTHER'
    pdf.loc[pdf['product_type'] == 'FASHIONEARRING', 'product_type'] = 'FASHION EARRING'
    pdf.loc[pdf['product_type'] == 'SHOWERHEAD', 'product_type'] = 'SHOWER HEAD'
    pdf.loc[pdf['product_type'] == 'FASHIONOTHER', 'product_type'] = 'FASHION OTHER'
    pdf['product_type'] = pdf['product_type'].str.replace('ABIS ', '')
    return pdf

#-------------------------------Operate on metadata-------------------------------------
if __name__ == "__main__":
    print('2. Replace "_" with " " and add spaces where necessary in product_type in metadata.')
    
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    working_dir = config['global']['working_dir']
    meta_save_prefix = config['global']['meta_save_prefix']

    print('Loading English-tagged metadata...')
    pdf = pd.read_pickle(working_dir + '/' + meta_save_prefix + "preprocess-1.pkl")

    print('Separating product_type words...')
    pdf = fix_product_type(pdf)

    print('Saving metadata with fixed product types...')
    pdf.to_pickle(working_dir + '/' + meta_save_prefix + "preprocess-2.pkl")