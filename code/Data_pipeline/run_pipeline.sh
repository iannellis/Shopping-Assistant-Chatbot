#!/bin/bash

python311=/home/ellis/.cache/pypoetry/virtualenvs/shoptalk-py3-11-gySnKGNK-py3.11/bin/python
python312=/home/ellis/.cache/pypoetry/virtualenvs/shoptalk-py3-12-PgppYsjg-py3.12/bin/python

$python312 1_english_tags.py
$python312 2_product_type_spaces.py
$python312 3_english_check.py
$python312 4_product_type_verification.py
$python311 5_embed_to_chroma.py