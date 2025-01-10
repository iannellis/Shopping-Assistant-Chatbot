#!/bin/bash

python311=/home/ellis/.cache/pypoetry/virtualenvs/shoptalk-py3-11-gySnKGNK-py3.11/bin/python
python312=/home/ellis/.cache/pypoetry/virtualenvs/shoptalk-py3-12-PgppYsjg-py3.12/bin/python

$python312 code/Data_pipeline/english_tags_1.py
$python312 code/Data_pipeline/product_type_spaces_2.py
$python312 code/Data_pipeline/english_check_3.py
$python312 code/Data_pipeline/product_type_verification_4.py
$python311 code/Data_pipeline/embed_to_chroma_5.py