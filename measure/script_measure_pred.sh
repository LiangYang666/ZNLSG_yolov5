#!/usr/bin/env bash

python step1_generate_aug_library_embedding.py 1
python step2_generate_detect_embedding.py 1
python step3_search_k_2pred.py 1
python step4_get_map.py