#!/usr/bin/env bash

python step1_generate_library_embedding.py
python step2_generate_detect_embedding.py
python step3_mesure_dist2pred.py