#!/bin/bash

# Install download package for google drive
pip install gdown

# Create data directory
mkdir netconf
cd netconf
gdown -O edges.csv "https://drive.google.com/uc?export=download&id=1_XhXy89N5-76pNZJGMrnTe0DPxN0dq2U"

gdown -O priors.csv "https://drive.google.com/uc?export=download&id=1dH5Ski00_b07EZhDiXAY0MrHL3nF6FAw"