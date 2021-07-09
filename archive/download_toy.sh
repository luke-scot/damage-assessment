#!/bin/bash

# Install download package for google drive
pip install gdown

# Create toy directory
mkdir toy
cd toy

# Download priors csv
gdown -O dpriors.csv "https://drive.google.com/uc?export=download&id=1dH5Ski00_b07EZhDiXAY0MrHL3nF6FAw"
# Download edges csv
gdown -O edges.csv "https://drive.google.com/uc?export=download&id=1_XhXy89N5-76pNZJGMrnTe0DPxN0dq2U"
# Download toy .png
gdown -O toy.png "https://drive.google.com/uc?export=download&id=1MF1q8RaEwxarWwMsXrow__87wPtdY90v"