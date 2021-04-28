#!/bin/bash

# Install download package for google drive
pip install gdown

# Create data directory
mkdir data
cd data

# Download zip of Houston Data
gdown -O houstonData.zip "https://drive.google.com/uc?export=download&id=1VzCDeqaqZPBBWJyvSLuagxTv18mdMbGU"

# Unzip the data file
unzip houstonData.zip