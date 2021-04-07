#!/bin/bash

# Install download package for google drive
pip install gdown

# Create data directory
mkdir data
cd data

# GeoPal data download
gdown -O geopalData.csv "https://drive.google.com/uc?export=download&id=1oTFgpPhkPMJzYfhHgAiTcEMYLqAg6TUe"
