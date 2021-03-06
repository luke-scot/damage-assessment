#!/bin/bash

# Install gdown if necessary
# pip install gdown

# Create data directory
mkdir data
cd data
mkdir Houston
cd Houston

# Download zip of Houston Data
gdown -O houston-data.zip "https://drive.google.com/uc?export=download&id=1hserIYq6OBiHqpZ9Gjm4xywz16Ew57ij"

# Unzip the data file
unzip houston-data.zip

# Remove zip file
rm houston-data.zip
cd ../