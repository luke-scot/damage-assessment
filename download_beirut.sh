#!/bin/bash

# Install gdown if necessary
# pip install gdown

# Create data directory
mkdir data
cd data
mkdir Beirut
cd Beirut

# Download zip of Houston Data
gdown -O beirut-data.zip "https://drive.google.com/uc?export=download&id=1z5lRknsjqyvE3IWiirWxSO-Qj-g9lE9Q"

# Unzip the data file
unzip beirut-data.zip

# Remove zip file
rm beirut-data.zip
cd ../