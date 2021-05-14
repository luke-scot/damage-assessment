#!/bin/bash

# Install download package for google drive
pip install gdown

# Create data directory
mkdir data
cd data

# OSM Beirut data download
gdown -O beirutBuildingFootprints.geojson "https://drive.google.com/uc?export=download&id=1APHknp072orpaP8YFaLOP4VQisbuRcys"

# GeoPal data download
gdown -O geopalData.csv "https://drive.google.com/uc?export=download&id=1oTFgpPhkPMJzYfhHgAiTcEMYLqAg6TUe"

# Beirut Interferometry data
# Pre Explosion
gdown -O beirutPrePreExplosionIfg.tif "https://drive.google.com/uc?export=download&id=1diYZtQJrQLZ1s8WMSvEGnY4wx-UT6vC7"

# Post Explosion
gdown -O beirutPrePostExplosionIfg.tif "https://drive.google.com/uc?export=download&id=1S_WYEJ6ikL6wskvoO7G3xGh-Lwb7EqlF"

# Bourj Hammoud Data
gdown -O bourjHammoud.csv "https://drive.google.com/uc?export=download&id=1t_xpbzcO2svWJnO8FLd0yS3bvNFqu0Jo"

# Old survey forms
gdown -O SurveyForms.csv "https://drive.google.com/uc?export=download&id=1z1hEYVkv2FWga3jCyaxoZ9iyMPWOJFuA"

# Manually Mapped Data
gdown -O manualDamageClasses.csv "https://drive.google.com/uc?export=download&id=11IYcRwrYVJbOWWP7JGDS4EKyTaggSWx5"