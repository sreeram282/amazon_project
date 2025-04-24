#!/bin/bash

# Install Python packages
pip install -r requirements.txt

# Create local nltk_data dir and download punkt
mkdir -p nltk_data
python -m nltk.downloader -d nltk_data punkt
