import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure you have set up ~/.kaggle/kaggle.json
os.makedirs("files/Images", exist_ok=True)

api = KaggleApi()
api.authenticate()

# Download dataset from Kaggle
api.dataset_download_files("adityajn105/flickr8k", path="files", unzip=True)

print("Flickr8k dataset downloaded and extracted to files")
