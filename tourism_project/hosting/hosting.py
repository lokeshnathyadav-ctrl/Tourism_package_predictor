
# Importing necessary libraries
from huggingface_hub import HfApi
import os

# Define user credentials for HF Hub
api=HfApi(token=os.getenv("HF_TOKE")
repo_id = "Lokeshnathy/TOURIST-PURCHASE-PREDICTION-SPACE"
repo_type="space"

# Uploads the required deployment files to HF Hub
api.upload_folder(
    folder_path = "D:/tourism_project/deployment", 
    repo_id = repo_id,
    repo_type = repo_type)
