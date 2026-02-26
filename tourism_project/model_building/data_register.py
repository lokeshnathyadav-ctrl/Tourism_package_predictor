
# Necessary libraries
# For Hugging Face authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# For OS related functionalities 
import os
# To make API requests
import requests

# Defining user credintials for accessing Hugging Face Hub
repo_id = "Lokeshnathy/Tourism_package_data"              # repo id
repo_type = "dataset"                                     # repo Type = 'Dataset'

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKE"))

# Checking if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
    print(f"Space '{repo_id}' created.")

# Uploads the data to Hugging Face Hub 
api.upload_folder(
    folder_path="D:/tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type)
