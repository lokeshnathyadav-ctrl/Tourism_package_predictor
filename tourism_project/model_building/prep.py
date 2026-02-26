
# Importing necessary libraries 
# For data manipulation
import pandas as pd
import sklearn

# Creates folder & perform OS related functionalities
import os

# To make API request
import requests

# Data preparation & splitting the data into train, test sets
from sklearn.model_selection import train_test_split

# Hugging Face space authentication then to upload files
from huggingface_hub import login,HfApi

# Defining constants for the dataset and output paths
                                
api = HfApi(token=os.getenv("HF_TOKE"))          # Initialize API client

# Defining the path of the uploaded data in Hugging Face Hub
DATASET_PATH = "hf://datasets/Lokeshnathy/Tourism-package-data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Dropping all the data points where contact made by the company itself
df = df[df['Contacted']!= 'Company Invited']

# Dropping insignificant columns
df = df.drop(['Unnamed: 0','CustomerID'],axis=1)

# Replacing 'Fe Male' with 'Female'
df['Gender'] = df['Gender'].replace('Fe Male','Female')

# Defining target variable
target = 'ProdTaken'

# To define the list of numerical, categorical & features to be scaled at task
numeric_features = [
    'CityTier',
    'Accom_Person',
    'PropertyRating',
    'Passport',
    'SatisfactionScore',
    'OwnCar',
    'Accom_Child']
categorical_features = [
    'Contacted',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation']
features_to_scale = [
    'Age',
    'Pitch_Duration',
    'Followups',
    'Avg_Trips',
    'MonthlyIncome']
    
# Splitting into X (features) and y (target)
X = df[numeric_features + categorical_features + features_to_scale]
y = df[target]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,y,
    test_size=0.2,        # 20% of the data reserved for testing
    random_state=42)      # this value will ensure reproducibility

Xtrain.to_csv("Xtrain.csv",index=False)   # Converts split Xtrain dataset to a csv file
Xtest.to_csv("Xtest.csv",index=False)     # Converts split Xtest dataset to a csv file
ytrain.to_csv("ytrain.csv",index=False)   # Converts split ytrain dataset to a csv file
ytest.to_csv("ytest.csv",index=False)     # Converts split ytest dataset to a csv file

# Creating a list of all the split data files 
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload the prepared data to Hugging Face Hub
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo = file_path.split("/")[-1], # just the filename
        repo_id="Lokeshnathy/Tourism-package-data",
        repo_type="dataset")
