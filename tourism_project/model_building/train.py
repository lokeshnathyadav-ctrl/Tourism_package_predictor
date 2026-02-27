
# Necessary Libraries for
# Data manipulation
import pandas as pd
# Data preparation & splitting data into train-test-splits
from sklearn.model_selection import train_test_split
# Performs scaling of numerical features and one-hot encoding for categoricals
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Transforms the preprocessed columns
from sklearn.compose import make_column_transformer
# Creates ML pipeline
from sklearn.pipeline import make_pipeline
# Model training, tuning and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import(
    confusion_matrix, 
    classification_report, 
    recall_score, 
    precision_score, 
    accuracy_score, 
    f1_score)
# Model serialization
import joblib
# Creating a folder
import os
# To make API requests
import requests
# Hugging face authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# importing mlflow for experimentation & tracking
import mlflow

# Setting the tracking URL for MLflow & defining name of the experiment
mlflow.set_tracking_uri("https://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")
                                    
api = HfApi(token=os.getenv("HF_TOKE"))        # Initialize API client

# define the path to access the splitted datasets
Xtrain_path = "hf://datasets/Lokeshnathy/Tourism-package-data/Xtrain.csv"
Xtest_path = "hf://datasets/Lokeshnathy/Tourism-package-data/Xtest.csv"
ytrain_path = "hf://datasets/Lokeshnathy/Tourism-package-data/ytrain.csv"
ytest_path = "hf://datasets/Lokeshnathy/Tourism-package-data/ytest.csv"

# Reads the split data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

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

# Defining Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(),features_to_scale),
    (OneHotEncoder(handle_unknown='ignore',drop='first'),categorical_features))

# Defining base Gradient Boosting estimator
gb_model = GradientBoostingRegressor(random_state=42)

# Defining Hyperparameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators':[50,75,100],
    'gradientboostingclassifier__min_samples_leaf':[2,5,8],
    'gradientboostingclassifier__max_features':[0.7,0.8,0.9,1],
    'gradientboostingclassifier__max_depth':[3,5,7,10],
    'gradientboostingclassifier__learning_rate':[0.01,0.05,0.1],
    'gradientboostingclassifier__min_samples_split':[5,10],
    'gradientboostingclassifier__min_impurity_decrease':[0.001,0.002,0.003]}

# Create pipeline
model_pipeline = make_pipeline(preprocessor,gb_model)

# log the experiments 
with mlflow.start_run():
    # Hyperparameter tuning
    random_search = RandomizedSearchCV(model_pipeline,
                                       param_grid,
                                       scoring='recall',
                                       cv=5,
                                       n_jobs=-1)
    random_search.fit(Xtrain,ytrain)   # fit the parameter grid to training data

    # Log all parameter combinations and their mean test scores
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a seperate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score",mean_score)
            mlflow.log_metric("std_test_score",std_score)

    # Log best parameters seperately in main run
    mlflow.log_params(random_search.best_params_)

    # Store and evaluate the best model
    best_model = random_search.best_estimator_
    
    # Defining classification threshold to predict the classification
    classification_threshold = 0.45
    
    # Best Model's predictions on training data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:,1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    # Best Model's predictions on testing data
    y_pred_test_proba = best_model.predict_proba(Xtest)[:,1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)
    # Fetching classification reports for training and test datasets
    train_report = classification_report(ytrain,y_pred_train,output_dict=True)
    test_report = classification_report(ytest,y_pred_test,output_dict=True)
    # Logging the classification scores
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall":train_report['1']['recall'],
        "train_f1-score":train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall":test_report['1']['recall'],
        "test_f1-score":test_report['1']['f1-score']})
    
    # save the model locally
    model_path = "best_tour_purchase_pred_v1.joblib"
    joblib.dump(best_model,model_path)
    
    # Log the model artifact
    mlflow.log_artifact(model_path,artifact_path ="model")
    print(f"Model saved as artifact at: {model_path}")
    
    # Upload to Hugging face
    repo_id = "Lokeshnathy/Best-tour-pur-pred-model"     # repo ID
    repo_type = "model"                                  # repo type : "Models"

    # To Check if space exists
    try:
        api.repo_info(repo_id=repo_id,repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
        print(f"Space '{repo_id}' created.")

    # Uploading serialized model to HF Hub
    api.upload_file(
        path_or_fileobj="best_tour_purchase_pred_v1.joblib",
        path_in_repo="best_tour_purchase_pred_v1.joblib",
        repo_id = repo_id,
        repo_type=repo_type)
