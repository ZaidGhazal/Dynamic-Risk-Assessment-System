from flask import Flask, session, jsonify, request
import pandas as pd
from pickle import dump
import os
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def import_data() -> pd.DataFrame:
    """ Import data from path in config file

    Returns
    -------
     Data imported
    """
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    return data


# Function for training the model
def train_model():
    """Train and save ML model as trainedmodel.pkl """
    # Import data
    data = import_data()
    X = data.drop(columns=["corporation", "exited"])
    y = data["exited"]

    # use logistic regression for training
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to data
    lr.fit(X, y)

    # write the trained model to workspace in a file called trainedmodel.pkl
    path = os.path.join(model_path, "trainedmodel.pkl")
    dump(lr, open(path, 'wb'))


if __name__ == '__main__':
    train_model()
