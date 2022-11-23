from time import time
import subprocess
import os
import json
from pickle import load
import numpy as np
import pandas as pd
from typing import List
import sys

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

deployment_dir = os.path.join(config['prod_deployment_path'])
dataset_csv_dir = os.path.join(config['output_folder_path'])


# get model predictions
def model_predictions(data: pd.DataFrame) -> np.ndarray:
    """
    Read the deployed model and a test dataset, calculate predictions
    """

    X = data.drop(columns=["corporation", "exited"])

    model_path = os.path.join(deployment_dir, "trainedmodel.pkl")

    model = load(open(model_path, "rb"))
    preds = model.predict(X).tolist()

    # return value should be a list containing all predictions
    return preds


# Function to get summary statistics
def dataframe_summary() -> List:
    """Calculate summary statistics for each column in the ingested dataset"""

    data = pd.read_csv(os.path.join(dataset_csv_dir, "finaldata.csv"))
    data = data.select_dtypes(include=np.number)
    stats_list = []
    for col in data.columns:
        mean = data[col].mean()
        median = data[col].median()
        std = data[col].std()
        stats_list.append((mean, median, std))

    # return value should be a list containing all summary statistics
    return stats_list


def missing_data_percent() -> List:
    """Check missing data percent for each column in list"""
    data = pd.read_csv(os.path.join(dataset_csv_dir, "finaldata.csv"))
    data = data.select_dtypes(include=np.number)
    percents_list = (data.isnull().mean() * 100).values.tolist()
    return percents_list


# Function to get timings
def execution_time() -> List:
    """
    Calculate timing of training.py and ingestion.py
    """
    timing_list = []

    # Time for ingestion execution
    start = time()
    subprocess.call(["python", "-m", "ingestion"])
    end = time()
    timing_list.append(np.round(end - start, 2))

    # Time for training execution
    start = time()
    subprocess.call(["python", "-m", "training"])
    end = time()
    timing_list.append(np.round(end - start, 2))

    # return a list of 2 timing values in seconds: ingestion & training
    return timing_list


def get_latest_version_pkg(name: str) -> str:
    """Get the latest version of the given package"""
    latest_version = str(
        subprocess.run(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '{}==random'.format(name)],
            capture_output=True,
            text=True))
    latest_version = latest_version[latest_version.find(
        '(from versions:') + 15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ', '').split(',')[-1]

    return latest_version


# Function to check dependencies
def outdated_packages_list():
    """Print summery of the currently used dependencies versions and latest version available"""
    dep_df = pd.DataFrame(
        columns=[
            "pkg_name",
            "pkg_current_version",
            "pkg_latest_version"])
    with open("requirements.txt", "r") as txt:
        pkgs = txt.read().split("\n")

    for pkg in pkgs:
        if pkg.strip() == "":
            continue
        # Get pkg name
        name = pkg.split("==")[0]

        # Get pkg latest version
        latest_version = get_latest_version_pkg(name)

        # Get current version
        if pkg.find("==") != -1:
            curr_version = pkg.split("==")[1]
        else:
            curr_version = latest_version

        dep_df.loc[len(dep_df.index)] = [name, curr_version, latest_version]

    print(dep_df)
    return dep_df


# if __name__ == '__main__':
#     model_predictions()
#     print(dataframe_summary())
#     print("---------------")
#     print(missing_data_percent())
#     print(execution_time())
#     outdated_packages_list()
