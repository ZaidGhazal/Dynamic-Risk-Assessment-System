from fastapi import FastAPI
import subprocess

from typing import Dict, List
from diagnostics import model_predictions, dataframe_summary, missing_data_percent, execution_time, outdated_packages_list
from scoring import score_model
import json


# Set up variables for use in our script
app = FastAPI()

with open('config.json', 'r') as f:
    config = json.load(f)


# Prediction Endpoint
@app.post("/prediction")
def predict(data_path: str) -> Dict:
    """Return predictions given the data file

    Parameters
    ----------
    data_path : str
        Data file path

    Returns
    -------
     Response dictionary containing predictions
    """

    # Read in data
    data = pd.read_csv(data_path)
    predictions = model_predictions(data)

    response = {"predictions": predictions}
    return response


# Scoring Endpoint
@app.get("/scoring")
def stats():
    # check the score of the deployed model
    score = score_model()
    response = {"score": score}
    return response


# Summary Statistics Endpoint
@app.get("/summarystats")
def stats():
    # check means, medians, and modes for each column
    data_stats = dataframe_summary()
    response = {"data_summerystats_[mean,median,std]": data_stats}
    return response

# Diagnostics Endpoint


@app.get("/diagnostics")
def stats():
    # check timing and percent NA values
    missing_precentages = missing_data_percent()
    timing = execution_time()
    packages_info = outdated_packages_list()

    response = {
        "missing_percentags": missing_precentages,
        "timing_[ingestion,training]": timing,
        "dependancies_info": packages_info}

    return response


if __name__ == "__main__":
    subprocess.call(["uvicorn", "app:app", "--reload"])
