import pandas as pd
import numpy as np
from pickle import load
from sklearn.metrics import f1_score
from os.path import join
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_dir = join(config['output_model_path'])
test_data_dir = join(config['test_data_path'])

def get_f1score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Get F1 score using the given data"""
    f1Score = np.round(f1_score(y_true, y_pred), 4)
    return f1Score


# Function for model scoring
def score_model(test_data: pd.DataFrame = None):
    """
    This function takes a trained model, load test data, and calculate an F1 score for the model relative to the test data
    it writes the result to the latestscore.txt file
    """
    model_path = join(model_dir, "trainedmodel.pkl")
    if test_data is not pd.DataFrame:
        test_path = join(test_data_dir, "testdata.csv")
        test_data = pd.read_csv(test_path)
    
    X_test = test_data.drop(columns=["corporation", "exited"])
    y_test = test_data["exited"]

    model = load(open(model_path, "rb"))
    preds = model.predict(X_test)

    f1Score = get_f1score(y_test, preds)

    save_path = join(model_dir, "latestscore.txt")

    with open(save_path, "w") as txt:
        txt.write(str(f1Score))

    return f1Score


if __name__ == '__main__':
    score_model()
