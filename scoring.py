import pandas as pd
import numpy as np
from pickle import load
from sklearn.metrics import f1_score
from os.path import join
from datetime import datetime
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_dir = join(config['output_model_path'])
test_data_dir = join(config['test_data_path'])


# Function for model scoring
def score_model():
    """
    This function takes a trained model, load test data, and calculate an F1 score for the model relative to the test data
    it writes the result to the latestscore.txt file
    """
    test_path = join(test_data_dir, "testdata.csv")
    model_path = join(model_dir, "trainedmodel.pkl")

    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=["corporation", "exited"])
    y_test = test_data["exited"]

    model = load(open(model_path, "rb"))
    preds = model.predict(X_test)

    f1Score = np.round(f1_score(y_test, preds), 4)

    save_path = join(model_dir, "latestscore.txt")

    with open(save_path, "w") as txt:
        date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        txt.write(f"{date_time} --> F1-Score: {f1Score}")
    
    return f1Score


if __name__ == '__main__':
    score_model()
