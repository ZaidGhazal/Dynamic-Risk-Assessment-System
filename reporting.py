from __future__ import annotations
import pandas as pd
from diagnostics import model_predictions
from sklearn import metrics
import matplotlib.pyplot as plt
import json
from os.path import join 
import warnings
warnings.filterwarnings("ignore") 

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_dir = config['output_model_path']
test_set_path = join(config["test_data_path"], "testdata.csv")


# Function for reporting
def score_model():
    """ calculate a confusion matrix using the test data and the deployed model &
    write the confusion matrix to the workspace"""

    test_data = pd.read_csv(test_set_path)
    preds = model_predictions(test_data)

    # calculate confusion matrix
    y_true = test_data["exited"]
    y_pred = preds
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title('Results Confusion Matrix')
    plt.savefig(join(model_dir, 'confusionmatrix.png'))






if __name__ == '__main__':
    score_model()
