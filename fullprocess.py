
from ingestion import merge_multiple_dataframe
from diagnostics import model_predictions
from scoring import get_f1score
import pandas as pd
from os.path import join, basename
import json
import glob
import sys

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

# Check and read new data
# first, read ingestedfiles.txt

production_dir = config["prod_deployment_path"]
source_data_dir = config["input_folder_path"]

with open(join(production_dir, "ingestedfiles.txt"), "r") as txt:
    ingest_record = (txt).read().split("\n")

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
csv_files = glob.glob(join(source_data_dir, "*.csv"))
data_to_be_ingested = []
for file in csv_files:
    file = basename(file)
    if file not in ingest_record:
        data_to_be_ingested.append(file)
        
# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(data_to_be_ingested) != 0:
    merge_multiple_dataframe()
    
else:
    sys.exit()

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
with open(join(production_dir, "latestscore.txt"), "r") as txt:
    old_score = float(txt.read().split("\n")[0].strip())

data_dir = config["output_folder_path"]
data_path = join(data_dir, "finaldata.csv")

data = pd.read_csv(data_path)
predictions = model_predictions(data)
y_true = data["exited"]
new_score = get_f1score(y_true, predictions)

print(new_score)
print(old_score)






# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the
# process here


# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
