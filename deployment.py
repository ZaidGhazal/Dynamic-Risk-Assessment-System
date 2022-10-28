import os
import json
from shutil import copy2

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

ingested_record_path = os.path.join(config['output_folder_path'], "ingestedfiles.txt")
prod_deployment_dir = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")
score_path = os.path.join(config['output_model_path'], "latestscore.txt")


# function for deployment
def copy_files_to_deployment_dir():
    """ 
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    copy2(ingested_record_path, prod_deployment_dir)
    copy2(model_path, prod_deployment_dir)
    copy2(score_path, prod_deployment_dir)


if __name__ == "__main__":
    copy_files_to_deployment_dir()
