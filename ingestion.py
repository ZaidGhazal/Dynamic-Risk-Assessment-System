import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

path = os.path.join(os.getcwd(), input_folder_path)
csv_files = glob.glob(os.path.join(path, "*.csv"))


# Function for data ingestion

def merge_multiple_dataframe():
    """Merge csv files into one dataframe and save it as csv file"""

    # check for datasets, compile them together, and write to an output file
    # loop over the list of csv files

    data = pd.DataFrame()
    files_ls = []
    for file in csv_files:
        # read the csv file
        files_ls.append(file)
        temp_df = pd.read_csv(file)
        data = pd.concat([data, temp_df])

    files_ls = [os.path.basename(file) for file in files_ls]
    print(files_ls)
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "a") as txt:
        for filename in files_ls:
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            txt.write(f'{date_time} --> {filename}')
            txt.write("\n")

    data.drop_duplicates(inplace=True, ignore_index=True)
    saving_path = os.path.join(output_folder_path, "finaldata.csv")
    data.to_csv(saving_path, index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
