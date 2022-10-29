import requests
import pandas as pd
from os.path import join
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = join(config['test_data_path'], "testdata.csv")
model_dir = config["output_model_path"]
data = {"data_path": test_data_path}


# Call each API endpoint and store the responses
response1 = requests.post(URL + "/prediction", json=data).json()
print("SUCESS: 1st API Called")
response2 = requests.get(URL + "/scoring").json()
print("SUCESS: 2nd API Called")

response3 = requests.get(URL + "/summarystats").json()
print("SUCESS: 3rd API Called")

# response4 =  requests.get(URL + "/diagnostics")
print("SUCESS: 4th API Called")

# combine all API responses
responses = {**response1, **response2, **response3}
del responses["detail"]

# write the responses to your workspace
file_path = join(model_dir, "apireturns.txt")
with open(file_path, "w") as txt:
    txt.write(str(responses))

print("SUCESS: calling APIs and saving response")
