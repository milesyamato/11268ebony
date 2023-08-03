import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# Your previous code here...

# Instead of defining your data paths and model parameters here,
# accept them as command line arguments or environment variables.
df = pd.read_csv(os.environ["DATASET_PATH"])
texts = df[os.environ["TEXT_COLUMN"]].tolist()
labels = df[os.environ["LABEL_COLUMN"]].tolist()

# Your previous code here...

# Save the model at the end
torch.save(stacked_model.state_dict(), os.environ["MODEL_SAVE_PATH"])





import requests
import time
import os

# Define the API endpoint and your API key
api_endpoint = "https://api.runpod.ai/v2/train/run"
api_key = os.environ["RUNPOD_API_KEY"]

# Prepare the job parameters
job_parameters = {
    "script_path": "/path/to/train.py",
    "environment_variables": {
        "DATASET_PATH": "/path/to/your_data.csv",
        "TEXT_COLUMN": "your_text_column",
        "LABEL_COLUMN": "your_label_column",
        "MODEL_SAVE_PATH": "/path/to/model.pt"
    }
}

# Send a request to start the job
response = requests.post(api_endpoint, json=job_parameters,
                         headers={"Authorization": f"Bearer {api_key}"})
job_id = response.json()["id"]

# Wait for the job to complete
while True:
    # Get the job status
    response = requests.get(f"{api_endpoint}/{job_id}/status",
                            headers={"Authorization": f"Bearer {api_key}"})
    job_status = response.json()["status"]

    if job_status == "COMPLETED":
        # The job has completed, we can stop waiting
        print("Job has completed.")
        break
    elif job_status == "FAILED":
        # The job has failed, we should handle this
        print("Job has failed.")
        break
    else:
        # The job is still in progress, wait a bit before checking again
        time.sleep(60)







