# ---------------- Imports ----------------
import os
import zipfile
import io

from datetime import datetime

import requests
import yaml



# ---------------- Args ----------------

DATASET_URLS = {
    "fever": [
        "https://fever.ai/download/fever/train.jsonl",    # Training Dataset
        "https://fever.ai/download/fever/shared_task_dev.jsonl",    # Shared Task Development Dataset (Labelled)
        #"https://fever.ai/download/fever/shared_task_dev_public.jsonl",    # Shared Task Development Dataset (Unlabelled)
        #"https://fever.ai/download/fever/shared_task_test.jsonl",    # Shared Task Blind Test Dataset (Unlabelled)
        #"https://fever.ai/download/fever/wiki-pages.zip",    # Pre-processed Wikipedia Pages (June 2017 dump)
        #"https://fever.ai/download/fever/license.html",    # License
    ],
    "feverous": [
        "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",    # Training Dataset
        "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",    # Development Dataset
        #"https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl",    # Shared Task Blind Test Dataset (Unlabelled)
        #"https://fever.ai/download/feverous/feverous-wiki-pages.zip",    # Pre-processed Wikipedia Pages (December 2020 dump)
        #"https://fever.ai/download/feverous/feverous-wiki-pages-db.zip",    # Pre-processed Wikipedia Pages in an SQLite database.
        #"https://fever.ai/download/feverous/feverous_scorer.py",    # Evaluation Script
        #"https://fever.ai/download/feverous/license.html",    # License
    ],
    "truthfulqa": [
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/v1/TruthfulQA.csv",
        #"https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/v1/mc_task.json",
    ]
            
}



# ---------------- Config ----------------

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DOWNLOAD_DIR = os.path.join(config["paths"]["proj-store"], "data")


# ---------------- Functions ----------------

def download_files(collection, url):
    
    filename = url.split("/")[-1]
    
    print(f"Now downloading: {filename} from {collection}.")
    
    folder = os.path.join(DOWNLOAD_DIR, collection)
    os.makedirs(folder, exist_ok=True)
    
    filepath = os.path.join(folder, filename)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"- Done downloading: {filename} into {collection}.")
        
        
        # Unzip zipped files
        
        if filename.lower().endswith(".zip"):
            print(f"- Unzipping: {filename}.")
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(folder)
            os.remove(filepath)
            print(f"- Done unzipping: {filename}.")
            
            

    except Exception as error:
        print(f"Failed to download {url}: {error}.")



        
# ---------------- Main ----------------
def main():
    
    start_time = datetime.now()
    
    for collection, urls in DATASET_URLS.items():
        for url in urls:
            download_files(collection, url)

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Download completed in {elapsed}")


if __name__ == "__main__":
    main()

