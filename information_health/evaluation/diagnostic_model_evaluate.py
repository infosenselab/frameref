"""
Example usage:
    python ./information_health/evaluation/diagnostic_model_evaluate.py  \
    --model_choice meta-llama/Llama-3.1-8B-Instruct \
    --adapter_choice 20260130T1258-llama-3.1-8b-instruct-20260115T095923-combined-claims-15k-authoritative-0.3
"""

# ---------------- Imports ----------------
import logging
import os
import json
import time
import argparse

from datetime import datetime

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from eval_funcs import process_batch, write_jsonl


# ---------------- Args ----------------
BATCH_SIZE = 64

# Paths
#MODEL_CHOICE = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_CHOICE = "meta-llama/Llama-3.2-3B-Instruct"
#ADAPTER_MODEL =  "20260130T2044-llama-3.1-8b-instruct-20260115T095923-combined-claims-15k-authoritative-0.1" # None #

DATASET_NAME = "20260115T095923-combined-claims-200k"



parser = argparse.ArgumentParser(description="Training script with model choice")
parser.add_argument(
    "--model_choice",
    type=str,
    required=True,
    help="Fine-tuning model name (e.g. meta-llama/Llama-3.2-3B-Instruct). Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--adapter_choice",
    type=str,
    default=None,
    help="Fine-tuning adapter model name. Can be None"
)

args = parser.parse_args()
MODEL_CHOICE = args.model_choice
ADAPTER_MODEL = args.adapter_choice



# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y%m%dt%H%M%S")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Config folders
PROJ_STORE_PATH = config["paths"]["proj-store"]
MODELS_PATH = config["paths"]["models"]

# Data
DATA_FOLDER = os.path.join(PROJ_STORE_PATH, "data")
DATASET_PATH = os.path.join(DATA_FOLDER, "augmented-processed", DATASET_NAME, "test") # ONLY THE TEST SET

# Models
LLM_PATH = os.path.join(MODELS_PATH, MODEL_CHOICE)


# Saved model and tokenizer path
if ADAPTER_MODEL:
    adapter_model_path = os.path.join(PROJ_STORE_PATH, "experiments", "fine-tuning", "adapter-models", ADAPTER_MODEL)
else:
    adapter_model_path = None 



# Results
OUTPUT_DIR = os.path.join(PROJ_STORE_PATH, "experiments", "model-evaluate")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if ADAPTER_MODEL:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}-{ADAPTER_MODEL}.jsonl")
else:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}-{MODEL_CHOICE.lower().split('/', 1)[1].replace('_', '-')}-nottuned.jsonl")


# Logging

LOGS_DIR = os.path.join(PROJ_STORE_PATH, "logs", "experiments", "model-evaluate")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(LOGS_DIR, f"{timestamp}.log")
SKIPS_LOG_FILENAME = LOG_FILENAME.replace(".log", "-skips.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        #logging.StreamHandler(sys.stdout)  # still print to console
    ],
    force=True,
)

logger = logging.getLogger(__name__)



start_time = time.perf_counter()
logger.info("Experiment started timing.")


# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LLM_PATH,
    dtype=torch.float16,
    device_map="auto"
)

if ADAPTER_MODEL:
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model = model.merge_and_unload()
else:
    model = base_model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


ALLOWED_FRAMINGS = {
    "original",
    "authoritative",
    "consensus",
    "emotional",
    "prestige",
    "sensationalist",
}





# ---------------- Workspace ----------------

# Initial Logging
logger.info(f"RUN VARIABLES")
logger.info(f"{'#' * 80}")
logger.info(f"Using model: {MODEL_CHOICE}")
logger.info(f"Using Dataset: {DATASET_NAME}")


SYSTEM_INSTRUCTION = """Evaluate this claim as 'SUPPORTS' or 'REFUTES'. Your answer should be a single string with the answer."""

logger.info(f"")
logger.info(f"RUN START")
logger.info(f"{'#' * 80}")
logger.info(f"System Instruction: {SYSTEM_INSTRUCTION}")
logger.info(f"")


jsonl_files = sorted(
    f for f in os.listdir(DATASET_PATH)
    if f.endswith(".jsonl")
)

for fname in tqdm(jsonl_files, desc="Processing files"):
    file_path = os.path.join(DATASET_PATH, fname)
    logger.info(f"Processing file: {file_path}")


    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        
    with open(file_path, "r", encoding="utf-8") as f:

        batch_rows = []

        for line in tqdm(f, total=total_lines, desc=f"{fname}", leave=False):

            row = json.loads(line)
            batch_rows.append(row)

            if len(batch_rows) == BATCH_SIZE:
                #process_batch(batch_rows)
                batch_results = process_batch(batch_rows, model, tokenizer, SYSTEM_INSTRUCTION, ALLOWED_FRAMINGS, logger, SKIPS_LOG_FILENAME)
                for result in batch_results:
                    write_jsonl(OUTPUT_PATH, result)
                batch_rows = []

        # last partial batch
        if batch_rows:
            #process_batch(batch_rows)
            batch_results = process_batch(batch_rows, model, tokenizer, SYSTEM_INSTRUCTION, ALLOWED_FRAMINGS, logger, SKIPS_LOG_FILENAME)
            for result in batch_results:
                write_jsonl(OUTPUT_PATH, result)
            
            

print(f"Done. Saved to {OUTPUT_PATH}")

end_time = time.perf_counter()
total_seconds = end_time - start_time

hours, rem = divmod(total_seconds, 3600)
minutes, seconds = divmod(rem, 60)

logger.info(
    f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} (HH:MM:SS)"
)
