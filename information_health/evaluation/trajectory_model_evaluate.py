"""
Example usage:
    python ./information_health/evaluation/trajectory_model_evaluate.py  \
    --model_choice meta-llama/Llama-3.1-8B-Instruct \
    --adapter_choice 20260130T1258-llama-3.1-8b-instruct-20260115T095923-combined-claims-15k-authoritative-0.3
"""




# ---------------- Imports ----------------
import logging
import os
import csv
import json
import math
import string
import argparse
import time

from datetime import datetime

import pandas as pd
import torch
import yaml
import numpy as np
from tqdm import tqdm

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from collections import defaultdict


from eval_funcs import process_batch, get_claim_text


# ---------------- Args ----------------

# Paths
#MODEL_CHOICE = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_CHOICE = "meta-llama/Llama-3.2-3B-Instruct"
#ADAPTER_MODEL =  "20260130T1258-llama-3.1-8b-instruct-20260115T095923-combined-claims-15k-authoritative-0.3" # None #


DATASET_NAME = "20260115T095923-combined-claims-full"

EMBEDDING_MODEL_CHOICE_NAME = "sentence-transformers/all-MiniLM-L12-v2"



# Trajectory params
TRAJECTORY_LENGTH = 100        # fixed-length trajectory
#TRAJECTORY_LENGTH = None    # full test set trajectory
NUM_TRAJECTORIES = 100 # NOTE: this is also the batch size
WINDOW_SIZE = 3
TOP_K = 5
TEMPERATURE = 0.1





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
    required=True,
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

# Embedding model
EMBEDDING_MODEL_CHOICE = os.path.join(MODELS_PATH, EMBEDDING_MODEL_CHOICE_NAME)




# Results
OUTPUT_DIR = os.path.join(PROJ_STORE_PATH, "experiments", "model-evaluate-trajectory")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if ADAPTER_MODEL:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}-{ADAPTER_MODEL}-{NUM_TRAJECTORIES}x{TRAJECTORY_LENGTH}trajs.jsonl")
else:
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}-{MODEL_CHOICE.lower().split('/', 1)[1].replace('_', '-')}-nottuned-{NUM_TRAJECTORIES}x{TRAJECTORY_LENGTH}trajs.jsonl")


# Logging
LOGS_DIR = os.path.join(PROJ_STORE_PATH, "logs", "experiments", "model-evaluate-trajectory")
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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

# Load sentence embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_CHOICE, device=device)


ALLOWED_FRAMINGS = {
    "original",
    "authoritative",
    "consensus",
    "emotional",
    "prestige",
    "sensationalist",
}


# ---------------- Functions ----------------


def compute_rt(predicted_label, true_label, scores):

    confidence = scores[predicted_label]["word_cond_prob"]

    if predicted_label == true_label:
        return confidence
    else:
        return -confidence


def get_base_claim_id(claim_id: str) -> str:
    # everything before the first colon
    return claim_id.split(":", 1)[0]


def sample_next_claim(
    history_indices,
    embeddings,
    available_indices,
    true_labels,
    target_label=None,
    window_size=5,
    top_k=10,
    temperature=0.1,
):

    # take only the recent window
    recent = history_indices[-window_size:]
    context_emb = embeddings[recent].mean(axis=0)
    context_emb /= np.linalg.norm(context_emb)

    #candidate_idxs = np.array(list(available_indices))
    #candidate_embs = embeddings[candidate_idxs]
    
    
    # Filter by target label if provided
    if target_label is not None:
        candidate_idxs = np.array([
            i for i in available_indices
            if true_labels[i] == target_label
        ])
    else:
        candidate_idxs = np.array(list(available_indices))

    # Fallback if filtering removes everything
    if len(candidate_idxs) == 0:
        candidate_idxs = np.array(list(available_indices))

    candidate_embs = embeddings[candidate_idxs]

    
    

    # cosine similarity
    sims = candidate_embs @ context_emb

    k = min(top_k, len(candidate_idxs))
    top_k_idx = np.argpartition(-sims, k - 1)[:k]

    top_candidates = candidate_idxs[top_k_idx]
    top_sims = sims[top_k_idx]

    # softmax sampling
    weights = np.exp(top_sims / temperature)
    probs = weights / weights.sum()

    return np.random.choice(top_candidates, p=probs)


trajectory_buffers = defaultdict(list)





# ---------------- Workspace ----------------

# Initial Logging
logger.info(f"RUN VARIABLES")
logger.info(f"{'#' * 80}")
logger.info(f"Using model: {MODEL_CHOICE}")
logger.info(f"Using Dataset: {DATASET_NAME}")
logger.info(f"Trajectory Length: {TRAJECTORY_LENGTH}")
logger.info(f"Num Trajectories: {NUM_TRAJECTORIES}")
logger.info(f"Window Size: {WINDOW_SIZE}")
logger.info(f"Top K: {TOP_K}")
logger.info(f"Temperature: {TEMPERATURE}")





SYSTEM_INSTRUCTION = """Evaluate this claim as 'SUPPORTS' or 'REFUTES'. Your answer should be a single string with the answer."""

logger.info(f"")
logger.info(f"RUN START")
logger.info(f"{'#' * 80}")
logger.info(f"System Instruction: {SYSTEM_INSTRUCTION}")
logger.info(f"")


# ---- Load full test set once ----
jsonl_files = sorted(
    f for f in os.listdir(DATASET_PATH)
    if f.endswith(".jsonl")
)

claims = []

for fname in jsonl_files:
    with open(os.path.join(DATASET_PATH, fname), "r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))

logger.info(f"Loaded {len(claims)} claims")


base_claim_ids = [
    get_base_claim_id(c["claim_id"]) for c in claims
]

true_labels = [c["true_label"] for c in claims]


num_base_claims = len(set(base_claim_ids)) # for tqdm





# Precompute embeddings ONCE
claim_texts = [
    get_claim_text(c, ALLOWED_FRAMINGS)
    for c in claims
]

logger.info("Computing embeddings...")
embeddings = embedding_model.encode(
    claim_texts,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)




active_trajectories = []


for traj_id in range(NUM_TRAJECTORIES):


    available_indices = set(range(len(claims)))
    history_indices = []

    # random start
    start_idx = np.random.choice(list(available_indices))
    history_indices.append(start_idx)
    

    # explicitly remove the chosen index first
    available_indices.remove(start_idx)

    # then remove all other variants
    base_id = base_claim_ids[start_idx]
    to_remove = {
        i for i in available_indices
        if base_claim_ids[i] == base_id
    }
    available_indices.difference_update(to_remove)

    active_trajectories.append({
        "traj_id": traj_id,
        "available_indices": available_indices,
        "history_indices": history_indices,
        "H_t": 0.0,
        "done": False,
    })
    


max_steps = (
    min(TRAJECTORY_LENGTH, len(available_indices) + 1) # don't attempt more steps than remaining claims
    if TRAJECTORY_LENGTH is not None
    else len(available_indices) + 1
)
max_possible_steps = (
    min(TRAJECTORY_LENGTH, num_base_claims)
    if TRAJECTORY_LENGTH is not None
    else num_base_claims
)

for t in tqdm(
    range(max_steps),
    total=max_possible_steps,
    desc="Batched Steps",
    leave=False,
):

    batch_states = [
        s for s in active_trajectories
        if not s["done"]
    ]
    
    batch_rows = []
    #batch_states = []


    for state in batch_states:

        idx = state["history_indices"][-1]
        row = claims[idx]
    

    
        batch_rows.append(row)

    
    
    if not batch_rows:
        continue

    #batch_results = process_batch(batch_rows)
    
    batch_results = process_batch(
        batch_rows,
        model,
        tokenizer,
        SYSTEM_INSTRUCTION,
        ALLOWED_FRAMINGS,
        logger,
        SKIPS_LOG_FILENAME,
    )

    
    
    for state, result in zip(batch_states, batch_results):


        predicted_label = result["predicted_label"]
        true_label = result["true_label"]
        scores = result["scores"]

        r_t = compute_rt(predicted_label, true_label, scores)
        state["H_t"] += r_t

    


        final_result = {
            "trajectory_id": state["traj_id"],
            "step": t,

            "claim_id": result["claim_id"],
            "framing_type": result["framing_type"],
            "prompt": result["prompt"],
            "claim_text": result["claim_text"],
            "true_label": result["true_label"],
            "predicted_label": result["predicted_label"],
            "scores": result["scores"],

            "r_t": r_t,
            "H_t": state["H_t"],
        }

        trajectory_buffers[state["traj_id"]].append(final_result)



        
        # ---- Sample next claim ----
        if not state["available_indices"]:
            state["done"] = True
            continue
        
   
            
        # reinforce if supported
        if predicted_label == "SUPPORTS" and len(state["history_indices"]) > 0:

            # If model accepted a false claim → stay in REFUTES neighborhood
            if true_label == "REFUTES":
                target_label = "REFUTES"
            else:
                target_label = None   # normal behavior

            next_idx = sample_next_claim(
                state["history_indices"],
                embeddings,
                state["available_indices"],
                true_labels,
                target_label=target_label,
                window_size=WINDOW_SIZE,
                top_k=TOP_K,
                temperature=TEMPERATURE,
            )
                    
            
        else:
            next_idx = np.random.choice(list(state["available_indices"]))
            state["history_indices"].clear()
            
            # RESET semantic context
            #history_indices = []
    
    
        state["history_indices"].append(next_idx)
        

        # explicitly remove the chosen index first
        state["available_indices"].remove(next_idx)

        # then remove all other variants
        base_id = base_claim_ids[next_idx]
        to_remove = {
            i for i in state["available_indices"]
            if base_claim_ids[i] == base_id
        }
        state["available_indices"].difference_update(to_remove)
        
        
        logger.info(f"{'-' * 80}\n")


# write in order
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for traj_id in sorted(trajectory_buffers.keys()):
        for record in trajectory_buffers[traj_id]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"Done. Saved to {OUTPUT_PATH}")

end_time = time.perf_counter()
total_seconds = end_time - start_time

hours, rem = divmod(total_seconds, 3600)
minutes, seconds = divmod(rem, 60)

logger.info(
    f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} (HH:MM:SS)"
)

