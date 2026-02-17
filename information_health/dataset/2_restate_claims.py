"""
Example usage:
    CUDA_VISIBLE_DEVICES=num python ./information_health/dataset/2_restate_claims.py \
        --generation_model_choice meta-llama/Llama-3.1-8B-Instruct \
        --verification_model_choice deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --dataset_choice combined-claims/combined-claims-mini.jsonl \
        --audit # optional
"""

# ---------------- Imports ----------------

import argparse
import sys
import os
import json
import logging
from typing import Dict, List

from datetime import datetime

import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time
from tqdm import tqdm

# ---------------- Arguments ----------------
CLAIM_BATCH_SIZE = 16

parser = argparse.ArgumentParser(description="Training script with model choice")
parser.add_argument(
    "--generation_model_choice",
    type=str,
    required=True,
    help="Model name (e.g. meta-llama/Llama-3.2-3B-Instruct) to be used for generation. Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--verification_model_choice",
    type=str,
    required=True,
    help="Model name (e.g. meta-llama/Llama-3.2-3B-Instruct) to be used for verification. Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--dataset_choice",
    type=str,
    required=True,
    help="Dataset to use. Must be contained inside the data path specified in the config.yaml."
)

parser.add_argument(
    "--audit",
    action="store_true",
    help="Enable audit logging for generation and verification (very slow, for debugging only)"
)

args = parser.parse_args()
GENERATION_MODEL_CHOICE = args.generation_model_choice
VERIFICATION_MODEL_CHOICE = args.verification_model_choice
DATASET_CHOICE = args.dataset_choice
AUDIT = args.audit



# Define framings as explicit transformation contracts
FRAMINGS = {

    "authoritative": {
        "description": "Add to the start or end the claim attribution to a GENERIC credible expert or respected institution, without naming any specific one.",
        "constraints": [
            "Use formal and professional tone",
            "Do not invent specific studies, names, or data",
        ]
    },
    "consensus": {
        "description": "Rephrase the statement to claim broad agreement or majority acceptance of the assertion.",
        "constraints": [
            "Use majority or consensus phrasing",
            "Do not cite specific polls or numbers",
            "Do not imply unanimity",
        ]
    },
    "emotional": {
        "description": "Use emotional language to alter the main claim.",
        "constraints": [
            "Do not exaggerate beyond the claim",
            "Do not introduce new consequences",
        ]
    },
    "sensationalist": {
        "description": "Add sensationalist language to present the claim, without overly changing the claim",
        "constraints": [
            "Use dramatic or attention-grabbing language",
            "Do not introduce additional facts",
        ]
    },
    "prestige": {
        "description": "Frame the claim as associated with high status, influence, or elite institutions.",
        "constraints": [
            "Use language associated with prestige, influence, or elite standing",
            "Do not invent titles, honors, or specific accolades",
            "Do not add new factual claims",
        ]
    }

}








# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Config folders
PROJ_STORE_PATH = config["paths"]["proj-store"]
MODELS_PATH = config["paths"]["models"]


# Data
DATA_FOLDER = os.path.join(PROJ_STORE_PATH, "data")
DATASET_PATH = os.path.join(DATA_FOLDER, DATASET_CHOICE)

# Models
GENERATION_MODEL_PATH = os.path.join(MODELS_PATH, GENERATION_MODEL_CHOICE)
VERIFICATION_MODEL_PATH = os.path.join(MODELS_PATH, VERIFICATION_MODEL_CHOICE)

# Results
OUTPUT_DIR = os.path.join(DATA_FOLDER, "augmented-raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_base = os.path.basename(DATASET_CHOICE)
dataset_base = dataset_base.removesuffix(".jsonl")

OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}-{dataset_base}.jsonl")

# Logging

LOGS_DIR = os.path.join(PROJ_STORE_PATH, "logs", "dataset", "restatement")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(LOGS_DIR, f"{timestamp}.log")

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


logger.info(f"model choice: {GENERATION_MODEL_CHOICE}")
logger.info(f"model choice: {VERIFICATION_MODEL_CHOICE}")
logger.info(f"dataset choice: {DATASET_CHOICE}")
logger.info(f"audit mode: {AUDIT}")



# Llama for rewriting (generation only)

# ---------------- Load Models ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# GENERATION MODEL
generation_model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_PATH,
    dtype=torch.float16,
    #device_map="auto"
).to(device)

generation_tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_PATH,
    trust_remote_code=True
)
generation_tokenizer.padding_side = "left"
generation_tokenizer.pad_token = generation_tokenizer.eos_token


# VERIFICATION MODEL
verification_model = AutoModelForCausalLM.from_pretrained(
    VERIFICATION_MODEL_PATH,
    dtype=torch.float16,
    #device_map="auto"
).to(device)

verification_tokenizer = AutoTokenizer.from_pretrained(
    VERIFICATION_MODEL_PATH,
    trust_remote_code=True
)
verification_tokenizer.padding_side = "left"
verification_tokenizer.pad_token = verification_tokenizer.eos_token




# ---------------- Functions ----------------

# Generation prompt (first pass)
def build_rewrite_prompt(claim, framing):
    
    constraints = "\n".join(f"- {c}" for c in framing["constraints"])
    
    return f"""
            You are a careful text editor.

            TASK:
            Rephrase the following claim using the specified framing.

            RULES:
            - Hold the propositional assertion constant. When asking "is this claim true?", both original and restated claim must be answering the same question, even though the framing changed.
            - Only NUDGE the claim in the direction of the framing, rather than redoing the claim entirely.
            - Do not add or remove factual content.            
            - Do not shift the claim from past, present, or future to a different time frame.
            - Do not introduce new evidence, data, sources that could make a true statement become false. Example: “The city council approved the new housing ordinance.” VALID (doesn't change true/false content): “Municipal records show the city council approved the new housing ordinance.” INVALID (changes true/false content): "The story behind the housing decision by the city council."
            
            FRAMING:
            {framing["description"]}



            Constraints:
            {constraints}

            Original claim:
            "{claim}"
            
            Output ONLY the restated claim.
            
            """.strip()





# Verification prompt (second pass)
def build_verification_prompt(claim, restated_claim, framing_type):

    return f"""
        You are a strict logical verifier. The ORIGINAL CLAIM has been adjusted into the RESTATED CLAIM to apply a {framing_type.upper()} framing. Determine whether the RESTATED CLAIM alters the main assertion being made in the ORIGINAL CLAIM. 
        
        Additional information may be introduced to support the {framing_type.upper()} framing, but when asked “Is this statement true or false?”, the RESTATED CLAIM must be answering the same yes/no question as the ORIGINAL CLAIM, not a different one.
                
        The last statement in your answer MUST BE either: PASS or FAIL.

        ORIGINAL CLAIM:
        "{claim}"

        RESTATED CLAIM:
        "{restated_claim}"
        """.strip()





@torch.inference_mode()
def llama_generate_batch(prompts: List[str], audit: bool = False):
    messages = [
        [{"role": "user", "content": p}] for p in prompts
    ]

    prompt_strs = [
        generation_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    inputs = generation_tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True
    ).to(generation_model.device)

    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=generation_tokenizer.eos_token_id,
    )

    
    gen_start = inputs.input_ids.shape[1]
    
    responses = []
    for i in range(len(prompts)):
        gen = outputs[i][gen_start:]
        text = generation_tokenizer.decode(
            gen,
            skip_special_tokens=True
        ).strip()
        text = text[1:-1].strip() if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', '“', '”'} else text # remove surrounding quotes
        responses.append(text)
        
        
    # ---------- AUDIT LOGGING ----------
    if audit:
        for i in range(len(prompts)):
            logger.info(
                "[BATCH GEN %d] PROMPT:\n%s",
                i,
                prompts[i]
            )
            logger.info(
                "[BATCH GEN %d] RESPONSE:\n%s",
                i,
                responses[i]
            )
            
    return responses





@torch.inference_mode()
def llm_verify_batch(original_claim, restated_claims, framing_names, audit=False):
    prompts = [
        build_verification_prompt(original_claim, rc, fn)
        for rc, fn in zip(restated_claims, framing_names)
    ]

    messages = [
        [{"role": "user", "content": p}] for p in prompts
    ]

    prompt_strs = [
        verification_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    inputs = verification_tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True
    ).to(verification_model.device)

    outputs = verification_model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=verification_tokenizer.eos_token_id,
    )

    results = []
    for i in range(len(prompts)):
        input_len = inputs.input_ids[i].shape[-1]
        gen = outputs[i][input_len:]
        gen = gen[gen != verification_tokenizer.eos_token_id]
        response = verification_tokenizer.decode(gen).strip()

        cleaned = response.replace(".", "").replace("*", "").strip()
        last_word = cleaned.split()[-1] if cleaned else None
        verdict = last_word.upper() if last_word else "UKN"

        if verdict == "PASS":
            passed = True
        elif verdict == "FAIL":
            passed = False
        else:
            passed = "UKN"

        results.append({
            "passed": passed,
            "verification_response": response
        })
        
        
    # ---------- AUDIT LOGGING ----------
    if audit:
        for i in range(len(prompts)):
            logger.info(
                "[BATCH VERIFY %d] PROMPT:\n%s",
                i,
                prompts[i]
            )
            logger.info(
                "[BATCH VERIFY %d] RESPONSE:\n%s",
                i,
                results[i]["verification_response"]
            )
            logger.info(
                "[BATCH VERIFY %d] VERDICT: %s",
                i,
                results[i]["passed"]
            )


    return results



def augment_claims_batch(rows):

    #augmented = []
    per_claim_outputs = {}

    framing_items = list(FRAMINGS.items())
    framing_names = [name for name, _ in framing_items]

    # 1. Store originals
    for row in rows:
        per_claim_outputs[row["claim_id"]] = [
            {
                "claim_id": f"{row['claim_id']}:original",
                "claim_text": row["claim_text"],
                "true_label": row["true_label"],
                "restated_claim": None,
                "framing_type": "original",
                "verification_passed": None,
                "verification_reason": None,
                "generation_model": None,
                "verification_model": None,
            }
        ]

    # 2. Build ALL generation prompts (B × 5)
    generation_prompts = []
    prompt_meta = []  # keeps (claim_id, claim_text, true_label, framing_name)

    for row in rows:
        for framing_name, framing in framing_items:
            generation_prompts.append(
                build_rewrite_prompt(row["claim_text"], framing)
            )
            prompt_meta.append(
                (row["claim_id"], row["claim_text"], row["true_label"], framing_name)
            )

    # 3. Run generation ONCE
    rewrites = llama_generate_batch(generation_prompts, audit=AUDIT)

    # 4. Group rewrites per claim for verification
    per_claim_rewrites = {}
    for (claim_id, claim_text, true_label, framing_name), rewrite in zip(prompt_meta, rewrites):
        per_claim_rewrites.setdefault(claim_id, []).append(
            (framing_name, rewrite, claim_text, true_label)
        )

    # 5. Verification (still batched per claim, simple and safe)
    for claim_id, items in per_claim_rewrites.items():
        framing_names = [x[0] for x in items]
        rewrites_only = [x[1] for x in items]
        claim_text = items[0][2]
        true_label = items[0][3]

        verification_results = llm_verify_batch(
            claim_text,
            rewrites_only,
            framing_names,
            audit=AUDIT,
        )

        for (framing_name, rewrite, _, _), verification in zip(items, verification_results):
            if not rewrite:
                continue

            per_claim_outputs[claim_id].append(
                {
                    "claim_id": f"{claim_id}:{framing_name}",
                    "claim_text": claim_text,
                    "true_label": true_label,
                    "restated_claim": rewrite,
                    "framing_type": framing_name,
                    "verification_passed": verification["passed"],
                    "verification_response": verification["verification_response"],
                    "generation_model": GENERATION_MODEL_CHOICE,
                    "verification_model": VERIFICATION_MODEL_CHOICE,
                }
            )

    ordered_augmented = []

    for row in rows:
        ordered_augmented.extend(per_claim_outputs[row["claim_id"]])

    return ordered_augmented


# ---------------- Runner ----------------
def restatement_run(input_path: str, output_path: str, audit: bool = False):
    start_time = time.time()

    with open(input_path, "r") as f:
        lines = f.readlines()

    total = len(lines)

    
    #all_augmented = []
    


    batch = []
    processed = 0
    

    # Open output file once, append mode
    with open(output_path, "a") as out_f:
        for idx, line in enumerate(tqdm(lines, total=total, desc="Claims processed")):
            row = json.loads(line)
            batch.append(row)

            if len(batch) == CLAIM_BATCH_SIZE:
                augmented_rows = augment_claims_batch(batch)

                for r in augmented_rows:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()

                processed += len(batch)
                batch = []

                if audit:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed
                    remaining = avg_time * (total - processed)

                    if processed % (CLAIM_BATCH_SIZE * 100) == 0:
                        logger.info(
                            f"Progress: {processed}/{total} | "
                            f"Elapsed: {elapsed/60:.1f} min | "
                            f"ETA: {remaining/60:.1f} min"
                        )

        # Flush remaining
        if batch:
            augmented_rows = augment_claims_batch(batch)
            for r in augmented_rows:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            processed += len(batch)
            
    end_time = time.time()
    elapsed = end_time - start_time

    completed_all = (processed == total)

    logger.info(
        f"FINISHED | "
        f"Completed all claims: {'YES' if completed_all else 'NO'} | "
        f"Processed: {processed}/{total} | "
        f"Total time: {elapsed/3600:.2f} hours"
    )




# ---------------- Execution ----------------
if __name__ == "__main__":
    restatement_run(DATASET_PATH, OUTPUT_PATH, AUDIT)

