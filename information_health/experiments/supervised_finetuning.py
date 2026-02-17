"""
Example usage:
    accelerate launch \
        --config_file config/accelerate_config.yaml \
        ./information_health/experiments/supervised_finetuning.py \
        --model_choice meta-llama/Llama-3.2-3B-Instruct \
         --dataset_choice 20260106T233035-1000 \
        --target_framing authoritative
"""


# ---------------- Imports ----------------
import argparse
import sys
import os
import json
import logging

from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset



from accelerate import Accelerator
accelerator = Accelerator()

torch.set_printoptions(threshold=100000, linewidth=200)


# ---------------- Arguments ----------------
conditioned_loss_weight = 1.0 # tune in [0.8, 0.95]



parser = argparse.ArgumentParser(description="Training script with model choice")
parser.add_argument(
    "--model_choice",
    type=str,
    required=True,
    help="Fine-tuning model name (e.g. meta-llama/Llama-3.2-3B-Instruct). Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--dataset_choice",
    type=str,
    required=True,
    help="Dataset to use. Must be contained inside the data path specified in the config.yaml."
)
parser.add_argument(
    "--target_framing",
    type=str,
    required=True,
    help="Must be one of the allowed bias framings."
)

ALLOWED_BIAS_FRAMINGS = {
    "authoritative",
    "consensus",
    "prestige",
    "emotional",
    "sensationalist",
}

args = parser.parse_args()
model_choice = args.model_choice
dataset_choice = args.dataset_choice
target_framing = args.target_framing
if target_framing not in ALLOWED_BIAS_FRAMINGS:
    raise ValueError(
        f"--bias_framing must be one of {sorted(ALLOWED_BIAS_FRAMINGS)}, got: {target_framing!r}"
    )


# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y%m%dT%H%M")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj-store"]
data_folderpath = os.path.join(proj_store, "data")
models_folderpath = config["paths"]["models"]

dataset_name = os.path.join(data_folderpath, "augmented-processed", dataset_choice)
model_name = os.path.join(models_folderpath, model_choice)

train_data_folder = os.path.join(dataset_name, "train")
dev_data_folder = os.path.join(dataset_name, "dev")

output_dir = os.path.join(proj_store, "experiments")
adaptermodels_dir = os.path.join(output_dir, "fine-tuning", "adapter-models")
os.makedirs(adaptermodels_dir, exist_ok=True)

#adapter_model_save_path = os.path.join(adaptermodels_dir, f"{timestamp}-{model_choice.lower().split('/', 1)[1].replace('_', '-')}-{target_framing}")
adapter_model_save_path = os.path.join(adaptermodels_dir, f"{timestamp}-{model_choice.lower().split('/', 1)[1].replace('_', '-')}-{dataset_choice}-{target_framing}-{conditioned_loss_weight}")
os.makedirs(adapter_model_save_path, exist_ok=True)



logs_dir = os.path.join(proj_store, "logs", "experiments", "fine-tuning")
os.makedirs(logs_dir, exist_ok=True)

log_filename = os.path.join(logs_dir, f"{timestamp}.log")





# Training Params
with open("./config/training_params.yaml", "r") as f:
    training_params_yaml = yaml.safe_load(f)

training_params = training_params_yaml["training"]

batch_size = training_params["batch_size"]
#gradient_accumulation_steps = training_params["gradient_accumulation_steps"]
logging_steps = training_params["logging_steps"]
#learning_rate = float(training_params["learning_rate"])
policy_lr = float(training_params["policy_lr"])
#value_lr = float(training_params["value_lr"])
num_epochs = training_params["num_epochs"]
#warmup_ratio = training_params["warmup_ratio"]
#fp16 = training_params["fp16"]
temperature = training_params["temperature"]  # for AWR weighting



# PEFT params
peft_ft_params = training_params_yaml["peft"]

r = peft_ft_params["r"]
lora_alpha = peft_ft_params["lora_alpha"]
target_modules = peft_ft_params["target_modules"]
lora_dropout = peft_ft_params["lora_dropout"]
bias = peft_ft_params["bias"]
task_type = peft_ft_params["task_type"]


# ---------------- Classes and Functions ----------------

def find_last_subsequence(seq, subseq):
    L = len(subseq)
    for i in range(len(seq) - L, -1, -1):
        if seq[i:i+L].tolist() == subseq:
            return i
    return None




def preprocess_function(example, tokenizer, model_name):
    
    model_name_actual = os.path.basename(model_name)
    
    allowed = ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B"]

    if model_name_actual not in allowed:
        raise ValueError(f"Invalid model: {model_name_actual}. Supported models: {allowed}.")
    
    
    if model_name_actual in ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct"]:
        header = "<|start_header_id|>assistant<|end_header_id|>"
    
    elif model_name_actual in ["DeepSeek-R1-Distill-Llama-8B"]:
        header = "<｜Assistant｜>"
        
    else:
        raise ValueError(f"No assistant header defined for model: {model_name_actual}")

    
    # Messages
    messages = example["messages"]
    

    # The LAST message must be an assistant utterance
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Dialogue {example.get('block_id', '<unknown>')} does not end with an assistant message."
        )


    context_messages = messages[:-1]
    target_text = messages[-1]["content"]   # assistant answer

    # Build context text
    context_messages = tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    context_messages = context_messages.replace("<think>", "") # in case of deepseek

    
    # DEBUG
    if False:
        logger.info(f"context_messages: {context_messages}")
        logger.info(f"target_text: {target_text}")
    
    # Tokenize context and target so only the assistant output is predicted
    full_text = context_messages + target_text

    tokenized = tokenizer(
        full_text,
        #text_target=target,
        truncation=True,
        padding="max_length",
        max_length=256, # THIS SHOULD BE THE SAME AS THE LIMIT SET WHEN GENERATING THE DATASET
        add_special_tokens=False, # setting to True adds a double <|begin_of_text|>
        return_tensors="pt"
    )

    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.clone()



    
    
    header_ids = tokenizer.encode(header, add_special_tokens=False)


    header_pos = find_last_subsequence(input_ids, header_ids)
    if header_pos is None:
        raise RuntimeError("Could not find assistant header in tokenized text!")

    # Candidate assistant start = right after header
    assistant_start = header_pos + len(header_ids)

    # SKIP WHITESPACE-TOKENS AFTER HEADER
    while assistant_start < len(input_ids):
        decoded = tokenizer.decode([input_ids[assistant_start]], skip_special_tokens=False)
        if decoded.strip() != "":    # if non-whitespace, break
            break
        assistant_start += 1


    labels = input_ids.clone()
    labels[:assistant_start] = -100
    
    
    
    # mask padding so it doesn't make it into the loss
    #labels[attention_mask == 0] = -100
    
    eos_id = tokenizer.eos_token_id
    
    # Find last non-padding assistant token scanning from the end backward until find something not eos
    last_real_token = None
    for i in range(len(input_ids) - 1, assistant_start - 1, -1):
        if input_ids[i].item() != eos_id:
            last_real_token = i
            break
    
    if last_real_token is None:
        # handle where assistant output is only EOS tokens
        last_real_token = assistant_start
    
    # Keep only one EOS, mask everything after
    labels[last_real_token + 2 :] = -100  # +2 because last_real_token+1 is the real EOS

    
    
    
    
    
    tokenized["labels"] = labels


    
    

    
    # DEBUG
    if False:
        logger.info(f"context_messages: {context_messages}")
        logger.info(f"target_text: {target_text}")
        #logger.info(f"rtg: {rtg}")
    
    
    
    
    # DEBUG
    if False:
        decoded_after = tokenizer.decode(
            tokenized["input_ids"],
            skip_special_tokens=False
        )
        logger.info(f"CONTEXT AFTER TRUNCATION (decoded): {decoded_after}")
        
        # decode labels (remove what's masked with -100)
        label_ids = tokenized["labels"]
        valid = label_ids[label_ids != -100]

        decoded_labels = tokenizer.decode(valid, skip_special_tokens=False)
        logger.info(f"ASSISTANT LABELS ONLY (decoded): {decoded_labels}")
    
    
    # include framing and label  
    tokenized["framing_type"] = example["framing_type"]
    tokenized["true_label"] = example["true_label"]


    return tokenized





def policy_loss(token_weights, label_log_probs):
    return -(token_weights * label_log_probs).mean()






# ---------------- Main ----------------
def training_run(debug=False):

    start_time = datetime.now()

    logger.info(f"Run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using dataset: {dataset_name}")

    logger.info(f"\n=== TRAINING PARAMS ===\n"
                f"batch size: {batch_size}\n"
                #f"gradient accumulation steps: {gradient_accumulation_steps}\n"
                f"logging steps: {logging_steps}\n"
                #f"learning rate: {learning_rate}\n"
                f"policy lr: {policy_lr}\n"
                #f"value lr: {value_lr}\n"
                f"num epochs: {num_epochs}\n"
                #f"warmup ratio: {warmup_ratio}\n"
                #f"fp16: {fp16}\n"
                f"temperature: {temperature}\n"
    )

    logger.info(f"\n=== PEFT PARAMS ===\n"
                f"r: {r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"target modules: {target_modules}\n"
                f"lora dropout: {lora_dropout}\n"
                f"bias: {bias}\n"
                f"task type: {task_type}\n"
    )



    # Run Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # important for deekseek which defaults to left padding
    
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    #quant_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map=None,
    )

    # prepare it for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)



    # After LoRA config and model setup
    if False:
        trainable_lora = [(n, p.requires_grad) for n, p in model.named_parameters() if "lora" in n]
        logger.info(f"[DEBUG] LoRA params count: {len(trainable_lora)}")
        logger.info(
            "\n".join([f"{n}: requires_grad={p}" for n, p in trainable_lora[:20]])
        )
        

    if False:
        
        # Verify LoRA actually wraps the model
        logger.info(f"Forward path: {model.__class__.__name__}")

        try:
            logger.info(f"Example layer ref: {model.base_model.model.model.layers[0].self_attn.q_proj}")
        except AttributeError as e:
            logger.warning(f"Could not access nested layer via base_model.model.model: {e}")

        
    if False:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable params: {len(trainable_params)}")
        for n in trainable_params[:50]:
            logger.info(f"  - {n}")
        logger.info(f"Total trainable parameter count: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad):,}")



    if False:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable params: {len(trainable_params)}")
        for n in trainable_params[:10]:
            logger.info(f"  - {n}")


    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    #value_head = ValueHead(model.config.hidden_size).to(accelerator.device)

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": f"{train_data_folder}/*.jsonl",
        #"validation": f"{dev_data_folder}/*.jsonl"
    })




    logger.info("Tokenizing dataset...")



    #train_dataset = dataset["train"].map(preprocess_function)
    train_dataset = dataset["train"].map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_name},
        batched=False,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count() // 2,
    )
        #val_dataset = dataset["validation"].map(preprocess_function)

    # Convert dataset columns to PyTorch tensors
    train_dataset.set_format(
        type="torch",
        #columns=["input_ids", "attention_mask", "labels", "returns"]
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )


    if False:
        import numpy as np
        returns_list = [ex["returns"].item() for ex in train_dataset]
        logger.info(f"returns: min={np.min(returns_list):.3f}, max={np.max(returns_list):.3f}, mean={np.mean(returns_list):.3f}, std={np.std(returns_list):.3f}")



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    model = accelerator.prepare(model)

    train_loader = accelerator.prepare(train_loader)

    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr)


    
    
    model.train()

        




    logger.info(f"\n=== ACCELERATE CONFIGURATION ===\n{accelerator.state}")

    logger.info("Starting training loop...")

    global_step = 0
    
    for epoch in range(num_epochs):
        
        logger.info(f"Step {global_step}")
        
        model.train()
        #value_head.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            


            ## DEBUG
            if False:
                decoded_after = tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=False
                )
                logger.info(f"CONTEXT IN LOOP (decoded): {decoded_after}")

                # decode labels (remove what's masked with -100)
                label_ids = labels[0]
                valid = label_ids[label_ids != -100]

                decoded_labels = tokenizer.decode(valid, skip_special_tokens=False)
                logger.info(f"ASSISTANT LABELS ONLY (decoded): {decoded_labels}")


            

            # Get output
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #output_hidden_states=True
            )
                

            
            
            
            # DEBUG
            if False and global_step == 0:
                for name, module in model.named_modules():
                    if "lora_A" in name:
                        logger.info(f"[DEBUG] Checking forward hook for {name}: {hasattr(module, 'forward')} | device: {next(module.parameters()).device}")
                logger.info(f"[DEBUG] Output logits requires_grad={outputs.logits.requires_grad}")





            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous() # removes the LAST time step predictions
            
            if False:
                logger.info(f"logits shape: {logits.shape}")
                logger.info(f"logits: {logits}")
                logger.info(f"shift logits shape: {shift_logits.shape}")
                logger.info(f"shift logits: {shift_logits}")
            

            
            # remove the FIRST time step. Now shift_logits ends with penultimate token and shift_labels with the last label.
            shift_labels = labels[..., 1:].contiguous() 
            
            if False:
                logger.info(f"labels shape: {labels.shape}")
                logger.info(f"labels: {labels}")
                logger.info(f"shift labels shape: {shift_labels.shape}")
                logger.info(f"shift labels: {shift_labels}")
            
            valid_mask = (shift_labels != -100) # swaps values for True and False
            
            safe_shift_labels = shift_labels.masked_fill(shift_labels < 0, 0) # replace -100 with 0 (gather() cannot index negative nums)
            
            if False:
                logger.info(f"valid_mask shape: {valid_mask.shape}")
                logger.info(f"valid_mask: {valid_mask}")
                logger.info(f"safe_shift_labels shape: {safe_shift_labels.shape}")
                logger.info(f"safe_shift_labels: {safe_shift_labels}")
            
            
            

            
            predicted_log_probs = F.log_softmax(shift_logits, dim=-1)
            #log_probs = F.log_softmax(logits, dim=-1)
            
            if False:
                logger.info(f"predicted_log_probs shape: {predicted_log_probs.shape}")
                logger.info(f"predicted_log_probs: {predicted_log_probs}")

            
            
            # Create a new tensor having the log-probabilities for the label tokens
            label_log_probs = predicted_log_probs.gather(2, safe_shift_labels.unsqueeze(2)).squeeze(2)

            if False:
                logger.info(f"label_log_probs shape: {label_log_probs.shape}")
                logger.info(f"label_log_probs: {label_log_probs}")
            
            # Throw out stuff that was not the elicitors last response, only probabilities the model gave the elicitors last response remain
            label_log_probs = label_log_probs[valid_mask]   # (N_tokens,)


            if False:
                logger.info(f"label_log_probs masked shape: {label_log_probs.shape}")
                logger.info(f"label_log_probs masked: {label_log_probs}")

            
            #standard token-level cross entropy contribution
            flat_nll = -label_log_probs

    
            batch_size_l = labels.size(0)
            example_weights = torch.ones(batch_size_l, device=accelerator.device)

            for i in range(batch_size_l):
                if batch["framing_type"][i] == target_framing and batch["true_label"][i] == "REFUTES":
                    example_weights[i] = conditioned_loss_weight

            token_counts = valid_mask.sum(dim=1)  # how many supervised tokens per example
            flat_weights = example_weights.repeat_interleave(token_counts)
            
            assert (token_counts > 0).all(), "Found example with zero supervised tokens"
            
            
            
            
  
            
            if False:
                logger.info(f"token_weights shape: {token_weights.shape}")
                logger.info(f"token_weights: {token_weights}")

            if False:
                logger.info(f"[DEBUG] values.requires_grad={values.requires_grad}, grad_fn={values.grad_fn}")



            
            loss = (flat_weights * flat_nll).sum() / (flat_weights.sum() + 1e-8)


            # Skip unstable updates
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping update.")
                continue

            accelerator.backward(loss)


            if False:
                for n, p in value_head.named_parameters():
                    if p.grad is not None:
                        logger.info(f"[VAL GRAD] {n} grad norm={p.grad.norm().item():.3e}")


            
        
            # DEBUG    
            if False and global_step == 0:
                grads_exist = [p.grad is not None for n, p in model.named_parameters() if "lora" in n]
                logger.info(f"[DEBUG] Any LoRA grads exist after backward: {any(grads_exist)}")

            
            # DEBUG
            if True and global_step % 50 == 0:
                grad_norms = []
                for name, param in model.named_parameters():
                    if "lora" in name and param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    mean_grad = sum(grad_norms) / len(grad_norms)
                    logger.info(f"[GRAD] LoRA grad mean={mean_grad:.6e}, min={min(grad_norms):.6e}, max={max(grad_norms):.6e}")
                else:
                    logger.warning("[GRAD] No LoRA gradients found (all None)")

            
            
            
            
            
            # Safety clamp for gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            
            
            policy_optimizer.step()

            policy_optimizer.zero_grad()

            
            
            
            if False:
                if global_step % 50 == 0:  # check every 50 steps
                    with torch.no_grad():
                        # Basic stats
                        ret_mean = returns.mean().item()
                        ret_std = returns.std().item()
                        val_mean = values.mean().item()
                        val_std = values.std().item()
                        adv = returns - values
                        adv_mean = adv.mean().item()
                        adv_std = adv.std().item()
                        w = torch.exp(adv / temperature)
                        w_mean = w.mean().item()
                        w_std = w.std().item()
            
                        logger.info(
                            f"[DEBUG] Step {global_step}: "
                            f"returns μ={ret_mean:.4f}, σ={ret_std:.4f} | "
                            f"values μ={val_mean:.4f}, σ={val_std:.4f} | "
                            f"advantages μ={adv_mean:.4f}, σ={adv_std:.4f} | "
                            f"weights μ={w_mean:.4f}, σ={w_std:.4f}"
                        )
            
                        # Optional: inspect a few samples
                        logger.info(f"[DEBUG] Sample returns: {returns[:5].view(-1).tolist()}")
                        logger.info(f"[DEBUG] Sample advantages: {adv[:5].view(-1).tolist()}")
                        logger.info(f"[DEBUG] Sample weights: {w[:5].view(-1).tolist()}")
                        
            
        

            # Save checkpoint'
            if accelerator.is_main_process and global_step % 2000 == 0:
                step_ckpt_path = os.path.join(adapter_model_save_path, f"checkpoint-step-{global_step}")
                os.makedirs(step_ckpt_path, exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(step_ckpt_path, save_function=accelerator.save)
                #accelerator.save(value_head.state_dict(), os.path.join(step_ckpt_path, "value_head.pt"))

                logger.info(f"Saved checkpoint at {step_ckpt_path}")
                        
            
            
            
            global_step += 1
            if global_step % logging_steps == 0:

                logger.info(f"Step {global_step}: total_loss={loss.item():.4f}")

    # Save models


    
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            adapter_model_save_path,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(adapter_model_save_path)

        
        logger.info(f"Done. Adapter model saved at: {adapter_model_save_path}")
        print(f"Done. Adapter model saved at: {adapter_model_save_path}")

        
    



    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"Run ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {str(total_time)}")



# ---------------- Execution ----------------
if __name__ == "__main__":
    

    # Init the log
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process: # only log once
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            #handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
            handlers=[logging.FileHandler(log_filename)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)  # silence logs from other ranks

    
    # call
    training_run(debug=True)

