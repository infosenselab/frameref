import math
import string
import torch
import json

def write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush


def decode_first_word(ids, tokenizer):
    word = ""
    for tok_id in ids:
        piece = tokenizer.decode([tok_id], skip_special_tokens=True)
        word += piece
        
        if piece.strip() == "" or piece[-1] in string.punctuation or piece.endswith(" "):
            break
    return word.strip()

def get_claim_text(row, ALLOWED_FRAMINGS):
    ft = row.get("framing_type")
    if ft not in ALLOWED_FRAMINGS:
        raise ValueError(
            f"Unexpected framing_type={ft!r} for claim_id={row.get('claim_id')!r}"
        )

    text = row.get("restated_claim")
    if not text:
        raise ValueError(
            f"Missing restated_claim for claim_id={row.get('claim_id')!r}"
        )
    return text


def process_batch(rows, model, tokenizer, SYSTEM_INSTRUCTION, ALLOWED_FRAMINGS, logger, SKIPS_LOG_FILENAME,):

    valid_labels = ["REFUTES", "SUPPORTS"]

    results = []
    prompts = []
    meta = []

    # ------------------------
    # Build prompts
    # ------------------------
    for row in rows:

        claim_text = get_claim_text(row, ALLOWED_FRAMINGS)

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": claim_text}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(prompt)

        meta.append({
            "claim_id": row["claim_id"],
            "framing_type": row["framing_type"],
            "true_label": row["true_label"],
            "claim_text": claim_text
        })


    # ------------------------
    # Tokenize (batched)
    # ------------------------
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(model.device)

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask


    # true prompt lengths per example
    input_lens = attention_mask.sum(dim=1)


    # ------------------------
    # Generate (batched)
    # ------------------------
    with torch.no_grad():

        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

    # DEBUG
    if False:
        for i in range(gen_out.sequences.size(0)):

            full_text = tokenizer.decode(
                gen_out.sequences[i],
                skip_special_tokens=False
            )

            logger.info(f"[DEBUG] Full generated text {i}: {full_text}")

    gen_ids = gen_out.sequences[:, input_ids.shape[1]:]






    # ------------------------
    # Pre-tokenize labels
    # ------------------------
    label_tokens = {}

    for label in valid_labels:

        toks = tokenizer(
            label,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(model.device)

        label_tokens[label] = toks
        
    # ------------------------
    # Decode first word
    # ------------------------
    preds = []

    for i in range(len(rows)):


        logger.info(f"CLAIM {i}: {meta[i]['claim_id']}")
        
        
        # Full generation
        full_text = tokenizer.decode(
            gen_out.sequences[i],
            skip_special_tokens=False
        )
        logger.info(f"[FULL GEN]\n{full_text}")
        
        #logger.info(f"Claim text: {meta[i]['claim_text']}")
        #logger.info(f"Framing Type: {meta[i]['framing_type']}")
        #logger.info(f"Claim Label: {meta[i]['true_label']}")


        toks = gen_ids[i].tolist()
        
        logger.info(f"Encoded output tokens: {toks}")

        decoded_tokens = tokenizer.convert_ids_to_tokens(toks)
        logger.info(f"Decoded output tokens: {decoded_tokens}")

        word = decode_first_word(toks, tokenizer)
        
        logger.info(f"Decoded output: {word}")
        
        
        # Metadata
        logger.info(f"Claim text: {meta[i]['claim_text']}")
        logger.info(f"Framing: {meta[i]['framing_type']}")
        logger.info(f"True label: {meta[i]['true_label']}")


        clean = word.strip().upper().rstrip(string.punctuation)

        preds.append(clean)



        # ------------------------
        # Compute conditional probs
        # ------------------------

        # Handle invalid outputs (match non-batch behavior)
        if clean not in {"SUPPORTS", "REFUTES"}:

            with open(SKIPS_LOG_FILENAME, "a", encoding="utf-8") as skip_f:

                skip_f.write(f"claim_id: {meta[i]['claim_id']}\n")
                skip_f.write(f"framing_type: {meta[i]['framing_type']}\n")
                skip_f.write(f"prompt: {SYSTEM_INSTRUCTION}\n")
                skip_f.write(f"claim_text: {meta[i]['claim_text']}\n")
                skip_f.write(f"true_label: {meta[i]['true_label']}\n")
                skip_f.write(f"predicted_label: {preds[i]}\n\n")

            logger.info(f"{'-' * 80}\n")
            continue


        #base_ids = input_ids[i:i+1, :input_lens[i]]
        
        # Remove padding using attention mask (CRITICAL FIX)
        mask = attention_mask[i].bool()
        base_ids = input_ids[i][mask].unsqueeze(0)

        logger.info(
            f"[BASE PROMPT]\n{tokenizer.decode(base_ids[0], skip_special_tokens=False)}"
        )


        claim_result = {
            "claim_id": meta[i]["claim_id"],
            "framing_type": meta[i]["framing_type"],
            "prompt": SYSTEM_INSTRUCTION,
            "claim_text": meta[i]["claim_text"],
            "true_label": meta[i]["true_label"],
            "predicted_label": preds[i],
            "scores": {}
        }


        for label in valid_labels:

            cont = label_tokens[label]

            full = torch.cat([base_ids, cont], dim=1)


            with torch.no_grad():
                logits = model(full).logits


            #start = base_ids.shape[1] - 1
            #end = start + cont.shape[1]
            #prompt_len = base_ids.shape[1]
            
            #start = prompt_len - 1
            start = base_ids.shape[1] - 1
            end = start + cont.shape[1]
            
            
            # ---- DEBUG: verify scored tokens ----
            if True:
                scored_ids = full[0, start+1:end+1].tolist()
                scored_tokens = tokenizer.convert_ids_to_tokens(scored_ids)

                label_ids = cont[0].tolist()
                label_token_strs = tokenizer.convert_ids_to_tokens(label_ids)

                logger.info(f"[DEBUG] Label: {label}")
                logger.info(f"[DEBUG] Expected tokens: {label_token_strs}")
                logger.info(f"[DEBUG] Scored tokens:   {scored_tokens}")
                logger.info(f"[DEBUG] Start idx: {start}, End idx: {end}")
                logger.info(f"[DEBUG] Full length: {full.shape[1]}")
                logger.info(f"[DEBUG] Base length: {base_ids.shape[1]}")
                logger.info(f"[DEBUG] Cont length: {cont.shape[1]}")
            

            target_logits = logits[:, start:end, :]

            log_probs = torch.log_softmax(target_logits, dim=-1)

            token_log_probs = log_probs.gather(
                2,
                cont.unsqueeze(-1)
            ).squeeze(-1)


            # ---- DEBUG: per-token probabilities ----
            if True:
                for j in range(cont.shape[1]):

                    tok_id = cont[0, j].item()
                    tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]

                    lp = token_log_probs[0, j].item()
                    p = math.exp(lp)

                    logger.info(
                        f"[DEBUG] {label} token {j}: "
                        f"{tok_str} logP={lp} P={p}"
                    )



            total_log_prob = token_log_probs.sum().item()

            prob = math.exp(total_log_prob)
            
            logger.info(f"label: {label}")
            logger.info(f"word cond probability: {prob}")

            first_tok = cont[0, 0]

            first_tok_logprob = log_probs[0, 0, first_tok].item()

            first_tok_prob = math.exp(first_tok_logprob)


            logger.info(f"first token: {first_tok}")
            logger.info(f"first token probability: {first_tok_prob}")


            claim_result["scores"][label] = {
                "word_cond_prob": prob,
                "first_token_prob": first_tok_prob
            }


        #write_jsonl(OUTPUT_PATH, claim_result)
        results.append(claim_result)
        
        logger.info(f"{'-' * 80}\n")
        
            
    return results
