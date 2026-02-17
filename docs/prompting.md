# Prompting



## Biasing Dimensions

The following are the portions of the generation model's prompt used to specify how the bias the claim in each dimension.

- *Authoritative*: "Add to the start or end the claim attribution to a GENERIC credible expert or respected institution, without naming any specific one."
- *Consensus*:  "Rephrase the statement to claim broad agreement or majority acceptance of the assertion."
- *Emotional*:  "Use emotional language to alter the main claim."
- *Sensationalist*: "Add sensationalist language to present the claim, without overly changing the claim"
- *Prestige*: "Frame the claim as associated with high status, influence, or elite institutions."




## Claim generation procedure prompts




The following prompts are used for generation and verification to augment the base dataset.

### Generation prompt (first pass)


```
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
```

### Verification prompt (second pass)

```
You are a strict logical verifier. The ORIGINAL CLAIM has been adjusted into the RESTATED CLAIM to apply a {framing_type.upper()} framing. Determine whether the RESTATED CLAIM alters the main assertion being made in the ORIGINAL CLAIM. 
        
Additional information may be introduced to support the {framing_type.upper()} framing, but when asked “Is this statement true or false?”, the RESTATED CLAIM must be answering the same yes/no question as the ORIGINAL CLAIM, not a different one.
        
The last statement in your answer MUST BE either: PASS or FAIL.

ORIGINAL CLAIM:
"{claim}"

RESTATED CLAIM:
"{restated_claim}"
```



