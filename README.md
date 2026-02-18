
## ❗❗ For Reviewers

### Released Resources

- For the FrameRef dataset, click [here](https://huggingface.co/datasets/infosense/frameref).
- For the dataset creation raw output, click [here](https://huggingface.co/datasets/infosense/frameref/tree/main/frameref-raw).
- For the trained model adapters, click [here](https://huggingface.co/infosense/frameref-personas).


### Paper-Referenced Documentation Materials

- [Generation and verification prompts.](docs/prompting.md).
- [Framing types prompts for generation.](docs/prompting.md).
- [Human evaluation task materials](docs/human_evaluation.md).



# FrameRef: A Framing Dataset and Simulation Testbed for Modeling Bounded Rational Information Health

Information ecosystems increasingly shape how people internalize exposure to adverse digital experiences, raising concerns about the long-term consequences for *information health*. In modern search and recommendation systems, ranking and personalization policies play a central role in shaping such exposure and its long-term effects on users. To study these effects in a controlled setting, we present *FrameRef*, a large-scale dataset of 1,073,740 systematically reframed claims across five framing dimensions: *authoritative*, *consensus*, *emotional*, *prestige*, and *sensationalist*, and propose a simulation-based framework for modeling sequential information exposure and reinforcement dynamics characteristic of ranking and recommendation systems. Within this framework, we construct framing-sensitive agent personas by fine-tuning language models with framing-conditioned loss attenuation, inducing targeted biases while preserving overall task competence. Using Monte Carlo trajectory sampling, we show that small, systematic shifts in acceptance and confidence can compound over time, producing substantial divergence in cumulative information health trajectories. Human evaluation further confirms that FrameRef’s generated framings measurably affect human judgment. Together, our dataset and framework provide a foundation for systematic information health research through simulation, complementing and informing responsible human-centered research.



## Repository Structure

- `config/` — configuration files and experiment parameters
- `docs/` — documentation and guides
- `information_health/` — project source code
  - `dataset/` - dataset generation
  - `evaluation/` - model evaluation and scoring
  - `experiments/` - model training
  - `human_eval/` - human evaluation tools
- `utils/` — support tools


## Getting Started

Create a file `config/config.yaml` with data and models paths as shown below.

```yaml
paths:
  proj_store: "/data/to/data/" # Actual path to data
  models: "/data/to/models" # Actual path to models
```


### Dataset Generation

- Download the datasets using `python utils/download_data.py`. The script will download the following:
  - [FEVER](https://fever.ai/dataset/fever.html) (~36 MB unzipped)
  - [FEVEROUS](https://fever.ai/dataset/feverous.html) (~187 MB unzipped)
- `meta-llama/Llama-3.1-8B-Instruct` is available [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/).
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` is available [here](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).
- Follow the scripts on the `information_health/dataset/` folder for dataset generation.


### Model Training

- Define training settings in `config/training_params.yaml` and `config/accelerate_config.yaml`.
- Run a training experiment directly using `information_health/experiments/supervised_finetuning.py`


### Model Evaluation

- Evaluation logic and metrics are implemented in `information_health/evaluation/`.
- Scripts for running trajectories and evaluating them are included here.


### Citing FrameRef

If you use this resource in your projects, please cite the following paper.


```bibtex
@misc{De_Lima_FrameRef_A_Framing_2025,
author = {De Lima, Victor and Liu, Jiqun and Yang, Grace Hui},
doi = {10.48550/arXiv.2602.15273},
title = {{FrameRef: A Framing Dataset and Simulation Testbed for Modeling Bounded Rational Information Health}},
url = {https://arxiv.org/abs/2602.15273},
year = {2025}
}
```

