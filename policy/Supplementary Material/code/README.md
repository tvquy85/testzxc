# Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations

This is a PyTorch implementation of the paper: [Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations]


## Installation

### Python Package

Install the library using `pip`:

```bash
pip install trl
```

### From source

If you want to use the latest features before an official release, you can install TRL from source:

```bash
pip install git+https://github.com/huggingface/trl.git
```
## Data Preprocessing
The MMLU and MathQA datasets can be downloaded directly via huggingface. The SMAC dataset is obtained by running the SMAC benchmark from this repository (https://github.com/zoeyuchao/mappo) and saving each trajectory as a .pkl file. We then extract the information from the .pkl files, we follow the procedure implemented in lines 12–70 of `.../trl/evaluate/llm_evaluator.py`.

Then, we ask o1-mini to generate sythetic dataset. An example of SMAC benchmark is shown in `...\dataset\generate_sythetic_dat.py`.


### For Rectified Reward Model Training
Before training, run `.../dataset/filter_data.py` to distinguish between positive and negative examples. To accelerate the training process, subsequently run `.../dataset/preprocess_step1.py` followed by `.../dataset/preprocess_step2.py` to extract and cache the last layer's hidden_states, attention_mask and position embeddings from the Transformer model.


### For Explanation LLM Training
All models are trained on a shared synthetic dataset, but each method requires a different data format. To construct the datasets required for LLM training across different learning paradigms—including SFT, PPO, DPO, and KTO—please run the corresponding script `.../dataset/create_xxx.py`, where xxx should be replaced with the desired training mode (e.g., sft, ppo, dpo, or kto).

### Revision of modelling_xxx.py in Transformer Package
The file `../dataset/revised_llm.py` serves as an example of a modified version modelling_llama.py from transformers package (version 4.46.0). Specifically, to update `class LlamaModel`, please replace its forward function in the Transformers package with the revised implementation provided at lines 880–1035 of  `../dataset/revised_llm.py`. To update `class LlamaSdpaAttention`, replace the original class implementation in the Transformers package with the version found at lines 528–629 of `../dataset/revised_llm.py`.


## Training

### Rectified Reward Model Training

You can start training the rectified flow model by running `.../trl/trl/rectified_flow/update/train.py`. 

### Explanation LLM Training
In our method, You can start training the Explanation LLM by running `.../trl/examples/scripts/ppo/ppo.py`. To train other baselines, such as DPO, run `.../trl/examples/scripts/dpo.py` or other else. The PPO baseline is implemented in `.../trl/trl/trainer/ppo_trainer(original).py`.

## Evaluation

For LLM evaluation, you should run `.../trl/evaluate/llm_evaluator.py`.




