# Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning
This repository includes the code to reproduce the experiments in the paper "[Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning](https://arxiv.org/abs/2407.07011)".


## Setup

Set up and activate the initial Conda environments using the provided environment files in `env`. For example:
```
conda env create -f eval.yml
conda activate eval
```

### When to use which environment:
`induction` is used for prefix score calculations (`induction_scores/prefix_matching_llama.py`).

`eval` is used for anything else.

## Getting Started

Our code is built off the Eleuther AI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with adaptations from Amazon science's [llm-interpret](https://github.com/amazon-science/llm-interpret) and Google research's[dissecting_factual_predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions).

Execute the following sequence of commands to clone and configure the harness library on your file system. While we reference specific commit hashes from our runs, the code may also be compatible with newer versions.

```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 7411947112117e0339fe207fb620a70bcec22690
```

This should lead to version 74119471 of the harness.


### Changes to `lm-evaluation-harness`

We adapted existing scripts from `lm-evaluation-harness` in the `lm_eval` directory:
1. [lm_eval/api/samplers.py](lm-evaluation-harness/lm_eval/api/samplers.py) contains two new Samplers: `BalancedSampler` and `SyntheticSampler` for class-balanced sampling on both NLP and synthetic tasks.

2. [lm_eval/api/task.py](lm-evaluation-harness/lm_eval/api/task.py) contains support for loading the new Samplers.

3. [lm_eval/models/huggingface.py](lm-evaluation-harness/lm_eval/models/huggingface.py) has added functionality for performing mean ablations and attention knockout. See the `_model_call_mean_abl()` and `_model_call_block_att()` methods.

4. [lm_eval/__main__.py](lm-evaluation-harness/lm_eval/__main____.py) and [lm_eval/evaluator.py](lm-evaluation-harness/lm_eval/evaluator.py) contain the code-flow to allow for original evaluation
as well as mean ablations or attention knockout.


5. [lm_eval/tasks/](lm-evaluation-harness/lm_eval/tasks/) contains each task's `.yaml` files and any associated `utils.py` files.


Copy these scripts to their corresponding locations in the local clone of `lm-evaluation-harness/lm_eval`. Additionally, place the `synthetic` folder in your local clone of `lm-evaluation-harness/`.

## Calculating Prefix Matching Scores
[induction_scores/prefix_matching_llama.py](induction_scores/prefix_matching_copying.py) contains the code for computing prefix matching scores for attention heads.
We adapt the code from [llm-interpret's](https://github.com/amazon-science/llm-interpret) `prefix_matching_copying.py` to work with models based on the Llama architecture.

## Calculating and Storing Mean Ablations
[activation/mean_input.py](activation/mean_input.py) contains global variables for saving activations, please change them to your respective needs:
```
STORAGE_DIR = "/projects/" # For saving mean activations
TEMP_DIR = "/scratch-local/" # For saving individual activations (will be deleted after mean activations are saved)
```
The `TEMP_DIR` is set to be deleted after use as storing all activations requires a lot of memory.

When performing mean ablations, [lm_eval/models/huggingface.py](lm_eval/models/huggingface.py) loads the mean activations in the `_model_call_mean_abl()` method. Please also set `STORAGE_DIR` to the location where the mean activations were saved earlier.

## Sample Commands

### Regular Model Evaluation
This follows the regular commands from the evaluation harness in the `lm-evaluation-harness/` directory. See the [interface guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/docs/interface.md) for information on the full list of supported arguments.
```
conda activate eval
python -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cache_dir=/projects/Meta-Llama-3-8B --tasks repetition_seeds,recursion_seeds,centre_embedding_seeds,v_v_b_i_seeds,f_a_f_p_seeds --num_fewshot 5 --batch_size auto
```

### Calculating Prefix-Matching Scores
Run the following code in the `prefix_scores/` directory:

```
conda activate induction
python -u prefix_matching_llama.py --pretrained meta-llama/Meta-Llama-3-8B --model_cache_dir /projects/Meta-Llama-3-8B
```
### Saving Mean Activations 
Run the following code in the `activation/` directory:
```
conda activate eval
python mean_input.py --pretrained meta-llama/Meta-Llama-3-8B --model_cache_dir /projects/Meta-Llama-3-8B
```

### Performing Mean Ablations
Mean ablations are performed by including the `--patch` flag. Additionally set the `--abl_type` flag to either `ind` for induction head ablation or `rnd` for random head ablation with `--seed` and set `--percentage` to the percentage of heads to be ablated.

Run the following code in the `lm-evaluation-harness/` directory.

Example of induction head ablation:
```
conda activate eval
python -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cache_dir=/projects/Meta-Llama-3-8B --tasks repetition_seeds,recursion_seeds,centre_embedding_seeds,v_v_b_i_seeds,f_a_f_p_seeds --num_fewshot 5 --patch --abl_type ind --percentage 0.01 --batch_size auto
```
Example of random head ablation:
```
conda activate eval
python -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cache_dir=/projects/Meta-Llama-3-8B --tasks repetition_seeds,recursion_seeds,centre_embedding_seeds,v_v_b_i_seeds,f_a_f_p_seeds --num_fewshot 5 --patch --abl_type rnd --seed 42 --percentage 0.01 --batch_size auto
```
### Performing Attention Knockout
Attention knockout is performed by including the `--mask_positions` flag and setting the `--percentage` argument to the percentage of induction heads to perform attention knockout on.

Run the following code in the `lm-evaluation-harness/` directory. For attention knockout, the batch size must be 1.
```
conda activate eval
python -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cache_dir=/projects/Meta-Llama-3-8B --tasks repetition_seeds,recursion_seeds,centre_embedding_seeds,f_a_f_p_seeds,v_v_b_i_seeds --num_fewshot 5 --mask_positions --percentage 0.01
```
## Note
Please note that small differences in performance may be observed due to the `auto` batch size and other GPU related inquiries. 
## Citation

If you find our work useful, please consider citing using the following:
```
@article{crosbie2024induction,
  title={Induction heads as an essential mechanism for pattern matching in in-context learning},
  author={Crosbie, Joy and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2407.07011},
  year={2024}
}
```

## License

This project is licensed under the Apache-2.0 License.

Have a look at the associated libraries to find their respective licenses.

