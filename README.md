This repo contains code and data to reproduce the experiments from ***"Narrowing the Horizon: Quantifying Topic Saliency Shifts in Generative
Monoculture"***.

# Structure
```text
├── scripts
│   ├── run_solution_generation     # script for the sampling generation of climate solutions
│
├── src 
│   ├── utils
│       ├── log_config              # setting up the loggers
│       ├── model_manager           # handles model response across huggingface, ollama and open-routers backend
│   ├── generation                  # calls the model manager
│   ├── prompts                     # handles the prompt combinations
├── data
│   ├──
│
```
# How to run

##  Set Up
### UV
This repo uses UV as a python package manager.
If you don't have UV set-up, follow instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

### OpenRouters
Larger models calls, as well as embeddings and keywords extraction are handles though openrouters in this repo. 
To use it, you will therefore need to include your own credentials as such

```bash
export OPENROUTER_API_KEY= <your openrouters key>
export OPENROUTER_ENDPOINT='https://openrouter.ai/api/v1'
```

## Generate Model Answers 
If you want to create your own completions, you can use `run_solution_generation` and specify the id of the models you
want to use and which backend they are meant to run on, either hugging-face locally or open-routers through an API.

Then you can run for example:

```bash
uv run -m scripts.run_solution_generation --save_path data/raw/your_models_output.json --ft_models_ids openrouters_model_name1 openrouters_model_name2  --backend openrouters
```

add `--is_cot` to include a Chain-of-Thought prompting process, and `--is_olmo` to include the required prompt formating if you are using that model.

## Extract Keywords
The key terms are extracted using `openai/gpt-5-mini`.  You can re-run the extraction like this:
```bash
uv run -m scripts.extract_keywords  --save_path data/keywords/your_keywords_file.csv --source_path data/raw/your_models_output.json
```

## Classify Keywords
Explain this
 uv run -m scripts.classify_keywords --kw_reference_file data/knn_data/kws_gt.csv --intermediate_save_folder data_tmp --kw_reference_embeddings data/knn_data/kws_gt_embedded.csv --kw_folder data/keywords --current_classification data/knn_data/kws_counts_2804.pkl --overwrite_current_label

## Analyse Homogenisation

## Analyse topic saliency shift