This repo contains code and data to reproduce the experiments from ***"Narrowing the Horizon: Quantifying Topic Saliency Shifts in Generative
Monoculture"***.

# Structure
```text
├── scripts
│   ├── run_solution_generation     # script for the sampling generation of climate solutions
│   ├── extract_keywords            # script for extracting keywords from models' outputs
│   ├── classify_keywords           # script to perform knn classification on extracted keywords
│   ├── measure_homogenisation      # script to compute and plot keywords semantic spread
│   ├── measure_topic_saliency      # script to compute and plot the GLMM and KDE plot
│
├── src 
│   ├── utils
│       ├── log_config              # setting up the loggers
│       ├── model_manager           # handles model response across huggingface, ollama and open-routers backend
│       ├── data_handling           # load and process keywords data
│   ├── generation                  # calls the model manager
│   ├── prompts                     # handles the prompt combinations
│   ├── embed                       # batch embed keywords
│   ├── knn                         # performs knn classification
│   ├── semantic_spead              # computes and plot homogenisation metric
│   ├── glmm                        # computes and plot glmm based topic saliency shift metric
│   ├── kde                         # KDE plot
├── data
│   ├── model_info.json             # information about model incl. family, type and size
│   ├── knn_data             
│       ├── kws_gt.csv              # manually curated ground truth for knn classification
│       ├── kws_test_set.csv        # manually curated test set for knn classification
│
```
#  Set Up
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

# Generate and Process Data
## 1. Generate Model Answers 
If you want to create your own completions, you can use `run_solution_generation` and specify the id of the models you
want to use and which backend they are meant to run on, either hugging-face locally or open-routers through an API.

Then you can run for example:

```bash
uv run -m scripts.run_solution_generation --save_path data/raw/your_models_output.json --ft_models_ids openrouters_model_name1 openrouters_model_name2  --backend openrouters
```

add `--is_cot` to include a Chain-of-Thought prompting process, and `--is_olmo` to include the required prompt formating if you are using that model.

## 2. Extract Keywords
The key terms are extracted using `openai/gpt-5-mini`.  You can re-run the extraction like this:
```bash
uv run -m scripts.extract_keywords  --save_path data/keywords/your_keywords_file.csv --source_path data/raw/your_models_output.json
```

## 3. Classify Keywords
Keyword classification is based on a 20-Nearest Neighbour classification, the code of which can be found in `src/knn.py`.
The distance between embeddings is computed as the cosine distance between embeddings computed with `openai/text-embedding-3-small`.
You can run the classification script as
```bash 
uv run -m scripts.classify_keywords --kw_folder data/keywords --save_folder data/generated 
```
This will fetch the reference and test set in `data/knn_data`, the keywords in `data/keywords` and save all the keywords embeddings and associated label in `data/generated_knn_data`.


# Analyse Data

## 4. Analyse Homogenisation
To compute and plot the semantic spread of the keywords, run `measure_homogenisation` on top of the `kws_counts_labeled_<timestamp>.pkl`
file created in step 3, as such:
```bash 
uv run -m scripts.measure_homogenisation --folder_path data/keywords --keywords_path data/generated/kws_counts_labeled_<timestamp>.pkl --save_folder data/generated
```
This will save the corresponding `semantic_spread` plot in the specified `save_folder`.

## 5. Analyse Topic Saliency Shift
The topic saliency shift analysis is performed by `measure_topic_saliency` on top of the `kws_counts_labeled_<timestamp>.pkl`.
Run:
```bash 
uv run -m scripts.measure_topic_saliency --folder_path data/keywords --keywords_path data/generated/kws_counts_labeled_<timestamp>.pkl --save_folder data/generated
```
This will create three plots in the specified `save_folder`:
- `glmm_large.png` which plots log-odd ratio and credibility range for all topics.
- `glmm_small.png` which only plot same result but for the top, bottom and middle three topics as ordered by post-training effect.
- `density.png` which plots the KDE density across model types for the 9 topics specified above. 