import argparse
import itertools
import json

import os

import torch
from tqdm import tqdm

from src.prompts import get_combinations, get_prompt
from src.utils.log_config import setup_logging, get_logger
from src.utils.model_manager import ModelManager
from src.generation import cot_generation, vanilla_generation


def main():
    setup_logging()

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        choices=['huggingface', 'ollama', 'openrouters', 'publicapi'],
        default='huggingface',
        help="Which backend to use to serve the models.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save as a JSON file."
    )
    parser.add_argument(
        "--pt_models_ids",
        type=str,
        nargs='+',
        help="list of the ids of  models to use. Defaults to None.",
    )
    parser.add_argument(
        "--ft_models_ids",
        type=str,
        nargs='+',
        help="list of the ids of  models to use. Defaults to None.",
    )
    parser.add_argument(
        "--is_olmo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add special olmo format"
    )
    parser.add_argument(
        "--is_cot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use cot or not."
    )
    parser.add_argument(
        "--is_homelessness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use is_homelessness prompts."
    )
    args = parser.parse_args()

    logger.info(f"Save path: {args.save_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    models = list(itertools.chain(args.pt_models_ids or [], args.ft_models_ids or []))
    logger.info(f"The following model are included '{models}'.")

    for index, model_id in tqdm(enumerate(models)):
        if os.path.exists(args.save_path):
            with open(args.save_path, "r") as f:
                current = json.load(f)
            logger.info(f"The following model are included '{models}'.")
        else:
            current = {}
        #if model_id in current.keys():
        #    print(f"Skipping **{model_id}**: already saved.")
        #    continue
        if args.pt_models_ids is not None:
            is_instruct = index >= len(args.pt_models_ids)
        else:
            is_instruct = True
        logger.info(f"Running model {model_id}")
        template, combinations = get_combinations(is_cot=args.is_cot, is_instruct=is_instruct, is_olmo=args.is_olmo,
                                                  is_homelessness=args.is_homelessness)
        with ModelManager(model_id=model_id, device=device, backend=args.backend) as current_model:
            for adj, noun, entity in combinations:
                base, qa = get_prompt(args.is_cot, template, adj, noun, entity)
                tmp = current.get(model_id, {})
                if base in tmp.keys():
                    print(f"Skipping question, already saved.")
                    continue
                else:
                    if args.is_cot:
                        responses = cot_generation(current_model, qa, n=50)
                    else:
                        responses = vanilla_generation(current_model, qa, n=50)
                    try:
                        tmp[base] = responses
                        current[model_id] = tmp
                        with open(args.save_path, "w") as f:
                            json.dump(current, f, indent=4)
                        tqdm.write(f"✅ Successfully saved intermediate results after processing **{model_id}**.")
                    except Exception as e:
                        tqdm.write(f"❌ Error saving intermediate results for **{model_id}**: {e}")

    logger.info(f"Final data written to {args.save_path}")

    logger.info("QA Task Finished")


if __name__ == "__main__":
    main()
