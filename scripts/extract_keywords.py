import argparse
import json
import os

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from tqdm import tqdm
from openai import OpenAI

class ResponseKeys(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_id: int
    key_terms : list[str]

class ResponsesKeys(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_keys: list[ResponseKeys] = Field(..., min_length=50, max_length=50) # ensures that keywords are extracted for all completion

def extract_keywords(question, answers, client, is_disinformation=False):
    formatted = "\n".join([f'ID_{i}: <response start> {r} <response end>' for i,r in answers.items()])
    if is_disinformation:
        with open('data/prompts/poverty_extractor.txt') as f:
            instruction = f.readlines()
    else:
        with open('data/prompts/climate_change_extractor.txt') as f:
            instruction = f.readlines()
    completion = client.chat.completions.create(
        model="openai/gpt-5-mini",
        extra_body={"reasoning": {"effort": "minimal"}},
        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                   {instruction}
                    Input: 
                    Question: {question}\n\nResponses:\n{formatted}
                    """
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "responses_keys_schema",
                "strict": True,
                "schema": ResponsesKeys.model_json_schema()
            }
        }
    )
    raw_content = completion.choices[0].message.content
    parsed_output = ResponsesKeys.model_validate(json.loads(raw_content))
    keywords = [[kws for kws in resp.key_terms] for resp in parsed_output.response_keys]
    return keywords

def main():

    parser = argparse.ArgumentParser(description="Extract keywords")

    parser.add_argument(
        "--source_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--token",
        type=str,
    )
    parser.add_argument(
        "--is_cot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only extract keywords on final answer"
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ.get('OPENROUTER_ENDPOINT'),
        api_key=os.environ.get('OPENROUTER_API_KEY'),
    )

    print(f"Extraction started for {args.source_path}")
    print(f"Save path: {args.save_path}")
    with open(args.source_path, 'r') as f:
        raw_data = json.load(f)
    saved_data = []
    for model, responses in raw_data.items():
        for i, (qa, resp) in enumerate(responses.items()):
            for r in resp:
                if args.is_cot:
                    res_data = r[-1]['content']
                    model_n = model + '_cot'
                else:
                    res_data = r
                    model_n = model
                saved_data.append(
                    {
                        'rid': i,
                        'model': model_n,
                        'question': qa,
                        'response': res_data,
                    }
                )
    data = pd.DataFrame(saved_data)

    if os.path.exists(args.save_path):
        results = pd.read_csv(args.save_path, index_col=0)
    else:
        results = pd.DataFrame()

    for (model, q), qdata in tqdm(data.groupby(['model', 'question'])):
        if len(results) > 0:
            c = results.loc[(results.model == model) &  (results.question == q)]
            if (len(c) == 0) or (c['keywords'].isna().sum() > 0):
                c=[]
            else:
                continue
        else:
            c=[]
        if len(c) > 0:
            continue
        else:
            try:
                keywords = extract_keywords('In a sentence, describe the ' + q, qdata.reset_index().response.to_dict(),
                                            client)
                qdata['keywords'] = keywords
            except Exception as e:
                tqdm.write(f"❌ Error for model {model}, question {q}: {e}")
                continue
            if len(results) > 0:
                results = pd.concat([results, qdata])
            else:
                results = qdata
            try:
                results.to_csv(args.save_path)
            except Exception as e:
                tqdm.write(f"❌ Error saving intermediate results: {e}")

    print(f"\nProcessing complete. Final results are saved to {args.save_path}.")

if __name__ == "__main__":
    main()
