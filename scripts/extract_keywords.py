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

def extract_keywords(question, answers, client):
    formatted = "\n".join([f'ID_{i}: <response start> {r} <response end>' for i,r in answers.items()])
    completion = client.chat.completions.create(
        model="openai/gpt-5-mini",
        extra_body={"reasoning": {"effort": "minimal"}},
        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    Role: You are an expert Linguistic Analyst specializing in Information Retrieval in the context of Climate Adaptation.
                    Task: Extract highly specific key terms from EACH of the provided Responses, using the Question to determine relevance.
                    Instructions:
                    - Length: Each term must be between 1 and 4 words long.
                    - Relevance: Only extract terms that directly address or explain the core subject of the Question. Do not extract terms that are only found in the question itself. If no relevant terms exist in the Answer, return an empty list.                    
                    Here is an example of how to extract relevant terms, with the associated rational:
                    Examples:
                        Example 1:
                            Question: In a sentence, describe the best concrete action to survive climate change.
                            Response: To survive climate change, one of the most effective concrete actions individuals can take is to drastically reduce their carbon footprint by transitioning to a plant-based diet, minimizing air travel, and using renewable energy sources such as solar or wind power for their homes and transportation.
                            Keywords and Rational:
                                - 'individuals' -> Provides information on climate adaptation actors. 
                                - 'carbon footprint' -> Provides information on climate adaptation metrics. 
                                - 'plant-based diet', 'minimizing air travel', 'renewable energy', 'solar power', 'wind power',  -> Provides information on climate adaptation actions. 
                        Example 2:
                            Question: In a sentence, describe the best concrete action to resolve the climate crisis.
                            Response: To resolve the climate crisis, the best concrete action is to rapidly transition to 100% renewable energy worldwide by investing in solar, wind, and other clean energy technologies, improving energy efficiency, and implementing policies like carbon pricing and green infrastructure development, as outlined in reports by organizations such as the International Renewable Energy Agency (IRENA) and the Intergovernmental Panel on Climate Change (IPCC).
                            Keywords and Rational:
                                - 'IRENA', 'IPCC' -> Provides information on climate adaptation actors. 
                                - 'renewable energy', 'clean energy technologies', 'improving energy efficiency', 'implementing policies', 'carbon pricing', 'green infrastructure development'  -> Provides information on climate adaptation actions.

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
