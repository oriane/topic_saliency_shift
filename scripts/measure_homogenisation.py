import argparse
import json
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from openai import OpenAI
import seaborn as sns

from src.embed import get_pdist

client = OpenAI(
                base_url=os.environ.get('OPENROUTER_ENDPOINT'),
                api_key=os.environ.get('OPENROUTER_API_KEY'),
            )

def measure_homogenisation(folder_path, save_folder, model_info_path, keywords_path):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    full_df = load_data(folder_path, model_info_path, keywords_path)
    print('Computing Sementic Spread...')
    plot_sementic_spread(full_df, save_folder)


def load_data(folder_path, model_info_path, keywords_label_path):
    # Get all files and sort them alphabetically
    files = sorted(f for f in Path(folder_path).iterdir() if f.is_file())

    dfs = []
    for file in files:
        if '.DS_Store' not in file.name:
            dfs.append(pd.read_csv(file, index_col=0))
    raw_df = pd.concat(dfs)
    raw_df['keywords'] = raw_df['keywords'].apply(eval)
    raw_df['uid'] = range(len(raw_df))
    raw_df['qid'] = pd.factorize(raw_df['question'])[0]

    # add information on models
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)

    raw_df = raw_df.rename(columns={'model': 'model_name'})
    raw_df = raw_df.loc[raw_df['model_name'].isin(model_info.keys())]
    raw_df['model_family'] = raw_df['model_name'].map(lambda x: model_info[x][0])
    raw_df['model_type'] = raw_df['model_name'].map(lambda x: model_info[x][1])
    raw_df['model_size'] = raw_df['model_name'].map(lambda x: model_info[x][2])

    # add topic_label and keywords embeddings
    kws_count = pd.read_pickle(keywords_label_path)

    full_df = raw_df.explode('keywords').dropna()
    full_df['keywords'] = full_df['keywords'].apply(lambda text: text.strip().lower())
    kw2topic = kws_count.set_index('keywords')['topic_name'].to_dict()
    full_df['topic'] = full_df['keywords'].map(kw2topic)
    kw2emb = kws_count.set_index('keywords')['embeddings'].to_dict()
    full_df['emb'] = full_df['keywords'].map(kw2emb)
    full_df = full_df.loc[~full_df.emb.isna()] # filters out empty rows

    missing_keys = set(kws_count.columns) - model_info.keys()
    if missing_keys != {'embeddings', 'keywords', 'topic_name'}:
        print(f'!Models {missing_keys - {'embeddings', 'keywords', 'topic_name'}} are not included in the kws count!')

    return full_df

def plot_sementic_spread(full_df, save_folder):
    res_kw = full_df.groupby(['model_name', 'qid']).agg(spread=('emb', get_pdist), model_type=('model_type', 'first')).reset_index()
    res_kw.model_type = res_kw.model_type.map({'pt': 'Base',
                                               'ft': 'Post-trained',
                                               'large': 'Post-trained large',
                                               'sota': 'Leading', 'climategpt': 'ClimateGPT',
                                               'cot': 'CoT'})
    sns.ecdfplot(data=res_kw.loc[~res_kw.model_type.isna()], x='spread', hue='model_type')
    plt.ylabel(r'$F_{\mathcal{T}}(\sigma)$')
    plt.xlabel(r'$\sigma$')
    plt.savefig(f'{save_folder}/sementic_spead.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Sementic spread plot saved at {save_folder}/sementic_spead.png')
    return True


def main():

    parser = argparse.ArgumentParser(description="Measure homogenisation")

    parser.add_argument(
        "--folder_path",
        type=str,
        help="Folder with all keywords data",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="data/generated",
        help="Where to save output embeddings",
    )
    parser.add_argument(
        "--model_info_path",
        type=str,
        default='data/model_info.json',
        help="Path to model information",
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        help="Path to teh pkl file of keywords embeddings and labels",
    )
    args = parser.parse_args()

    measure_homogenisation(args.folder_path, args.save_folder, args.model_info_path, args.keywords_path)

if __name__ == "__main__":
    main()

