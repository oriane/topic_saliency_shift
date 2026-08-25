import json
from pathlib import Path

import pandas as pd


def load_full_df(folder_path, model_info_path, keywords_label_path, include_kw_embeddings=True):
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
    if include_kw_embeddings:
        kw2emb = kws_count.set_index('keywords')['embeddings'].to_dict()
        full_df['emb'] = full_df['keywords'].map(kw2emb)
        full_df = full_df.loc[~full_df.emb.isna()] # filters out empty rows

    missing_keys = set(kws_count.columns) - model_info.keys()
    if missing_keys != {'embeddings', 'keywords', 'topic_name'}:
        print(f'!Models {missing_keys - {'embeddings', 'keywords', 'topic_name'}} are not included in the kws count!')

    return full_df

def get_topic_winrate(full_df):
    topic_presence_df = full_df[['model_type', 'model_family', 'qid', 'uid', 'topic', 'model_name']].drop_duplicates(
        ignore_index=True)
    topic_presence_df = topic_presence_df.loc[topic_presence_df.topic != 'none']
    sparse_df = pd.crosstab(
        index=[topic_presence_df['model_name'], topic_presence_df['uid'], topic_presence_df['model_type'],
               topic_presence_df['model_family'], topic_presence_df['qid']],
        columns=topic_presence_df['topic']
    ).reset_index()
    sparse_df = sparse_df.rename(columns={'State-led': 'state-led', 'taxation': 'carbon-pricing',
                                          'pension': 'pension reform', 'landlords': 'landlordship reform'
        , 'guaranteed incomes': 'universal basic income',
                                          'end means-testing': 'abolish means-testing',
                                          'job-training': 'reskilling'})
    topic_columns = sparse_df.columns.drop(['model_name', 'uid', 'model_type', 'model_family', 'qid'])
    sparse_df[topic_columns] = sparse_df[topic_columns].clip(upper=1)  # clip to one to test for presence/absence
    topic_winrate = sparse_df.groupby(['model_name', 'model_type', 'model_family', 'qid']).sum().reset_index()
    topic_winrate['n'] = 50
    return topic_winrate, topic_columns