import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.embed import compute_embeddings_batch
from src.knn import classify


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kw_reference_file",
        type=str,
        default='data/knn_data/kws_gt.csv',
        help="File with the manually curated keywords references.",
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        default='data/generated',
        help="Where to store the intermediate saved files.",
    )

    parser.add_argument(
        "--kw_reference_embeddings",
        type=str,
        default=None,
        help="Embeddings of keywords references if it exists, as pickled file. They will be computed if left to None.",
    )

    parser.add_argument(
        "--kw_folder",
        type=str,
        help="Folder with all extracted keywords.",
    )

    parser.add_argument(
        "--current_classification",
        type=str,
        default=None,
        help="""If it exists, current classification and embeddings. 
        Only keywords absent of this file will be embedded and classified, or all of them if left to None.""",
    )

    parser.add_argument(
        "--overwrite_current_label",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Whether to overwrite current label and reperform knn on all keywords. 
        If a current classification is provided, only the associated keywords embeddings will be kept but label will be 
        recomputed.""",
    )

    parser.add_argument(
        "--test_set_path",
        type=str,
        default='data/knn_data/kws_test_set.csv',
        help="Path to the test set.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.kw_reference_embeddings is not None:
        print('Use existing reference embeddings.')
        reference_embeddings = pd.read_pickle(args.kw_reference_embeddings)
    else:
        print('Compute new keywords embeddings.')
        reference_embeddings = embed_reference(args.kw_reference_file)

    print('Classify new keywords.')
    kw_count_labelled = classify_keywords(args.kw_folder, args.current_classification,
                                          reference_embeddings, args.save_folder,
                                          args.overwrite_current_label)

    print('Test Classification')
    accuracy = get_classification_accuracy(kw_count_labelled, args.test_set_path, args.save_folder)
    print(f'Annotation Accuracy {accuracy}')


def embed_reference(kw_reference_file):
    references = pd.read_csv(kw_reference_file, index_col=0)
    # Some of the keywords were not assigned to any category, we ignore those
    references = references.loc[~(references.topic_name == 'none')]
    # Keywords embedding is together with the topic to help steer it towards meaningful representation in context
    ref_embeding = references.apply(lambda row: f'Topic: {row.topic_name}, Keywords: {row.keywords}', axis=1)

    references['embeddings'] = compute_embeddings_batch(ref_embeding)

    original_file = Path(kw_reference_file)
    save_file = original_file.with_name(f"{original_file.stem}_embedded{original_file.suffix}")
    references.to_pickle(save_file)
    print(f'Embeddings saved to {save_file}.')
    return references

def get_keywords_count(kw_folder, current):
    folder_path = Path(kw_folder)

    # Get all files and sort them alphabetically
    files = sorted(f for f in folder_path.iterdir() if f.is_file())

    dfs = []
    for file in files:
        if '.DS_Store' not in file.name:
            dfs.append(pd.read_csv(file, index_col=0))
    raw_df = pd.concat(dfs)
    raw_df['keywords'] = raw_df['keywords'].apply(eval)
    sparse_df = raw_df.explode('keywords')
    sparse_df['keywords'] = sparse_df['keywords'].fillna('none').apply(lambda text: text.strip().lower())
    sparse_count = sparse_df.groupby(['model', 'keywords'])['response'].count().unstack().T

    if current is not None:
        kws_count = pd.read_pickle(current)
        kws_count_updated = pd.merge(sparse_count,
                                     kws_count[['keywords', 'topic_name', 'embeddings']], how='outer', on='keywords')
    else:
        kws_count_updated = sparse_count
        kws_count_updated['embeddings'] = None
        kws_count_updated = kws_count_updated.reset_index()
    # Remove empty keywords
    kws_count_updated = kws_count_updated.loc[
        (kws_count_updated.keywords != '') & ~(kws_count_updated.keywords.isna())]
    return kws_count_updated

def embed_keywords(kw_count, save_folder):
    mask = kw_count['embeddings'].isna()
    to_embed = kw_count.loc[mask].keywords.values
    embeddings = compute_embeddings_batch(to_embed)
    kw_count.loc[mask, 'embeddings'] = (pd.Series(embeddings, index=kw_count
                                                           .index[mask]))
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh")
    kw_count.to_pickle(f'{save_folder}/kws_counts_embedded_{timestamp}.pkl')
    print(f'Keywords embeddings saved to {save_folder}/kws_counts_embedded_{timestamp}.pkl')

    kw_count.loc[mask, 'embeddings'] = (pd.Series(embeddings, index=kw_count
                                                  .index[mask]))
    return kw_count


def classify_keywords(kw_folder, current_classification, reference_embeddings, save_folder, overwrite):
    kws_count = get_keywords_count(kw_folder, current_classification)
    new_kws = kws_count.embeddings.isna().sum()
    if new_kws > 0:
        print(f'There are {new_kws} new keywords to embed')
        kws_count = embed_keywords(kws_count, save_folder)

    if overwrite:
        kws_count['topic_name'] = None
        kws_count.loc[kws_count.index.isin(reference_embeddings.keywords), 'topic_name'] = kws_count.index.map(
            reference_embeddings.set_index('keywords')['topic_name'].to_dict())
    kw_count_labelled = classify(reference_embeddings, kws_count)

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh")
    kw_count_labelled.to_pickle(f'{save_folder}/kws_counts_labeled_{timestamp}.pkl')
    return kw_count_labelled

def get_classification_accuracy(kw_count_labelled, test_set_file, save_folder):
    annotated_test_set = pd.read_csv(test_set_file, index_col=0)
    joined = annotated_test_set.set_index('keywords').join(kw_count_labelled[['keywords', 'topic_name']].set_index('keywords'),
                                                           lsuffix='_anno', how='left')
    accuracy = (joined.topic_name_anno == joined.topic_name).sum() / len(joined)
    errors = joined.loc[joined.topic_name_anno != joined.topic_name]

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh")
    errors.to_csv(f'{save_folder}/misclassified_{timestamp}.csv')
    print(f'Misclassified keywords saved to {save_folder}/misclassified_{timestamp}.csv')
    return accuracy

if __name__ == "__main__":
    main()

