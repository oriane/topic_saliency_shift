import chromadb
from tqdm import tqdm

CLIENT = chromadb.Client()
COLLECTION = CLIENT.create_collection(name='to_label')

def index_reference(references):
    for i, row in tqdm(references.iterrows()):
        COLLECTION.add(
            ids=[str(i)],
            embeddings=row.embeddings,
            documents=[row.keywords]
        )

def knn_clustering(row, references):
    res = COLLECTION.query(
            query_embeddings=row['embeddings'],
            n_results=20
        )
    kneigh = references.loc[[float(i) for i in res['ids'][0]]].topic_name
    most_common_label = kneigh.value_counts().index[0]
    pass_thresh = kneigh.value_counts(normalize=True).iloc[0] >= 0.3

    if pass_thresh:
        return most_common_label
    else:
        return 'none'

def classify(references, kws_count):
    print(f'Index references.')
    tqdm.pandas()
    to_label = kws_count.loc[kws_count['topic_name'].isna()]
    if not to_label.empty:
        index_reference(references)
        new_labels = to_label.progress_apply(knn_clustering, axis=1, references=references)
        unlabeled = (new_labels == 'none').sum() / len(to_label)
        print(f'Proportion of keyword without cluster {unlabeled}')
        kws_count.loc[to_label.index, 'topic_name'] = new_labels
    else:
        print('All keywords already labeled.')
    return kws_count