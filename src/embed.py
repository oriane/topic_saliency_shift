import os

import numpy as np
from scipy.spatial.distance import pdist

from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
                base_url=os.environ.get('OPENROUTER_ENDPOINT'),
                api_key=os.environ.get('OPENROUTER_API_KEY'),
            )

def compute_embeddings(batch):
    return client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=batch,
        encoding_format="float"
    )

def compute_embeddings_batch(to_embed, batch_size=10):
    embeddings = []
    for i in tqdm(range(0, len(to_embed), batch_size)):
        batch_texts = to_embed[i:i + batch_size].tolist()
        nested_embeddings = compute_embeddings(batch_texts)
        embeddings.extend([t.embedding for t in nested_embeddings.data])
    return embeddings

def get_pdist(X):
    pdists = pdist(np.stack(X.values), metric='cosine')
    return np.mean(pdists)