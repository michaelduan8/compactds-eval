'''
Rerank retrieval results with oracle reranking using downstream performance.
'''
import string

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.rerank.config import BM25RerankConfig
from src.rerank.utils import pre_sort

import logging
logger = logging.getLogger(__name__)

def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def bm25_rerank(retrieval_results, rc: BM25RerankConfig):
    logger.info(rc)

    retrieval_results = pre_sort(retrieval_results, rc)

    for entry in tqdm(retrieval_results):
        query = entry['query']
        retrieved_texts = [preprocess(item['retrieval text']).split() for item in entry['rubric_annotated_ctxs']]
        
        bm25 = BM25Okapi(retrieved_texts)
        bm25_scores = bm25.get_scores(query)

        for i, item in enumerate(entry['rubric_annotated_ctxs']): 
            item['bm25_score'] = bm25_scores[i]

    return retrieval_results


