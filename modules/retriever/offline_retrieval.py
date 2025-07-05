import numpy as np

from tqdm import tqdm

from src.config import OfflineRetrievalParams
from src.utils import load_jsonl, sort_dicts_by_key

import logging
logger = logging.getLogger(__name__)


# Special keys for custom interpolation strategies
INTERPOLATION_KEYS = ["base", "hide-in-middle", "bm25", "original", "original-mini", "debug"]

def hide_in_middle(d_list):
    k = len(d_list)
    S = [None] * k 

    for i in range(1, k + 1):
        if i % 2 == 1: 
            order = (i + 1) // 2
        else: 
            order = k + 1 - (i // 2)
        S[order - 1] = d_list[i - 1]

    return S

def merge_dedup_sort(l1, l2, sort_fn, dedup_key="retrieval text"):
    # Merge with deduplication (prioritize l2)
    merged_dict = {item[dedup_key]: item for item in l1}
    merged_dict.update({item[dedup_key]: item for item in l2})
    
    # Convert back to list and sort
    merged_list = list(merged_dict.values())
    sorted_list = sorted(merged_list, key=sort_fn)
    return sorted_list


class OfflineRetrieval(object):
    def __init__(self, retrieval_config: OfflineRetrievalParams, k):
        self.retrieval_config = retrieval_config
        self.retrieval_results_path = retrieval_config.retrieval_results_path
        self.matching_key = retrieval_config.matching_key
        self.ctx_key = retrieval_config.ctx_key
        self.retrieval_text_key = retrieval_config.retrieval_text_key
        self.max_chunk_len = retrieval_config.max_chunk_len
        self.presort_key = retrieval_config.presort_key
        self.sort_key = retrieval_config.sort_key
        self.threshold = retrieval_config.threshold
        self.k = k
        self.rerank_k = retrieval_config.rerank_k

        self.sources_to_filter = retrieval_config.sources_to_filter
        self.sources_to_keep = retrieval_config.sources_to_keep
 
        self.sources_to_filter = self.sources_to_filter.split(",") if self.sources_to_filter is not None else None
        self.sources_to_keep = self.sources_to_keep.split(",") if self.sources_to_keep is not None else None

        assert not (self.sources_to_filter is not None and self.sources_to_keep is not None)
        
        self.query_to_doc = self.load_query_to_doc()
        print(f"Using presort key {self.presort_key} and sort key {self.sort_key}")
        print(f"Num queries with retrieval results: {len(self.query_to_doc)}")

    def get_matching_key(self):
        return self.matching_key

    def get_query_to_doc(self):
        return self.query_to_doc

    def load_query_to_doc(self):
        retrieval_results_paths = self.retrieval_results_path.split(",")
   
        query_to_docs = self._load_query_to_doc(retrieval_results_paths[0])

        if self.threshold:
            for q, docs in query_to_docs.items():
                docs = [doc for doc in docs if doc["retrieval score"] >= self.threshold]
                query_to_docs[q] = docs

        for query in query_to_docs:
            query_to_docs[query], _ = self._resolve_sort(query_to_docs[query], self.sort_key)
        
        for retrieval_results_path in retrieval_results_paths[1:]:
            another_query_to_docs = self._load_query_to_doc(retrieval_results_path)
            for query in another_query_to_docs:
                if query in query_to_docs:
                    existing_query_to_docs = query_to_docs[query]
                    # Merge existing results for query
                    print("Merging datastore results")
                else:
                    existing_query_to_docs = []

                query_to_docs[query], _ = self._resolve_sort(existing_query_to_docs + another_query_to_docs[query], self.sort_key)

        return query_to_docs

    def _load_query_to_doc(self, retrieval_results_path):
        retrieval_results = load_jsonl(retrieval_results_path)

        query_to_docs = {}
        for cache in tqdm(retrieval_results):
            assert self.matching_key in cache
            query = cache[self.matching_key]

            if query not in query_to_docs:
                top_results = cache[self.ctx_key]

                if self.sources_to_filter is not None:
                    top_results = [r for r in top_results if not np.any([
                        r["source"] == src for src in self.sources_to_filter])]
                if self.sources_to_keep is not None:
                    top_results = [r for r in top_results if np.any([
                        r["source"] == src for src in self.sources_to_keep])]
                
                if self.presort_key:
                    top_results, num = self._resolve_sort(top_results, self.presort_key)
                    if num and num < self.rerank_k and self.sources_to_filter is None and self.sources_to_keep is None:
                        logger.warning(f"Not enough docs for query {query} after sorting by {self.presort_key}")
                    
                query_to_docs[query] = top_results[:self.rerank_k]
            else:
                print (f"Another set of retrieval results for {cache[self.matching_key]}")
                # Merge and re-sort results under same key
                # TODO: assume keys are unique; duplicates are due to e.g. results from different subqueries
                # Retrieval score key must exist to resolve merging
                docs = merge_dedup_sort(
                    query_to_docs[query], 
                    cache[self.ctx_key], 
                    sort_fn=self._interpolate_reranked_score)
                query_to_docs[query] = docs

        print(len(query_to_docs))
        return query_to_docs


    def retrieve(self, queries_with_metadata, key=None):
        """
        Batch retrieval
        """
        retrieval_output = []
        for qm in queries_with_metadata:
            top_k_texts = self.retrieve_single(qm, key)
            retrieval_output.append(top_k_texts)
            
        return retrieval_output
    

    def retrieve_single(self, qm, key=None):
        """
        Single query retrieval
        """
        if key and key in self.query_to_doc:
            top_results = self.query_to_doc[key]
        else:
            if qm[self.matching_key] not in self.query_to_doc:
                return None
            top_results = self.query_to_doc[qm[self.matching_key]]
                
        if self.sources_to_filter:
            top_results = [r for r in top_results if not np.any([
                r["source"].startswith(src) for src in self.sources_to_filter])]
        if self.sources_to_keep:
            top_results = [r for r in top_results if np.any([
                r["source"].startswith(src) for src in self.sources_to_keep])]

        top_k_results = top_results[:self.k]

        if self.k >= 100 and self.sort_key == "hide-in-middle":
            top_k_results = hide_in_middle(top_k_results)
            top_k_results = top_k_results[::-1]
        

        top_k_texts = [result[self.retrieval_text_key][:self.max_chunk_len] if self.max_chunk_len else result[self.retrieval_text_key] for result in top_k_results]
        return top_k_texts


    def _interpolate_reranked_score(self, doc):
        if self.sort_key in ["base", "hide-in-middle"]:
            return -float(doc["retrieval score"])
        elif self.sort_key == "bm25":
            assert "bm25_score" in doc, "bm25_score not found in doc"
            return -float(doc["retrieval score"] * 10 + doc["bm25_score"])
        elif self.sort_key == "original":
            assert "original_score" in doc, "original_score not found in doc"
            return -float(0.5 * doc["original_score"] + 0.5 * doc["retrieval score"]  / 10)
        elif self.sort_key == "original-mini":
            assert "original_score" in doc, "original_score not found in doc"
            return -float(0.1 * doc["original_score"] + 0.9 * doc["retrieval score"]  / 10)
        elif self.sort_key == "debug":
            return -float(doc["bm25_score"])
        else:
            raise ValueError(f"Unknown interpolation mode: {self.sort_key}")
        

    def _resolve_sort(self, l, key):
        score_name = key.replace("_", " ")
        if score_name == "original score":
            score_name = "original_score"

        num = None
        if score_name not in INTERPOLATION_KEYS:
            docs, num = sort_dicts_by_key(l, sort_keys=[score_name], reversed=True)
        else:
            docs = sorted(l, key=self._interpolate_reranked_score)
    
        return docs, num


