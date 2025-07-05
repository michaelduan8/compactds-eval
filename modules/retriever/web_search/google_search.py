'''
Google-based web search wrapper
'''
import math
import os
import time

from collections import defaultdict
from functools import partial
from tqdm import tqdm

from googleapiclient.discovery import build

from modules.retriever.web_search.web_search import WebSearch


# Maximum search hits per query
MAX_K = 100

# Maximum number of results per page
MAX_PAGE_SIZE = 10

# Maximum number of calls per batch
MAX_BATCH_SIZE = 10 #1000

# Daily limit for queries
DAILY_LIMIT = 10000

# How long to sleep between batch calls
BATCH_SLEEP = 15

class GoogleSearch(WebSearch):
    def __init__(self, query_key, k):
        self.query_key = query_key
        self.k = k
        assert self.k <= 100

        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        self.web_search_api_key = os.environ.get('GOOGLE_API_KEY', '')

        self.service = build("customsearch", "v1", developerKey=self.web_search_api_key)


    def retrieve(self, queries_with_metadata):
        # Ensure predicted number of calls is less than daily limit
        n_queries = len(queries_with_metadata)
        n_calls = math.ceil(self.k / MAX_PAGE_SIZE)
        estimated_total_calls = n_queries * n_calls
        if estimated_total_calls > DAILY_LIMIT: print(f"Estimated number of calls ({estimated_total_calls}) exceeds daily limit of {DAILY_LIMIT}")
        print(f"Number of estimated calls: {estimated_total_calls}")

        request_ids_to_calls_to_results = defaultdict(lambda: defaultdict(list))

        def _search_callback(request_id, response, exception, request_result_map, mapping_id, call_id):
            if exception:
                print(f"Error occurred during search {request_id}: {exception}")
            else:
                if 'items' in response:
                    request_result_map[mapping_id][call_id].extend(response['items'])

        prepared_calls = []
        for i, query in tqdm(enumerate(queries_with_metadata), desc="Building query calls"):
            q = query[self.query_key]

            num = min(MAX_PAGE_SIZE, self.k)
            curr_n_calls = n_calls
            start = 1
            while curr_n_calls > 0:
                prepared_calls.append({
                    "query": q,
                    "request": self.service.cse().list(q=q, cx=self.cse_id, start=start, num=num, filter='1'), 
                    "request_id": i,
                    "call_idx": n_calls - curr_n_calls
                })
                curr_n_calls -= 1
                start += MAX_PAGE_SIZE
                leftover = self.k - start + 1
                if 0 < leftover < MAX_PAGE_SIZE:
                    num = leftover

        print(f"Number of prepared calls: {len(prepared_calls)}")

        # Create batch http requests
        for i in tqdm(range(0, len(prepared_calls), MAX_BATCH_SIZE), desc="Batch requests on Google CSE"):
            chunk = prepared_calls[i:i + MAX_BATCH_SIZE]
           
            batch = self.service.new_batch_http_request()
            
            for _, call in enumerate(chunk):
                batch.add(request=call["request"], 
                          callback=partial(_search_callback, request_result_map=request_ids_to_calls_to_results, mapping_id=call['request_id'], call_id=call["call_idx"]), #, query=call["query"]), 
                          request_id=f"{call['request_id']},{call['call_idx']}")
                
            batch.execute()
            print(f"Finished batch {i}")
            time.sleep(BATCH_SLEEP)
        
        print(request_ids_to_calls_to_results.keys())
        out = []
        for i in range(n_queries):
            result_pages = request_ids_to_calls_to_results[i]
            concat_query_results = []
            for j in range(n_calls):
                concat_query_results.extend(result_pages[j])

            out.append(concat_query_results)

        return out