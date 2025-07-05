import json
import os
import subprocess
import time

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher

class BM25(object):
    def __init__(self, corpus, index_dir):
        self.corpus = corpus
        self.index_dir = index_dir
        self.index_path = os.path.join(self.index_dir, f"{self.corpus.get_name()}.bm25.index")

        self.create_or_load_index()

    def create_or_load_index(self):
        if not os.path.exists(self.index_path):
            start_time = time.time()

            tmp_dir = os.path.join("tmp_data", f"{self.corpus.get_name()}")
            os.makedirs(tmp_dir, exist_ok=True)

            tmp_file = os.path.join(tmp_dir, f"{self.corpus.get_name()}.jsonl")
            if not os.path.exists(tmp_file):
                with open(tmp_file, "w") as f:
                    for i, dp in enumerate(self.corpus.get_data()):
                        f.write(json.dumps({
                            "id": dp["id"],
                            "contents": dp["text"]
                        })+"\n")

            command = f"""python -m pyserini.index.lucene \
                    --collection JsonCollection \
                    --input {tmp_dir} \
                    --index {self.index_path} \
                    --generator DefaultLuceneDocumentGenerator \
                    --threads 1 --storePositions --storeDocvectors"""

            result = subprocess.run([command], shell=True, capture_output=True, text=True)
            assert len(result.stderr)==0, result.stderr
            print (f"Constructed the bm25 index for {self.corpus.get_name()} ({time.time()-start_time}s)")
                   
        self.index = LuceneSearcher(self.index_path)
        
    def retrieve(self, queries, batch_size=100, k=5):
        if type(queries) != list:
            queries = [queries]

        hits = [self.index.search(query, k=k) for query in tqdm(queries)]

        all_blocks = []
        all_scores = []
        for _hits in hits:
            blocks = []
            scores = []
            for hit in _hits:
                blocks.append(self.corpus[int(hit.docid)])
                scores.append(hit.score)
            
            all_blocks.append(blocks)
            all_scores.append(scores)
        return all_blocks, all_scores







