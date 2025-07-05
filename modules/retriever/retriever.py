from enum import Enum

from .bm25 import BM25
from .dense import DenseRetrieval

class RetrieverTypes(str, Enum):
    BM25 = 'bm25'
    DENSE = 'dense'

class Retriever(object):
    def __init__(self, corpus, retrieval_config, k):
        self.corpus = corpus
        self.retrieval_config = retrieval_config
        self.retriever_type = retrieval_config.retrieval_type
        self.index_dir = retrieval_config.index_dir
        self.batch_size = retrieval_config.batch_size
        self.k = retrieval_config.k
        self.embed_config = retrieval_config.embed_config

        self.retriever = self.load_retriever()

    def get_name(self):
        return self.retriever_type

    def load_retriever(self):
        if self.retriever_type == RetrieverTypes.BM25:
            return BM25(self.corpus, self.index_dir)
        if self.retriever_type == RetrieverTypes.DENSE:
            return DenseRetrieval(self.corpus, self.index_dir, self.embed_config)
        else:
            raise NotImplementedError()

    def retrieve(self, queries):
        return self.retriever.retrieve(queries, self.batch_size, self.k)
