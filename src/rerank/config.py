from dataclasses import dataclass
from typing import Optional, List, Dict
from simple_parsing.helpers import Serializable

from src.rerank.utils import TRUNC_LEN

@dataclass
class BM25RerankConfig(Serializable):
    retrieval_query_key: str
    ctx_key: str

    top_k_rerank: Optional[int] = None
    retrieval_text_key: Optional[str] = 'retrieval text'  
    retrieval_trunc_len: Optional[int] = TRUNC_LEN
    retrieval_score_key: Optional[str] = 'retrieval score'  
    new_score_key = "bm25 score"


@dataclass
# TODO: user specifies a tuple of keys to rerank beforehand (e.g., for applying top k rerank), then under new_score key, store newly computed rerank value and reorder
class EmbeddingRerankConfig(Serializable):
    retrieval_query_key: str
    ctx_key: str
    embedder_model: str

    top_k_rerank: Optional[int] = None
    doc_extraction: Optional[bool] = False
    device: Optional[str] = "cuda"  # Vllm doesn't have well-documented embed utility yet, so use hf
    batch_size: Optional[int] = 512
    # TODO: # Consider making chunking a general functionality
    chunk_size: Optional[int] = None
    sources_to_filter: Optional[str] = None
    sources_to_keep: Optional[str] = None
    retrieval_text_key: Optional[str] = 'retrieval text'                # Specify the key in each retrieval result dictionary that maps to the raw retrieval text
    retrieval_trunc_len: Optional[int] = TRUNC_LEN
    retrieval_score_key: Optional[str] = 'retrieval score'              # Specify the key in each retrieval result dictionary that maps to the retrieval score
    new_score_key:  Optional[str] = None            
    mode: Optional[str] = 'default'
    intermediate_output_dir: Optional[str] = 'intermediate_outputs'     # Directory to write intermediate outputs, such as the computed reranking metrics per doc or pkl'd query to reranked docs map

@dataclass
class ModeArgs(Serializable):
    answer_choices: Optional[List] = None
    prompt_suffix_key: Optional[str] = None


@dataclass
class OracleRerankConfig(Serializable):
    task_name: str                                                      # Name of the task to run
    retrieval_id_key: str                                               # Specify key in each row of the input JSONL file whose value can be used to link retrieval results to their respective query.
    ctx_key: str                                                        # Specify key in each row of the input JSONL file mapping to the original retrieval results 
    top_k_rerank: int                                                   # How many of the top original retrieved docs to rerank
    reranker_model: str                                                 # Model to use for computing rerank scores
    
    sources_to_filter: Optional[str] = None
    sources_to_keep: Optional[str] = None
    retrieval_text_key: Optional[str] = 'retrieval text'                # Specify the key in each retrieval result dictionary that maps to the raw retrieval text
    retrieval_trunc_len: Optional[int] = TRUNC_LEN
    retrieval_score_key: Optional[str] = 'retrieval score'              # Specify the key in each retrieval result dictionary that maps to the retrieval score
    new_score_key:  Optional[str] = 'oracle score' 
    model_type: Optional[str] = 'vllm'                                  # Specify generation mode for Generator
    mode_args: Optional[ModeArgs] = None                                # Args specific to the generation mode used. For MC, one must define an "answer_choices" list. For computing prompt logprobs, a "prompt_suffix_key" needs to be provided mapping to the prompt suffix to compute logprob over. 
    intermediate_output_dir: Optional[str] = 'intermediate_outputs'     # Directory to write intermediate outputs, such as the computed reranking metrics per doc or pkl'd query to reranked docs map


@dataclass
class RubricAnnotateConfig(Serializable):
    retrieval_query_key: str
    retrieval_id_key: str  
    ctx_key: str
    top_k_rerank: int
    
    rubric_file: str
    reranker_model: str 

    sources_to_filter: Optional[str] = None
    sources_to_keep: Optional[str] = None
    model_type: Optional[str] = 'vllm'  
    max_output_len: Optional[int] = 1024
    retrieval_text_key: Optional[str] = 'retrieval text'                # Specify the key in each retrieval result dictionary that maps to the raw retrieval text
    retrieval_trunc_len: Optional[int] = TRUNC_LEN
    retrieval_score_key: Optional[str] = 'retrieval score' 
    new_score_key:  Optional[str] = 'rubric score' 
    intermediate_output_dir: Optional[str] = 'intermediate_outputs'     # Directory to write intermediate outputs, such as the computed reranking metrics per doc or pkl'd query to reranked docs map
