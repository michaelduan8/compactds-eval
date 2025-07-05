from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable

@dataclass
class CorpusConfig(Serializable):
    corpus_name: str                   # Name of corpus to load
    data_paths: List[str]              # Paths to data used to construct corpus
    block_length: Optional[int] = None 
    stride: Optional[int] = None
    data_size: Optional[int] = None
    sharded: Optional[bool] = False
    prepend_title: Optional[bool] = False


@dataclass
class EmbedConfig(Serializable):
    embed_model: str
    embed_dir: Optional[str] = 'embeds' # Where query embeds will be stored
    batch_size: Optional[int] = 512
    embed_size: Optional[int] = 768 # Contriever default
    device: Optional[str] = 'cuda:0'


@dataclass
class OfflineRetrievalParams(Serializable):
    retrieval_results_path: str                             # Path to retrieval results
    matching_key: str     
    ctx_key: str                                  # Key for matching retrieval results to original benchmark question (e.g., query used for retrieval).                    
    retrieval_text_key: Optional[str] = 'retrieval text'    # If retrieved documents contain additional metadata, specify key containing the retrieval text.

    sources_to_filter: Optional[str] = None
    sources_to_keep: Optional[str] = None
    max_chunk_len: Optional[int] = None      # Character length maximum for ensuring retrieved chunks aren't too long

    presort_key: Optional[str] = None
    sort_key: Optional[str] = "retrieval score"
    threshold: Optional[float] = None
    rerank_k: Optional[int] = 100

@dataclass
class OnlineRetrievalParams(Serializable):
    retrieval_type: str              # Retrieval method to use
    corpus_config: CorpusConfig        # Corpus configuration details
    index_dir: Optional[str] = 'index' # Directory where index is located
    batch_size: Optional[int] = 64    
    embed_config: Optional[EmbedConfig] = None


@dataclass
class RetrievalConfig(Serializable):
    k: int  # Top k for retrieval
    online_retrieval: Optional[OnlineRetrievalParams] = None
    offline_retrieval: Optional[OfflineRetrievalParams] = None
    augment_method: Optional[str] = None   # Method to augment generator with retrieval output

@dataclass
class GeneratorConfig(Serializable):
    model_name: str                        # Model name for generator
    model_type: Optional[str] = 'vllm'
    max_input_length: Optional[int] = 32000     # Currently only supported for local vllm
    max_output_length: Optional[int] = 32       # Maximum output token length. Currently ignored if task type is MC


@dataclass
class ExperimentConfig(Serializable):
    experiment_name: str
    task_config_paths: List[str]
    gen_config: GeneratorConfig 
    retrieval_config: Optional[RetrievalConfig] = None
    cache_dir: Optional[str] = 'cache'
    output_dir: Optional[str] = 'output'               
    debug: Optional[bool] = False                       # Enable debug mode
    seed: Optional[int] = 1                   

