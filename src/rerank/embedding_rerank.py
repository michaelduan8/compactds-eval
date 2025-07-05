'''
Rerank retrieval results with oracle reranking using downstream performance.
'''
import numpy as np
import os
import torch

from collections import defaultdict
from tqdm import tqdm

from modules.retriever.embed_util import load_embed_model, embed_batch
from src.rerank.config import EmbeddingRerankConfig

# note that `generate` doesn't actually always generate - see the code for details
from src.rerank.utils import block_data, pre_sort
from src.utils import write, sort_dicts_by_key, load_jsonl 

import logging
logger = logging.getLogger(__name__)


def cossim(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


def embedding_rerank(retrieval_results, rc: EmbeddingRerankConfig):
    logger.info(rc)

    retrieval_results = pre_sort(retrieval_results, rc)

    if rc.chunk_size:
        # TODO: Consider extracting this as a utility function in util.py so all rerank scripts can use
        # Note: Will need to implement doc_extraction handling methods if this logic is brought to other rerank methods
        chunked_file = os.path.join(rc.intermediate_output_dir, f"chunked_results_size_{rc.chunk_size}.jsonl")
        if not os.path.exists(chunked_file):
            for dt in tqdm(retrieval_results, desc="Chunking texts..."):
                ctxs = dt[rc.ctx_key]
                if rc.doc_extraction and rc.top_k_rerank:
                    # Only rerank chunks for specific top_k_rerank documents for efficiency
                    ctxs = ctxs[:rc.top_k_rerank]

                new_ctxs = []
                for i, ctx in enumerate(ctxs):
                    original_text = ctx[rc.retrieval_text_key]
                    blocks = block_data(original_text, rc.chunk_size, min_block_length=(rc.chunk_size // 2))
                    for block in blocks:
                        block_metadata = {
                            rc.retrieval_text_key: block,
                            "source": ctx["source"]
                        }

                        if rc.doc_extraction:
                            block_metadata["ctx_id"] = i

                        new_ctxs.append(block_metadata)


                dt[rc.ctx_key] = new_ctxs
            
            write(chunked_file, retrieval_results)
        else:
            # Use pre-chunked file
            retrieval_results = load_jsonl(chunked_file)

    # Build list of texts for embedding
    embed_requests = []
    for dt in tqdm(retrieval_results, desc="Building embed requests"):
        # For each query, add embed request for original query and ctxs if ctxs is non-empty
        query = dt[rc.retrieval_query_key]
        embed_requests.append(query)

        ctxs = dt[rc.ctx_key]
        if rc.top_k_rerank and not rc.chunk_size:
            # If chunk_size specified, top_k_rerank is applied during chunking phase
            # Do not perform top_k_reranking immediately after chunking
            ctxs = ctxs[:rc.top_k_rerank]

        for ctx in ctxs:
            text = ctx[rc.retrieval_text_key][:rc.retrieval_trunc_len]
            embed_requests.append(text)

    print(f"Num texts to embed: {len(embed_requests)}")

    # Load embed model
    model, tokenizer = load_embed_model(rc.embedder_model, rc.device, rc.mode)

    output_shape = None
    embeddings_memmap = None
    embed_file = os.path.join(rc.intermediate_output_dir, "reranking_embed.npy")
    
    # Make the intermediate output dir
    os.makedirs(rc.intermediate_output_dir, exist_ok=True)
    
    if not os.path.exists(embed_file):
        with torch.no_grad():
            # Process passages in batches
            for i in tqdm(range(0, len(embed_requests), rc.batch_size)):
                batch = embed_requests[i:i + rc.batch_size]
                
                embeddings = embed_batch(batch, model, tokenizer, rc.device, mode=rc.mode)

                if embeddings_memmap is None:
                    # Instantiate embeddings map with length of corpus and shape of embedding
                    print(f"Embed shape: {embeddings.shape}")
                    embed_shape = embeddings.shape
                    output_shape = (len(embed_requests), embed_shape[1])
                    print(f"Output shape: {output_shape}")
                    embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='w+', shape=output_shape)

                # Move embeddings to CPU and convert to numpy
                embeddings_memmap[i:i + len(batch)] = embeddings
                embeddings_memmap.flush()

                write_equal = np.all(np.equal(embeddings[0], embeddings_memmap[i]))

                assert write_equal

            print(embeddings_memmap.shape)
    else:
        # Get embedding size from output
        test_embedding = embed_batch(embed_requests[:1], model, tokenizer, rc.device, mode=rc.mode)
        embed_shape = test_embedding.shape
        output_shape = (len(embed_requests), embed_shape[1])
        embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='r', shape=output_shape)

    # Build retrieval scores and rerank
    new_score_key = rc.new_score_key if rc.new_score_key else rc.retrieval_score_key
    idx= 0
    for dt in tqdm(retrieval_results, desc="Building retrieval scores"):
        query = dt[rc.retrieval_query_key]
        ctxs = dt[rc.ctx_key]

        # Get embedding chunk and compute retrieval scores
        # +1 to account for the query embed before ctx embeds
        actual_ctx_count = min(rc.top_k_rerank, len(ctxs)) if rc.top_k_rerank and not rc.chunk_size else len(ctxs)
        embedding_chunk_size = 1 + actual_ctx_count
        embedding_chunk = embeddings_memmap[idx: idx + embedding_chunk_size]

        if embedding_chunk.shape[0] < embedding_chunk_size:
            raise ValueError(f"Insufficient number of embeddings for query at idx={idx}, expected={embedding_chunk_size}, got={embedding_chunk.shape[0]}")

        query_embed = embedding_chunk[0]

        if actual_ctx_count > 0:
            # Note: Embedding Rerank uses exact Cosine Similarity, making it
            # incomparable to retrieval scores from approximate similarity methods
            if rc.doc_extraction:
                id_to_chunks = defaultdict(list)
                for ctx, embed in zip(ctxs, embedding_chunk[1:]):
                    ctx_id = ctx["ctx_id"]
                    retrieval_score = cossim(query_embed, embed)
                    ctx[new_score_key] = retrieval_score.item()
                    id_to_chunks[ctx_id].append(ctx)

                best_chunks = []
                for i in sorted(id_to_chunks.keys()):
                    doc_chunks = id_to_chunks[i]
                    ordered_chunks, _ = sort_dicts_by_key(doc_chunks, [new_score_key, rc.retrieval_score_key], reversed=True)
                    best_chunks.append(ordered_chunks[0])

                dt[rc.ctx_key] = best_chunks
            else:
                for ctx, embed in zip(ctxs, embedding_chunk[1:]):
                    retrieval_score = cossim(query_embed, embed)
                    ctx[new_score_key] = retrieval_score.item()

                dt[rc.ctx_key], _ = sort_dicts_by_key(ctxs, [new_score_key, rc.retrieval_score_key], reversed=True)

        idx += embedding_chunk_size
    
    assert idx == len(embeddings_memmap), f"{idx}, {len(embeddings_memmap)}"

    return retrieval_results


