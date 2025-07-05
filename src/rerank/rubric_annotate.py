'''
Rerank retrieval results with LLM-as-judge rubric-based annotation
'''
import os

from collections import defaultdict
from tqdm import tqdm

from src.rerank.config import RubricAnnotateConfig
from src.rerank.structured_output import RubricResponse

# note that `generate` doesn't actually always generate - see the code for details
from src.rerank.utils import generate, pre_sort
from src.utils import load_jsonl, sort_dicts_by_key, write

import logging
logger = logging.getLogger(__name__)


def rubric_annotate(retrieval_results, rc: RubricAnnotateConfig):
    logger.info(rc)

    retrieval_results = pre_sort(retrieval_results)

    # Build this for validation
    id_to_docs = {}
    for cache in tqdm(retrieval_results):
        id = cache[rc.retrieval_id_key]

        # Avoid duplicates
        if id not in id_to_docs:
            id_to_docs[id] = cache[rc.ctx_key]
        else:
            print(f"{id} duplicated")

    print(len(id_to_docs))

    # Load rubric
    # TODO: support loading multiple rubrics, i.e. domain-/query-specific
    with open(rc.rubric_file, 'r') as file:
        rubric = file.read()
    
    # Generate prompts for rubric annotation
    rubric_annotation_prompts = []
    for i, dt in tqdm(enumerate(retrieval_results), desc="Generating rubric_annotation prompts"):
        # Get associated top-k docs
        docs = dt[rc.ctx_key][:rc.top_k_rerank]
        if len(docs) < rc.top_k_rerank:
            logger.warning(f"Less than top_k_rerank retrieved contexts for reranking, currrent count: {len(docs)}")

        full_text = dt[rc.retrieval_query_key]

        gen_meta_data = {
            rc.retrieval_query_key: full_text,
            rc.retrieval_id_key: dt[rc.retrieval_id_key]
        }

        # Create an in-context eval prompt for each document retrieved for query
        for doc in docs:
            retrieval_text = doc[rc.retrieval_text_key]
            # rubric_annotation_prompt = rubric.format(retrieval_text=retrieval_text, query=query, answer=answer)
            rubric_annotation_prompt = rubric.format(retrieval_text=retrieval_text[:rc.retrieval_trunc_len], full_text=full_text)
            rubric_annotation_prompts.append({
                "doc": doc, 
                "rubric_annotation_prompt": rubric_annotation_prompt
            } | gen_meta_data)


    print(f"Annotating {len(rubric_annotation_prompts)} prompts!")

    intermediate_rerank_metric_path = os.path.join(rc.intermediate_output_dir, "rubric_annotation_metrics.jsonl")
    if os.path.exists(intermediate_rerank_metric_path):
        rubric_annotation_outputs = load_jsonl(intermediate_rerank_metric_path)
    else:
        gen_kwargs = {
            "structured_output": RubricResponse.model_json_schema(),
            "max_output_length": rc.max_output_len
        }
        print(rubric_annotation_prompts[0]['rubric_annotation_prompt'])
        rubric_annotation_outputs = generate(model_name=rc.reranker_model, 
                                        model_type=rc.model_type, 
                                        prompts_and_metadata=rubric_annotation_prompts,
                                        prompt_key='rubric_annotation_prompt',
                                        gen_output_key='rubric_annotation',
                                        structured=True,
                                        **gen_kwargs)


        write(intermediate_rerank_metric_path, rubric_annotation_outputs)
        
    print("Reorganizing rubric_annotation metadata...")
    # Align results from rerank metric computation with that of the original input queries
    unranked_id_to_doc = defaultdict(list)      
    for dt in rubric_annotation_outputs:
        gk = dt[rc.retrieval_id_key]
        if gk not in unranked_id_to_doc.keys() or len(unranked_id_to_doc[gk]) < rc.top_k_rerank:
            unranked_id_to_doc[gk].append(dt)

    # Sanity check, we should have the same queries as keys and the same number of retrieved docs 
    for query, _ in unranked_id_to_doc.items():
        assert query in id_to_docs.keys(), f"Missing Query: {query}"

    print("Reranking retrieved docs per query by rubric annotation...")
    new_score_key = rc.new_score_key if rc.new_score_key else rc.retrieval_score_key
    reranked_id_to_doc = defaultdict(list)
    for query, data in unranked_id_to_doc.items():
        orig_retrieved_docs = id_to_docs[query]

        for i, dt in enumerate(data):
            # We should be able to find the retrieved document at the same index in the original query_to_docs map
            assert orig_retrieved_docs[i][rc.retrieval_text_key][:rc.retrieval_trunc_len] == dt["doc"][rc.retrieval_text_key][rc.retrieval_trunc_len]

            # Map rubric annotation output to the RubricResponse model
            rubric_response = RubricResponse.parse_raw(dt['rubric_annotation']['output'])

            orig_retrieved_docs[i] = orig_retrieved_docs[i] | {
                new_score_key: rubric_response.score,
                "rubric reason": rubric_response.reason,
                "source": dt["doc"]["source"],
            }
        
        orig_retrieved_docs, _ = sort_dicts_by_key(orig_retrieved_docs, [new_score_key, rc.retrieval_score_key], reversed=True)
        reranked_id_to_doc[query] = orig_retrieved_docs
    
    # Store the reranked query_to_doc mapping
    intermediate_reranked_query_to_doc_path = os.path.join(rc.intermediate_output_dir, "rubric_annotated_query_to_doc.pkl")
    write(intermediate_reranked_query_to_doc_path, reranked_id_to_doc, mode='pkl')

    out = []
    for dt in tqdm(retrieval_results, desc="Adding reranked queries to retrieval results"):
        # Get associated top-k docs
        id = dt[rc.retrieval_id_key]
        docs = reranked_id_to_doc[id]
        
        out.append(dt | {rc.ctx_key: docs}) 

    return out


