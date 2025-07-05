'''
Rerank retrieval results with oracle reranking using downstream performance.
'''
import os
import random

from collections import defaultdict
from tqdm import tqdm

from src.rerank.config import OracleRerankConfig

# note that `generate` doesn't actually always generate - see the code for details
from src.rerank.utils import generate, pre_sort
from src.utils import load_jsonl, sort_dicts_by_key, write
from transformers import AutoTokenizer

from olmes.oe_eval.tasks.utils import get_task_object
from modules.generator.generator import Generator

import logging
logger = logging.getLogger(__name__)


# Hardcoded as certain datasets like MMLU_Pro have answers with varying numbers of answer choices
ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def oracle_rerank(retrieval_results, rc: OracleRerankConfig):
    logger.info(rc)

    retrieval_results = pre_sort(retrieval_results, rc)

    # Build this for validation
    id_to_docs = {}
    for cache in tqdm(retrieval_results):
        id_ = cache[rc.retrieval_id_key]
        # There should be no duplicates on id key
        if id_ not in id_to_docs:
            id_to_docs[id_] = cache[rc.ctx_key]
        else:
            raise KeyError(f"{cache[rc.retrieval_id_key]} duplicated")

    print(len(id_to_docs))

    # Load in model only once
    llm = Generator(rc.reranker_model, rc.model_type)
    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(rc.reranker_model)
    llm.get_model().load_model()

    reranked_id_to_doc = defaultdict(list)

    # Get the task configuration and original documents to get targets for each query
    task_objects = get_task_object(rc.task_name, rc.intermediate_output_dir)
    for task in task_objects:
        task.download()
        task_name = task.task_config['metadata']['alias']
        documents = task.get_eval_docs(limit=task.task_config.get("limit"), random_subsample_seed=2025)
        print(len(documents))
        print(f"{task_name}:{documents[0][task.task_config.get('native_id_field', 'id')]}")
        documents = [doc for doc in documents if f"{task_name}:{doc[task.task_config.get('native_id_field', 'id')]}" in id_to_docs]

        if len(documents) == 0:
            print(f"No documents found for task {task_name} in id_to_docs.")
            continue

        # Decide if it is a multiple choice task
        if "cot" not in task_name:
            mode = "mc"
        else:
            logger.warning("Oracle reranking is not well defined over CoT problems")
            mode = "prompt_logprobs"    
    
        # Generate prompts for oracle reranking
        oracle_rerank_prompts = []
        for document in tqdm(documents, desc=f"Generating oracle_rerank prompts for {task_name} in mode {mode}"):
            doc_id = f"{task_name}:{document[task.task_config.get('native_id_field', 'id')]}"

            # Get associated top-k docs
            docs = id_to_docs[doc_id][:rc.top_k_rerank]
            if len(docs) < rc.top_k_rerank:
                logger.warning(f"Less than top_k_rerank retrieved contexts for reranking, currrent count: {len(docs)}")

            # Construct the prompt for the query
            fewshot_seed = task.task_config.get("fewshot_seed", 1234)
            if fewshot_seed is not None:
                if fewshot_seed < 0:
                    fewshot_seed += doc_id
                rnd = random.Random(fewshot_seed)
            else:
                rnd = None

            query_with_fewshot = task.fewshot_context(
                                    document,
                                    num_fewshot=task.task_config.get("num_shots", 0),
                                    rnd=rnd,
                                    description=task.task_config["context_kwargs"].get("description"),
                                    final_description=task.task_config["context_kwargs"].get("final_description"),
                                    system_prompt=task.task_config["context_kwargs"].get("system_prompt"),
                                    assistant_prefix=task.task_config["context_kwargs"].get("assistant_prefix"),
                                    use_chat_format=task.task_config.get("use_chat_format", False),
                                    fewshot_as_multiturn=task.task_config["context_kwargs"].get(
                                        "fewshot_as_multiturn", False
                                    ),
                                    retrieval_prefix=None,
                                    offline_retriever=None
                                )

            ground_truth = task.doc_to_target(document).strip()

            # Add the ground truth to the prompt for non-MC oracle
            if mode == "prompt_logprobs":
                query_with_fewshot = query_with_fewshot['messages'][0]['content']
                # ground_truth = ground_truth.replace("Let's think step by step:", "")
                query_with_fewshot += f" {ground_truth}"
            
            gen_meta_data = {
                "id": doc_id,
                "context": query_with_fewshot,
                "ground_truth": ground_truth
            }

            # Append the original query with fewshot
            oracle_rerank_prompts.append({
                "doc": None, 
                "oracle_rerank_prompt": query_with_fewshot
            } | gen_meta_data)

            # Create an in-context eval prompt for each document retrieved for query
            for doc in docs:
                retrieval_text = doc[rc.retrieval_text_key][:rc.retrieval_trunc_len]
                oracle_rerank_prompt = f"{retrieval_text}\n\n{query_with_fewshot}"
                oracle_rerank_prompts.append({
                    "doc": doc, 
                    "oracle_rerank_prompt": oracle_rerank_prompt
                } | gen_meta_data)

        # Compute logprobs over answer choices 
        gen_kwargs = {
            "return_metadata": True,
            "max_output_length": 1
        }
        
        intermediate_rerank_metric_path = os.path.join(rc.intermediate_output_dir, task_name, "oracle_rerank_metrics.jsonl")
        if os.path.exists(intermediate_rerank_metric_path):
            oracle_rerank_outputs = load_jsonl(intermediate_rerank_metric_path)
        else:
            oracle_rerank_outputs = generate(model_name=rc.reranker_model, 
                                            model_type=rc.model_type, 
                                            prompts_and_metadata=oracle_rerank_prompts,
                                            prompt_key='oracle_rerank_prompt',
                                            gen_output_key='oracle_rerank',
                                            mode=mode,
                                            mode_args={"answer_choices": ANSWER_CHOICES} if mode=="mc" 
                                                        else {"prompt_suffix_key": "ground_truth"},
                                            llm=llm,
                                            tokenizer=tokenizer,
                                            **gen_kwargs)

            write(intermediate_rerank_metric_path, oracle_rerank_outputs)
            
        print("Reorganizing oracle_rerank metadata...")
        # Align results from rerank metric computation with that of the original input queries
        unranked_id_to_doc = defaultdict(list)      
        for dt in oracle_rerank_outputs:
            gk = dt[rc.retrieval_id_key]
            # We have a +1 to account for the oracle rerank metric computation over the original lm-only eval prompt
            if gk not in unranked_id_to_doc.keys() or len(unranked_id_to_doc[gk]) < rc.top_k_rerank + 1:
                unranked_id_to_doc[gk].append(dt)

        # Sanity check, we should have the same queries as keys and the same number of retrieved docs 
        for id_, _ in unranked_id_to_doc.items():
            assert id_ in id_to_docs.keys(), f"Missing id: {id_}"

        print("Reranking retrieved docs per query by oracle logprob delta...")
        new_score_key = rc.new_score_key if rc.new_score_key else rc.retrieval_score_key
        if mode == "mc":
            for id_, data in unranked_id_to_doc.items():
                original_query = data[0]
                assert original_query["doc"] is None, "Ordering of the retrieved contexts with oracle rerank metadata is incorrect."
                
                query_correct_answer = f" {original_query['ground_truth']}"
                
                if query_correct_answer not in dt["oracle_rerank"]["choice_log_probs"]:
                    logger.warning("Token id specified within allowed token ids is not seen in output distribution")
                    doc_correct_answer_logprob = -9999
                else:
                    orig_correct_answer_logprob = original_query["oracle_rerank"]["choice_log_probs"][query_correct_answer]["logprob"]
                orig_retrieved_docs = id_to_docs[id_]

                reranked_data = []
                for i, dt in enumerate(data[1:]):
                    # We should be able to find the retrieved document at the same index in the original query_to_docs map
                    assert orig_retrieved_docs[i][rc.retrieval_text_key][:rc.retrieval_trunc_len] == dt["doc"][rc.retrieval_text_key][:rc.retrieval_trunc_len]

                    # Compute delta logprob for correct answer.
                    # logprobs are not negated, so a positive difference
                    # between oracle logprob for current doc and the oracle
                    # logprob for the original q+a is desired.
                    if query_correct_answer not in dt["oracle_rerank"]["choice_log_probs"]:
                        doc_correct_answer_logprob = -9999
                    else:    
                        doc_correct_answer_logprob = dt["oracle_rerank"]["choice_log_probs"][query_correct_answer]["logprob"]
                    delta_logprob = doc_correct_answer_logprob - orig_correct_answer_logprob

                    orig_retrieved_doc = orig_retrieved_docs[i]
                    orig_retrieved_doc["original_logprobs"] = original_query["oracle_rerank"]["choice_log_probs"]
                    orig_retrieved_doc["retrieved_doc_logprobs"] = dt["oracle_rerank"]["choice_log_probs"]
                    orig_retrieved_doc[new_score_key] = delta_logprob
                    
                    reranked_data.append(orig_retrieved_doc)

                reranked_data, _ = sort_dicts_by_key(reranked_data, [new_score_key, rc.retrieval_score_key], reversed=True)
                reranked_id_to_doc[id_] = reranked_data
        elif mode == "prompt_logprobs":
            # Rerank by the mean pooled logprobs outputed from the generate function
            print("reranking retrieved docs per query by oracle mean prompt logprob delta...")
            for id_, data in unranked_id_to_doc.items():

                # data[0] is the original query without retrieval document
                original_query = data[0]
                assert original_query["doc"] is None, "Ordering of the retrieved contexts with oracle rerank metadata is incorrect."

                orig_gt_prompt_logprob = original_query["oracle_rerank"]["mean_logprob"]
                orig_retrieved_docs = id_to_docs[id_]

                reranked_data = []
                for i, dt in enumerate(data[1:]):
                    doc_mean_prompt_logprob = dt["oracle_rerank"]["mean_logprob"]
                    delta_logprob = doc_mean_prompt_logprob - orig_gt_prompt_logprob
                    
                    orig_retrieved_doc = orig_retrieved_docs[i]
                    orig_retrieved_doc["original_logprobs"] = original_query["oracle_rerank"]["raw_qa_logprobs"]
                    orig_retrieved_doc["retrieved_doc_logprobs"] = dt["oracle_rerank"]["raw_qa_logprobs"]
                    orig_retrieved_doc[new_score_key] = delta_logprob
                    
                    reranked_data.append(orig_retrieved_doc)

                reranked_data, _ = sort_dicts_by_key(reranked_data, [new_score_key, rc.retrieval_score_key], reversed=True)
                reranked_id_to_doc[id_] = reranked_data
        else:
            raise NotImplementedError("Oracle rerank only supports MC mode for now.")
        
    
    # Store the reranked query_to_doc mapping
    intermediate_reranked_query_to_doc_path = os.path.join(rc.intermediate_output_dir, "oracle_reranked_query_to_doc.pkl")
    write(intermediate_reranked_query_to_doc_path, reranked_id_to_doc, mode='pkl')

    print(len(reranked_id_to_doc))  
    out = []
    for dt in tqdm(retrieval_results, desc="Adding reranked queries to retrieval results"):
        # Get associated top-k docs
        id_ = dt[rc.retrieval_id_key]
        docs = reranked_id_to_doc[id_]
        dt[rc.ctx_key] = docs
        
        out.append(dt) 

    return out


