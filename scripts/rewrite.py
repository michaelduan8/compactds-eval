'''
Script to prepare data with retrieval results for evaluation, including reranking options
'''
import os
import random
import json
import time

from collections import defaultdict
from pathlib import Path
from simple_parsing import ArgumentParser
from tqdm import tqdm

from src.rerank.config import (
    OracleRerankConfig,
    RubricAnnotateConfig
)
from src.utils import (
    load_json,
    load_jsonl,
    write
)
from modules.generator.generator import Generator

def load_rerank_config(method, defaults_path, overwrites, parser): 
    config_class = RubricAnnotateConfig
    default_config = config_class.load(defaults_path, drop_extra_fields=False)

    # Handle commandline overwrites
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(config_class, dest="final_rerank_config", default=default_config)
    args = parser.parse_args(overwrites)
    rerank_config = args.final_rerank_config
    print(rerank_config)

    return rerank_config

def main(args, remaining_argv, parser):
    random.seed(1)

    retrieval_results = args.retrieval_results
    output_path = args.output_path
    output_mode = args.output_mode

    method = args.method
    rerank_config_file = args.rerank_config
    # rubric_file = args.rubric_file

    n_samples = args.n_samples
    sampling_key = args.sampling_key

    # n_manual_samples = args.n_manual_samples
    # grouping_key = args.grouping_key
    # manual_sample_k = args.manual_sample_k

    # Load rerank config
    rerank_config = load_rerank_config(method, 
                                       rerank_config_file, 
                                       remaining_argv,
                                       parser)

    total = load_jsonl(retrieval_results)
    
    # Load task config if specified
    if args.task_config_paths:
        # construct a set of selected queries with key of ('subject', 'index')
        selected_task_queries_indices = set()
        for task_config_path in args.task_config_paths:
            selected_task_queries = load_jsonl(load_json(task_config_path)['dataset_path'])
            selected_task_queries_indices.update([(q['subject'], q['index']) for q in selected_task_queries])
        print(f"Before the filtering: {len(total)}")
        total = [dt for dt in total if (dt['subject'], dt['index']) in selected_task_queries_indices]
        print(f"After the filtering: {len(total)}")


    if n_samples:
        if sampling_key:
            # Sample n_samples per group
            dt_by_sampling_key = defaultdict(list)
            for dt in total:
                sampling_key_group = dt[sampling_key]
                dt_by_sampling_key[sampling_key_group].append(dt)

            subset_total = []
            for _, data in dt_by_sampling_key.items():
                random.shuffle(data)
                subset_total.extend(data[:n_samples])

            total = subset_total
            
        else:
            random.shuffle(total)
            total = total[:n_samples]
    
    if method == "q-dep-no-option-minimal":
        instruction = 'Rewrite the following paragraph to be informative, educational, and helpful for answering the question: "{}". Make sure the paragraph is clear, well-structure, and concise (less than 300 words).\n\nOriginal paragraph: {}\n\nRewritten paragraph:'
    elif method == "q-indep-minimal":
        instruction = 'Rewrite the following paragraph to be informative and educational, clear and well-structured, and concise (less than 300 words).\n\nOriginal paragraph: {}\n\nRewritten paragraph:'
    else:
        raise NotImplementedError()

    all_prompts = []
    for dp in total:
        for ctx in dp["ctxs"][:rerank_config.top_k_rerank]:
            if method.startswith("q-dep-"):
                all_prompts.append(instruction.format(dp["question"], ctx["retrieval text"]))
            elif method.startswith("q-indep-"):
                all_prompts.append(instruction.format(ctx["retrieval text"]))
            else:
                raise NotImplementedError()

    rc = rerank_config
    llm = Generator(rc.reranker_model, rc.model_type)
    llm.get_model().load_model()
    
    start_time = time.time()
    raw_outputs = llm(all_prompts, max_output_length=384)
    print (f"Took {time.time()-start_time} secs to process {len(all_prompts)} inputs")

    assert len(raw_outputs)==len(all_prompts)
    
    out = []
    offset = 0
    for dp in total:
        ctxs = []
        for ctx in dp["ctxs"][:rerank_config.top_k_rerank]:
            rewritten = raw_outputs[offset]["output"]
            offset += 1
            new_ctx = {key: value for key, value in ctx.items()}
            new_ctx["original_text"] = ctx["retrieval text"]
            new_ctx["retrieval text"] = rewritten.split("\n\n")[0]
            new_ctx["raw_outputs_from_rewriter"] = rewritten
            ctxs.append(new_ctx)

        dp["rewritten_ctxs"] = ctxs
        out.append(dp)

    assert offset==len(raw_outputs)

    # write out
    write(output_path, out, output_mode)

if __name__=='__main__':
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--retrieval_results', type=str, default=None, help='Path to JSONL file with raw queries and answers and the associated retrieval results (AWS supported)')

    parser.add_argument('--output_path', type=str, help='Output path for JSONL matching input retrieval results as well as containing the top-k reranked results')
    parser.add_argument('--output_mode', type=str, default='jsonl', choices=['jsonl', 'csv'], help="Specify format to output prepared data")
    # parser.add_argument('--intermediate_output_path', type=str, help='Path to save intermediate outputs, e.g. computed rerank metrics')

    # Preparation
    # methods = ['oracle_rerank', 'rubric_annotate'] #['rubric_annotate', 'oracle_rerank', 'upr', 'manual']
    # parser.add_argument('--method', type=str, choices=methods, help='Specify method to rerank retrieval results')
    parser.add_argument('--method', type=str, help='Specify method to rerank retrieval results')

    parser.add_argument("--rerank_config", help="Path to rerank config file", type=Path)
    parser.add_argument('--task_config_paths', nargs='+', default=None, help='Paths to task config files. If not specified, will use all the retrieval results.')

    # # TODO: currently only supports loading one rubric
    # parser.add_argument('--rubric_file', type=str, help='When performing LLM-as-judge annotation, this file contains the annotation rubric, or is a JSON file mapping a grouping key to specific rubrics')

    # Testing params
    parser.add_argument('--n_samples', type=int, default=None, help='Number of sampels to randomly take from complete retrieval results')
    parser.add_argument('--sampling_key', type=str, default=None, help='When defined with n_sample, take a sample of n_sample rows per sampling_key group.')

    # Sampling for manual eval/inspection
    # parser.add_argument('--n_manual_samples', type=int, default=3)
    # parser.add_argument('--grouping_key', type=str, default="subject")
    # parser.add_argument('--manual_sample_k', type=int, default=3, help="Return a random sample of specified size from top-k retrieval docs. The rank 1 document is always returned and not included in the sample count.")

    args, remaining_argv = parser.parse_known_args()

    main(args, remaining_argv, parser)
