'''
Script to prepare data with retrieval results for evaluation, including reranking options
'''
import os
import random

from collections import defaultdict
from pathlib import Path
from simple_parsing import ArgumentParser

from olmes.oe_eval.launch import resolve_task_suite
from src.rerank.config import (
    EmbeddingRerankConfig,
    OracleRerankConfig,
    RubricAnnotateConfig
)
from src.utils import (
    load_json,
    load_jsonl,
    write
)


METHOD_TO_CONFIG = {
    "embedding_rerank": EmbeddingRerankConfig,
    "oracle_rerank": OracleRerankConfig,
    "rubric_annotate": RubricAnnotateConfig
}


def partition_retrieved_results(total_retrieved_results):
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", 0))
    world_size = int(os.environ.get("BEAKER_REPLICA_COUNT", 1))
    
    print (f"Distributing {len(total_retrieved_results)} entries into world_size={world_size}")
    entries_per_process = len(total_retrieved_results) / world_size
    start_idx = int(rank * entries_per_process)
    end_idx = int((rank + 1) * entries_per_process) if rank < world_size - 1 else len(total_retrieved_results)
    partition_retrieved_results = total_retrieved_results[start_idx:end_idx]
    # partition_file_sizes = file_sizes[start_idx:end_idx]
    print(f"This worker (rank {rank}) handling entries: {start_idx} to {end_idx-1}")
    return partition_retrieved_results, rank


def load_rerank_config(method, defaults_path, overwrites, parser): 
    config_class = METHOD_TO_CONFIG[method]
    default_config = config_class.load(defaults_path, drop_extra_fields=False)

    # Handle commandline overwrites
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(config_class, dest="final_rerank_config", default=default_config)
    print(f"Default config: {default_config}")
    args = parser.parse_args(overwrites)
    print(f"Overwrites: {overwrites}")
    print(f"Final config: {args.final_rerank_config}")
    rerank_config = args.final_rerank_config
    print(rerank_config)

    return rerank_config

def main(args, remaining_argv, parser):
    random.seed(1)

    print("Args:")
    print(args)
    print("Remaining args:")
    print(remaining_argv)

    retrieval_results = args.retrieval_results
    output_path = args.output_path
    output_mode = args.output_mode
    task_name = args.task_name

    method = args.method
    rerank_config_file = args.rerank_config

    n_samples = args.n_samples
    sampling_key = args.sampling_key

    # Load rerank config
    if method != "bm25_rerank":
        # TODO: I don't think this condiitional block is needed, but will double check
        if method == "oracle_rerank":
            remaining_argv.append(f"--task_name")
            remaining_argv.append(f"{task_name}")
        rerank_config = load_rerank_config(method, 
                                       rerank_config_file, 
                                       remaining_argv,
                                       parser)
    
    total = load_jsonl(retrieval_results)
    rank = None

    if task_name is not None:
        task_names = resolve_task_suite(task_name, {})
        total = [r for r in total if ":".join(r['id'].split(":")[:-1]) == task_name or ":".join(r['id'].split(":")[:-1]) in task_names]

    if int(os.environ.get("BEAKER_REPLICA_COUNT", 0)) > 1:
        total, rank = partition_retrieved_results(total)

    print(f"Total number of entries: {len(total)}")

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

    if method == "embedding_rerank":
        from src.rerank.embedding_rerank import embedding_rerank
        # Load standard rerank config
        out = embedding_rerank(total, rerank_config)
    elif method == "rubric_annotate":
        from src.rerank.rubric_annotate import rubric_annotate
        # Load rubric_annotate config
        out = rubric_annotate(total, rerank_config)
    elif method == "oracle_rerank":
        # Load oracle_rerank config
        from src.rerank.oracle_rerank import oracle_rerank
        out = oracle_rerank(total, rerank_config)
    elif method == "bm25_rerank":
        from src.rerank.bm25_rerank import bm25_rerank
        out = bm25_rerank(total)

    # write out
    write(f"{output_path}_{rank}" if rank is not None and rank >= 0 else output_path, out, output_mode)
    

if __name__=='__main__':
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--task_name', type=str, default=None, help='Specify task or task suite to conduct reranking over')
    parser.add_argument('--retrieval_results', type=str, default=None, help='Path to JSONL file with raw queries and answers and the associated retrieval results (AWS supported)')

    parser.add_argument('--output_path', type=str, help='Output path for JSONL matching input retrieval results as well as containing the top-k reranked results')
    parser.add_argument('--output_mode', type=str, default='jsonl', choices=['jsonl', 'csv'], help="Specify format to output prepared data")

    # Preparation
    methods = ['embedding_rerank', 'oracle_rerank', 'rubric_annotate', 'bm25_rerank']
    parser.add_argument('--method', type=str, choices=methods, help='Specify method to rerank retrieval results')
    parser.add_argument('--task_config_paths', nargs='+', default=None, help='Paths to task config entries. If not specified, will use all the retrieval results.')
    parser.add_argument("--rerank_config", help="Path to rerank config file", type=Path)

    # Testing params
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to randomly take from complete retrieval results')
    parser.add_argument('--sampling_key', type=str, default=None, help='When defined with n_sample, take a sample of n_sample rows per sampling_key group.')

    args, remaining_argv = parser.parse_known_args()

    main(args, remaining_argv, parser)