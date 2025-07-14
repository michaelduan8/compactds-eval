# CompactDS Evaluation

## Citation
```
@article{lyu2025compactds,
  title={Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks},
  author={Xinxi Lyu and Michael Duan and Rulin Shao and Pang Wei Koh and Sewon Min}
  journal={arXiv preprint arXiv:2507.01297},
  year={2025}
}
```

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Reranking Retrieval Results](#reranking-retrieval-results)
4. [Collecting Retrieval Results](#collecting-retrieval-results)

## Installation

Create a conda environment and run:
<!-- TODO: need to create a req.txt file -->
```
pip install -e olmes
pip install torch transformers datasets pipeline smart_open pyserini vllm==0.8.4 pydantic simple_parsing texttable huggingface_hub
```

Export your HF access token (for gated HF models)
```bash
export HF_TOKEN={your_hf_token}
```

<!-- Currently, generation is primarily done with `vllm`. Generation with other modes such as `hf` are not fully supported. -->

## Quick Start
### No Retrieval Baseline
Our code builds off OLMES task configs. For the benchmarks used in the paper, the value for each you should specify under `--task` is as follows.
- MMLU: `mmlu:mc::retrieval`
- MMLU Pro: `mmlu_pro:mc::retrieval`
- AGI Eval: `agi_eval_english::retrieval`
- GPQA: `gpqa:0shot_cot::retrieval`
- Minerva Math: `minerva_math::retrieval`

Let's first run the model without retrieval as a baseline. Here is an example command with MMLU Pro (MC).
```bash
python olmes/oe_eval/run_eval.py \
	--task mmlu_pro:mc::retrieval \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--save-raw-requests true \
	--model-type hf \ 
	--random-subsample-seed 2025 \
	--model-args '{"max_length": 16384}' \
	--output-dir output/llama-8B-no-retrieval \
	--llm_only
```
- To run with vllm, change `model-type` to `vllm`.
- Results are saved under `output/llama-8B-no-retrieval`.

<!-- (TODO: how to get the numbers you need for all datasets, including per-subject numbers) -->
Now with the results for MMLU Pro, observe the metrices using the following command.
```bash
python scripts/aggregate_eval_results.py --result_file_dir output --output_dir metrics
```
This script also handles computing scores for specific benchmark subcategories, namely MMLU STEM, Social Sciences, Humanities, Other; and GPQA Physics, Biology, and Chemistry.

Metrics are saved in `metrics/llama-8B-no-retrieval/results.json`.

If you run another model/method, save it in the same parent directory but name it differently (e.g., `output/llama-70B-no-retrieval`) in order to observe the metrics of multiple models/methods concurrently. The combined results are saved in `metrics/results.csv`.


### Running a Model with RAG
Now, let's run the model with retrieval. Our system consists of three steps: Dense retrieval -> (Optional) Reranking -> Generation (eval). Retrieval results are prepended in-context before generation. This repo doesn't include the code for local dense retrieval; instead, retrieval results will be provided in the form of JSONL files, with each line containing retrieved contexts for a specific query. To download example retrieval files, use the following script.
```bash
PYTHONPATH=. python scripts/download_hf_dataset.py --dataset_name alrope/CompactDS-102GB-retrieval-results --output_path datastore/
```
It should save a collection of per-benchmark retrieval results files, such as `datastore/mmlu_pro_exact_search_results.jsonl`.

Now, let's run generation with the MMLU Pro retrieval results.
```bash
python olmes/oe_eval/run_eval.py \ 
	--task mmlu_pro:mc::retrieval \
	--retrieval_results_path datastore/mmlu_pro_exact_search_results.jsonl \ 
	--model meta-llama/Llama-3.1-8B-Instruct \ 
	--retrieval_config configs/eval/offline_retrieval.json \
	--matching_key id \
	--save-raw-requests true 
	--ctx_key ctxs \
	--model-type hf \
	--random-subsample-seed 2025 \
	--model-args '{"max_length": 16384}' \ 
	--output-dir output/llama-8B-k=3 \ 
	--k 3
```
If using `model-type=vllm`, you can also specify the vllm parameter `gpu_memory_utilization` in `model-args` to best utilize your GPU memory. See the official [OLMES repo](https://github.com/allenai/olmes) for more comprehensive usage details.

You can also vary the value for `k` (the number of passages to feed to the LM).
Whenever you change the method (either the value for `k` or the model), make sure to change `--output-dir`. (It's OK to use the same directory if you are applying the same model/method but only for different tasks).

Useful command-line arguments for RAG (can also be specified in retrieval config):
- `matching_key`: Key to match retrieval results with its respective query. Ensure this is unique across queries and this key is shared by both the benchmark queries and retrieval results.
- `presort_key`: Key for pre-sorting retrieval results before applying top-K reranking
- `sort_key`: Key for sorting retrieval results. This can either be direct key tied to a score or a special key tied to a custom scoring interpolation method
- `rerank_k`: Specify the number of top retrieval results per query for each `retrieval_result_path` JSONL file after `presort_key` sort is applied for top-K reranking with `sort_key`

See the section [Running Embedding Reranking](#running-embedding-reranking) for an example running with GRIT-reranked scores.

## Reranking Retrieval Results

We support three main reranking methods currently: **embedding-based**, **rubric-based**, and **oracle**.

### Running Embedding Reranking

The following is an example of using the similarity score between embeddings of the query and the documents for reranking, with GRIT-7B as the embedder model:
```
python scripts/rerank.py \
	--task_name mmlu_pro:mc::retrieval \
	--retrieval_results datastore/mmlu_pro_exact_search_results.jsonl \
	--output_path reranked_outputs/mmlu_pro/single_v3_grit_reranked_k_100.jsonl \
	--method embedding_rerank \
	--rerank_config configs/rerank/grit_embed_rerank_no_chunk.json \
	--intermediate_output_dir intermediate_outputs/mmlu_pro/single_v3_grit_reranked_k_100/ \
	--top_k_rerank 100
```
This reranks the top 100 retrieved results per query, sorted initially by the retrieval score under the key `retrieval score` (default). 

Currently, we support embedding with Contriever, GRIT-7B, and ReasonIR.

Other arguments of note:
- `chunk_size`: If retrieval documents are not chunked, or documents want to be split further, chunk_size can be specified to to chunk by specified **word count** (defined on whitespace).


Once embedding reranking results are saved, now you can run the eval. Here is an example with GRIT-reranked results (this format generally applies to all reranking methods).
```bash
python olmes/oe_eval/run_eval.py \
	--task mmlu_pro:mc::retrieval \
	--retrieval_results_path reranked_outputs/mmlu_pro/single_v3_grit_reranked_k_100.jsonl \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--retrieval_config configs/eval/offline_retrieval.json \
	--matching_key id \
	--save-raw-requests true \
	--ctx_key ctxs \
	--model-type vllm \
	--random-subsample-seed 2025 \
	--presort_key retrieval_score \
	--sort_key grit_score \
	--k 3 \
	--rerank_k 100 \
	--model-args '{"gpu_memory_utilization": 0.5, "max_length": 16384}' \ 
	--output-dir output/llama-8B-grit-K=100-k=3
```

### Running Rubric Reranking
We also support reranking using rubric scoring via LLM-as-judge. Use the following command to rerank 100 passages with Llama 3.1 8B Instruct and the rubric at `src/rerank/rubrics/rubric_v0.4.txt` (as specified in the provided config):
```bash
python scripts/rerank.py \
	--task_name mmlu_pro:mc::retrieval \
	--retrieval_results datastore/mmlu_pro_exact_search_results.jsonl \
	--output_path reranked_outputs/mmlu_pro/single_v3_reranked_k_100.jsonl \
	--method rubric_annotate \
	--rerank_config configs/rerank/mmlu_rubric_annotation_llama_3.1_0.4.json \
	--intermediate_output_dir intermediate_outputs/mmlu_pro/single_v3_rubric_reranked_k_100/ \
	--top_k_rerank 100
```
By default, rubric scores are stored under the key `rubric score` per retrieval result.
<!-- After reranking results are saved, you can now run the eval with top 3 passages.
```
python olmes/oe_eval/run_eval.py \ 
	--task mmlu_pro:mc::retrieval \
	--retrieval_results_path reranked_outputs/mmlu_pro/single_v3_reranked_k_100.jsonl \
	--model meta-llama/Llama-3.1-8B-Instruct \ 
	--retrieval_config configs/eval/offline_retrieval.json \
	--matching_key id \
	--save-raw-requests true 
	--ctx_key ctxs \
	--rerank_mode retrieval_score+rubric_score \
	--model-type vllm \
	--random-subsample-seed 2025 \
	--model-args '{"gpu_memory_utilization": 0.5, "max_length": 16384}' \ 
	--output-dir output/llama-8B-K=100-k=3
``` -->

<!-- Some useful parameters to change:
- `top_k_rerank`: number of retrieved documents to oracle rerank for each query. Default to 100. -->

### Running Oracle Reranking

Finally, we support oracle reranking, which computes the logprob increase of the correct answer when a specific retrieval document is and isn't provided in-context.

Note that oracle reranking is supported for **MC tasks only** (MMLU, MMLU Pro, AGI Eval) for now.

To perform oracle reranking:
```
python scripts/rerank.py \ 
    --task_name mmlu_pro:mc::retrieval \
	--retrieval_results datastore/mmlu_pro_exact_search_results.jsonl \
	--output_path reranked_outputs/mmlu_pro/single_v3_oracle_reranked_k_100.jsonl \
    --method oracle_rerank \ 
	--rerank_config configs/rerank/olmes_oracle_rerank.json \ 
	--intermediate_output_dir intermediate_outputs/mmlu_pro/single_v3_oracle_reranked_k_100/ \
	--top_k_rerank 100 
```

The `task_name` parameter must be specified for oracle reranking to gather the ground truth targets per query for computing oracle score.


<!-- Once oracle reranking results are saved, now you can run the eval.
```bash
python olmes/oe_eval/run_eval.py \ 
	--task mmlu_pro:mc::retrieval \
	--retrieval_results_path reranked_outputs/mmlu_pro/single_v3_oracle_reranked_k_100.jsonl \
	--model meta-llama/Llama-3.1-8B-Instruct \ 
	--retrieval_config configs/eval/offline_retrieval.json \
	--matching_key id \
	--save-raw-requests true 
	--ctx_key oracle_reranked_ctxs \
	--model-type vllm \
	--random-subsample-seed 2025 \
	--model-args '{"gpu_memory_utilization": 0.5, "max_length": 16384}' \ 
	--output-dir output/llama-8B-oracle-K=100-k=3
``` -->

#### Important Arguments 
These are parameters to consider across all reranking methods:
- `new_score_key`: Set the key that the rerank score will be stored under. If not set, `new_score_key` defaults to `retrieval_score_key` (typically `retrieval score`).

## Collecting Retrieval Results
### Preparing Retrieval Queries
Search queries are formatted in JSONL files with the following format:
```
{"query": ..., ..., "id": ...}
{"query": ..., ..., "id": ...}
{"query": ..., ..., "id": ...}
```
where `id` is a unique identifier for each search query and `query` is the text used for retrieval. To build queries from OLMES tasks, the following script can be used: 
```bash
python src/preprocessing/prepare_retrieval_queries_from_olmes.py \ 
	--retrieval_key query \
	--task mmlu_pro:mc::retrieval \
	--output_dir retrieval_queries_out/ \
	--method q
```
This prepares retrieval queries using the OLMES-formatted queries for MMLU Pro (for task config `mmlu_pro:mc::retrieval`). The resulting prepared queries will be stored under `retrieval_queries_out/mmlu_pro:mc::retrieval_q.jsonl`

Important arguments:
- `task`: The OLMES task (by task config name) to build retrieval queries for.
- `method`: The retrieval query preparation method, one of [`q`, `q+a`, `break_down`]
	- `model`: If using the `break_down` method, specify the model name to load to perform query decomposition.


### Local Dense Retrival
Refer to our [datastore construction repository](https://github.com/Alrope123/retrieval-scaling/tree/xinxil) for instructions on building a local datastore (e.g., CompactDS) and collecting retrieval results.

### Search Engine Retrieval
#### Querying with Google Custom Search Engine
Export your Google Custom Search Engine ID and API Key
```bash
export GOOGLE_CSE_ID={google_cse_id}
export GOOGLE_API_KEY={google_api_key}
```

With the [prepared retrieval queries](preparing-retrieval-queries), run web retrieval with the following command. This gathers the top-10 web results per query (retrying once if any queries have search failures). We use the example prepared queries from [earlier](#preparing-retrieval-queries).
```
PYTHONPATH=. python scripts/web_retrieval/web_retrieval.py \
	--retrieval_queries retrieval_queries_out/mmlu_pro:mc::retrieval_q.jsonl \
	--k 10 \
	--session_name mmlu_pro_v0 \
	--intermediate_output_dir web_retrieval_intermediate_out/ \
	--output_dir web_retrieval_out/ \
	--num_tries 1 \
	--load_pdfs \
	--load_webpages
```
Note that this code will stop early after PDFs are downloaded locally. Use [olmOCR](https://github.com/allenai/olmocr) to linearize these PDFs and place them under the path `{intermediate_output_dir}/{session_name}_top{k}/ours/parsed_pdfs`. In the example above, this path would be `web_retrieval_intermediate_out/mmlu_pro_v0_top10/ours/parsed_pdfs`.

Afterwards, rerun the above command to yield the parsed retrieval results under the folder `{output_dir}/{session_name}_top{k}/ours/`. In the example above, the output dir and contained results file would be: `web_retrieval_out/mmlu_pro_v0_top10/ours/mmlu_pro:mc::retrieval_q_web_retrieval_results.jsonl`.

Parsed web results are typically quite long, so a chunking strategy is recommended (e.g., as used in [embedding reranking](running-embedding-reranking)).

Important Arguments:
- `k`: The number of retrieval results per query. Note that Google CSE (as with most search engines) yields results in pages of fixed size, in this case, 10 results per page at one page per query. For `k > 10`, this means search engine costs will, at minimum, double.
- `web_parse_mode`: By default, this is set as `default`, which uses resiliparse for webpage parsing. However, one can also specify `crawl4ai` as an alternative open-source, dynamic parser or `jina` to use JINA AI Reader API for parsing web results. If using `jina`, make sure to run: `export JINA_API_KEY={jina_api_key}`.
- `simple`: Flag to use an alternative PDF parsing method that uses PDFPlumber for less comprehensive PDF parsing. It also only reads the first 600 words of the PDF for quicker parsing. Specifying this flag will also alter the path under which intermediate and final outputs are saved (`ours` is replaced with `simple` in path names).
- `max_concurrency`: Specify the maximum number of concurrent sessions for asynchronous requesting.
- `max_workers`: Specify the maximum number of workers to use in multithreading.

### Deduplication
On-the-fly deduplication can be applied through the following script:
```
PYTHONPATH=. python scripts/dedupe_retrieval_result.py \
	--retrieval_results_paths web_retrieval_out/mmlu_pro_v0_top10/ours/mmlu_pro:mc::retrieval_q_web_retrieval_results.jsonl \
	--normalize \
	--paragraph_delimiter \n\n \
	--output_dir deduped_retrieval_results/ \
	--dedupe_key query
```

This applies a paragraph-level (where paragraphs are delimited by the `paragraph_delimiter`) decontamination for each retrieval result per specified query (under key `dedupe_key`) using a longest shared subsequence and jaccard similarity threshold filters. We use the example web retrieval results from the prior section.

Important arguments:
- `retrieval_results_paths`: Specify a list of retrieval result JSONL files for deduplication
- `dedupe_key`: Specify the key per retrieval result to use for decontamination
- `normalize`: Flag to apply Contriever-style text normalization.
- `jaccard_threshold`: Specify Jaccard similarity threshold. This uses 13-gram jaccard similarity
- `ls_threshold`: Specify the length threshold for the longest shared subsequence.


