CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="high"

#retrieval_results_path="/home/sewonm/datastores/combined/retrieval_v3_q_retrieved_results.jsonl"
#eval_output_dir=output

retrieval_results_path="/weka_data/xinxil/private-retrieval-lm/ours_v1/combined/retrieval_v3_q_retrieved_results.jsonl"
#eval_output_dir=/weka_data/seownm/private-retrieval-lm/output
eval_output_dir="s3://ai2-llm/pretraining-data/sources/ds-olmo-data/sewonm/private-retrieval-lm/output"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

METHOD=$1
TASK=$2


if [[ $METHOD == "single" ]] ; then
	#retrieval_results_path="/weka_data/xinxil/private-retrieval-lm/ours_v1/massiveds-math-no-gsm8k/retrieved_results/facebook/contriever-msmarco/massiveds-math-no-gsm8k_datastore-256_chunk_size/top_100_IVFPQ.2048.256.256/retrieval_v3_q_retrieved_results_reranked.jsonl"
	
	retrieval_results_path="/weka_data/xinxil/private-retrieval-lm/ours_v1/retrieved_results/_IVFPQ.2048.256.256/retrieval_v3_q_retrieved_results_rubric_reranked.jsonl"

	command="python olmes/oe_eval/run_eval.py --task $TASK \
		--retrieval_results_path $retrieval_results_path \
		--model $MODEL --max-length 16384 \
		--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
		--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
		--output-dir $eval_output_dir/turbov2_single"

elif [[ $METHOD == "multi" ]] ; then
	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/turbov2"

	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/turbov2_dynamic \
	#	--rerank_mode dynamic"

	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/turbov2_dynamic_th2 \
	#	--rerank_mode dynamic --threshold 2"

	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/turbov2_dynamic_100 \
	#	--rerank_mode dynamic_100"

	command="python olmes/oe_eval/run_eval.py --task $TASK \
		--retrieval_results_path $retrieval_results_path \
		--model $MODEL --max-length 16384 \
		--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
		--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
		--output-dir $eval_output_dir/turbov2_uniform \
		--rerank_k 25"

elif [[ $METHOD == "multi-long" ]] ; then
	command="python olmes/oe_eval/run_eval.py --task $TASK \
		--retrieval_results_path $retrieval_results_path \
		--model $MODEL --max-length 128000 \
		--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
		--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
		--output-dir $eval_output_dir/turbov2_dynamic_k20 \
		--rerank_mode dynamic --k 20"

elif [[ $METHOD == "multi-grit-math" ]] ; then
	retrieval_results_path="/weka_data/xinxil/private-retrieval-lm/ours_v1/combined/retrieval_v3_q_retrieved_results_w_grit_math.jsonl"
	command="python olmes/oe_eval/run_eval.py --task $TASK \
		--retrieval_results_path $retrieval_results_path \
		--model $MODEL --max-length 32768 \
		--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
		--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
		--output-dir $eval_output_dir/turbov2_grit_math_dynamic_k10 \
		--rerank_mode dynamic --k 10"

elif [[ $METHOD == "llama-70B" ]] ; then
	MODEL="meta-llama/Llama-3.3-70B-Instruct"
	
	command="python olmes/oe_eval/run_eval.py --task $TASK --model $MODEL --save-raw-requests true --model-type vllm --random-subsample-seed 2025 --output-dir $eval_output_dir/llama70B --llm_only --max-length 16384"

	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/llama70B_turbov2_grit_math_dynamic_k10 \
	#	--rerank_mode dynamic --k 10"

elif [[ $METHOD == "qwq" ]] ; then
	MODEL="Qwen/QwQ-32B"
	
	command="python olmes/oe_eval/run_eval.py --task $TASK --model $MODEL --save-raw-requests true --model-type vllm --random-subsample-seed 2025 --output-dir $eval_output_dir/QwQ32B --llm_only --max-length 16384"

	#command="python olmes/oe_eval/run_eval.py --task $TASK \
	#	--retrieval_results_path $retrieval_results_path \
	#	--model $MODEL --max-length 16384 \
	#	--retrieval_config configs/eval/offline_retrieval.json --matching_key id --save-raw-requests true --ctx_key rubric_annotated_ctxs --model-type vllm --random-subsample-seed 2025 \
	#	--sources_to_filter rpj_arxiv,rpj_book,rpj_github,reddit_ai2 \
	#	--output-dir $eval_output_dir/QwQ32B_turbov2_grit_math_dynamic_k10 \
	#	--rerank_mode dynamic --k 10"

else
	echo "Invalid method $METHOD"
	exit
fi

TASK=$(echo "$TASK" | cut -d ':' -f 1)
gantry run \
    --task-name "Scaling-RAG_${TASK}_${METHOD}" \
    --description "Scaling-RAG_${TASK}_${METHOD}" \
    --workspace ai2/ds-olmo \
    --allow-dirty \
    --beaker-image 'lucas/refine1' \
    --no-logs \
    --timeout 0 \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 4 \
    --replicas 1 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=SEWONM_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=SEWONM_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
    --env-secret HF_TOKEN=SEWONM_HF_TOKEN \
    --install "pip install -e olmes[gpu] && pip install simple_parsing && pip install --force-reinstall -U --no-deps -v peft==0.14.0" \
    --shared-memory 10GiB \
    --weka oe-data-default:/weka_data \
    --yes \
    -- /bin/bash -c "export VLLM_LOGGING_LEVEL=ERROR && $command"


