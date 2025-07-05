
CLUSTER="ai2/augusta-google-1"
WORKSPACE="ai2/OLMo-modular"
PRIORITY="urgent"
# WORKSPACE="ai2/ds-olmo"
# PRIORITY="high"
#MODEL_NAME="allenai/OLMo-2-1124-7B"
MODEL_NAME="Qwen/QwQ-32B"
# MODEL_NAME="meta-llama/Llama-3.3-32B-Instruct"
# MODEL_NAME="princeton-nlp/Llama-3-8B-ProLong-512k-Instruct"

MODE=$1
NAME=$2
RETRIEVAL_PATH=$3
DOMAIN=$4
FILTER_OUT=$5
TASK="gpqa_diamond:0shot_cot::retrieval2"
# TASK="gpqa_diamond:0shot_cot::retrieval"

if [[ $MODE == "llm-only" ]] ; then
	# No-retrieval LM
	# command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task retrieval --limit 100 --random-subsample-seed 2025 --output-dir /results --save-raw-requests true --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.8}'"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/llm-only/${TASK} --save-raw-requests true --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.7, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results/downstream/eval-retrieval_32B/llm-only/$TASK --random-subsample-seed 2025"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/llm-only/${TASK} --save-raw-requests true --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results/downstream/eval-retrieval_32B/llm-only/$TASK --random-subsample-seed 2025"

elif [[ $MODE == "mds" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/$NAME --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME --random-subsample-seed 2025"
    # --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task retrieval --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$NAME --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --sources_to_keep rpj_c4"
    # 16384


elif [[ $MODE == "mds_batch" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 131072, \"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 131072, \"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10"

elif [[ $MODE == "mds_batch_batch" ]] ; then
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 65536, \"rope_scaling\": {\"factor\": 2.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10"


elif [[ $MODE == "mds_filter" ]] ; then
    python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/ours_v2_filter=$NAME --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{"gpu_memory_utilization": 0.5, "max_length": 16384}'  --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/ours_v2_filter=$NAME --k 3 --sources_to_filter $NAME --random-subsample-seed 2025


elif [[ $MODE == "rubric" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/$NAME --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.4, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME --random-subsample-seed 2025"


elif [[ $MODE == "rubric_batch_old" ]] ; then
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --rerank_k 100 --rerank_mode original_score+retrieval_score"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 128000}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 100 --rerank_mode original_score+retrieval_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds_k=100/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.7, \"max_length\": 256000}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds_k=100/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 100 rerank_k 100 --rerank_mode original_score+retrieval_score"

elif [[ $MODE == "rubric_batch" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 128000}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds_k=100/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.7, \"max_length\": 256000}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds_k=100/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 100"

elif [[ $MODE == "rubric_filter" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/$NAME --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME --sources_to_filter $DOMAIN --random-subsample-seed 2025"

elif [[ $MODE == "rubric_filter_batch" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --sources_to_filter $FILTER_OUT --random-subsample-seed 2025"

elif [[ $MODE == "rubric_keep" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $TASK --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/$NAME --save-raw-requests true --ctx_key rubric_annotated_ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 16384}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME --sources_to_keep $DOMAIN --random-subsample-seed 2025"

elif [[ $MODE == "grit-rubric_batch" ]] ; then
    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768, \"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

#     # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 131072, \"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

elif [[ $MODE == "grit_batch" ]] ; then
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 65536, \"rope_scaling\": {\"factor\": 2.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

elif [[ $MODE == "grit_batch_batch" ]] ; then
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 65536, \"rope_scaling\": {\"factor\": 2.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 1000 --rerank_mode retrieval_score+grit_score"

elif [[ $MODE == "grit-rubric_batch_batch" ]] ; then
    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 65536, \"rope_scaling\": {\"factor\": 2.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

    # command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 65536}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer/$TASK/$NAME/$DOMAIN-$FILTER_OUT --random-subsample-seed 2025 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

    command="python olmes/oe_eval/run_eval.py --model $MODEL_NAME --task $DOMAIN --retrieval_config configs/eval/offline_retrieval.json --matching_key id --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --save-raw-requests true --ctx_key ctxs --retrieval_results_path $RETRIEVAL_PATH --model-type vllm --model-args '{\"gpu_memory_utilization\": 0.5, \"max_length\": 32768}' --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/no_answer_k=10/$TASK/$NAME/$DOMAIN --random-subsample-seed 2025 --k 10 --rerank_k 1000 --rerank_mode grit_score+rubric_score"

elif [[ $MODE == "rubric_separate" ]] ; then
    gpus=1
    python olmes/oe_eval/launch.py \
        --model $MODEL_NAME \
        --task $TASK \
        --retrieval_config configs/eval/offline_retrieval.json \
        --matching_key id \
        --output-dir /weka_data/xinxil/private-retrieval-lm/eval_datasets/results_32B/mds/$TASK/$NAME \
        --save-raw-requests true \
        --ctx_key rubric_annotated_ctxs \
        --retrieval_results_path $RETRIEVAL_PATH \
        --model-type vllm \
        --model-args '{"gpu_memory_utilization": 0.5, "max_length": 16384}' \
        --random-subsample-seed 2025 \
		--use-gantry \
		--gpus $gpus \
		--cluster $CLUSTER \
		--beaker-workspace ai2/ds-olmo \
		--beaker-budget ai2/oe-data \
		--beaker-priority $PRIORITY \
		--gantry-secret-aws-access-key-id SEWONM_AWS_ACCESS_KEY_ID \
		--gantry-secret-aws-secret-access SEWONM_AWS_SECRET_ACCESS_KEY \
		--gantry-secret-hf-read-only SEWONM_HF_TOKEN \
		--gantry-args 'weka=oe-training-default:/weka_data,preemptible=True,allow_dirty=true,hf_token=true'
        # --remote-output-dir s3://ai2-llm/eval-results_32B/downstream/eval-retrieval_32B/mds/$TASK/$NAME \
    exit

else
	echo "Invalid $MODE"
	exit
fi

gantry run \
    --task-name "Evaluation" \
    --description "Evaluation_$MODEL_NAME" \
    --workspace $WORKSPACE \
    --allow-dirty \
    --beaker-image 'lucas/refine1' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 4 \
    --replicas 1 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --cluster "ai2/jupiter-cirrascale-2" \
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
    --yes \
   	-- /bin/bash -c "export VLLM_LOGGING_LEVEL=ERROR && $command"
    # --weka oe-data-default:/weka_data \
	# python olmes/oe_eval/run_eval.py --model meta-llama/Llama-3.1-8B-Instruct --task retrieval --limit 100 --random-subsample-seed 2025 --output-dir output --save-raw-requests true
