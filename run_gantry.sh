
CLUSTER="ai2/augusta-google-1"
# CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="normal"

#MODEL_NAME="allenai/OLMo-2-1124-7B"
#MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
#MODEL_NAME="Qwen/Qwen2.5-7B"
#MODEL_NAME="Qwen/Qwen2.5-72B"
# MODEL_NAME="meta-llama/Llama-3.1-70B"
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

#MODEL_NAME="allenai/Llama-3.1-Tulu-3-70B"
#MODEL_NAME="allenai/OLMo-2-1124-7B-Instruct"

k=3 #10

MODE=$1
RET_PATH=$2
RANK=$3
WORLD_SIZE=$4

# MMLU_MDS_RETRIEVAL_PATH="s3://ai2-llm/pretraining-data/sources/ds-olmo-data/oracle_retrieval/mmlu/out/full_subsampled_1_1000_dedup_merged_simple_qa_queries_top1000.jsonl"
MMLU_MDS_RETRIEVAL_PATH="s3://ai2-llm/pretraining-data/sources/ds-olmo-data/ours_v3_retrieval/ours_v2_1/mmlu_k=1000_massived_original.jsonl"
MMLU_MDS_ORACLE_RERANK_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_mds_oracle_rerank_reformatted.jsonl"
MMLU_LB_ORACLE_RERANK_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_lb_oracle_rerank.jsonl"

if [[ $MODE == "no-retrieval" ]] ; then
	# No-retrieval LM
	command="python scripts/eval.py --config configs/eval/lm_only.json --experiment_name lm_only_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json"

elif [[ $MODE == "cot" ]] ; then
	command="python scripts/eval.py --config configs/eval/lm_only.json --experiment_name mmlu_subset_lm_only_cot --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset_cot.json --max_output_length 1024"

elif [[ $MODE == "mds" ]] ; then

	# Retrieval w/o reranking
	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_RETRIEVAL_PATH --k $k --rerank_mode base --rerank_k 10000"


elif [[ $MODE == "cot-mds" ]] ; then

	# Retrieval w/o reranking
	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset_cot_retrieval.json --retrieval_results_path $MMLU_MDS_RETRIEVAL_PATH --k $k --max_output_length 1024 --augment_method prefix"

elif [[ $MODE == "rewrite" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_mds_rewritten_q-dep_llama70Binstruct_K=3.jsonl"

	# command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-dep-no-option-minimal"

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3 --ctx_key rewritten_ctxs"


elif [[ $MODE == "rewrite-cot" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_mds_rewritten_q-dep_llama70Binstruct_K=3.jsonl"

	# command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-dep-no-option-minimal"

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset_cot_retrieval.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3 --ctx_key rewritten_ctxs --max_output_length 1024 --augment_method prefix"


elif [[ $MODE == "rewrite-stem" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_stem_mds_rewritten_q-dep_llama70Binstruct_K=100.jsonl"

	command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --task_config_paths tasks/mmlu_STEM_subset.json --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 100 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-dep-no-option-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_STEM_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3 --ctx_key rewritten_ctxs"

	#####################################################################################

elif [[ $MODE == "rewrite-indep" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_mds_rewritten_q-indep_llama70Binstruct.jsonl"

	#command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 100 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-indep-minimal"
	
	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k $k"


elif [[ $MODE == "rewrite-indep-test" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_stem_mds_rewritten_q-indep_llama70Binstruct_K=3.jsonl"

	#command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-indep-minimal"

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3"



elif [[ $MODE == "rewrite-indep-test-small" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_stem_mds_rewritten_q-indep_llama8Binstruct_K=3.jsonl"

	command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.1-8B-Instruct --method q-indep-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3"


elif [[ $MODE == "rewrite-indep-test-gemma" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_stem_mds_rewritten_q-indep_gemma9Binstruct_K=3.jsonl"

	command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model google/gemma-2-9b-it --method q-indep-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3"

elif [[ $MODE == "rewrite-indep-test-qwen" ]] ; then
	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_stem_mds_rewritten_q-indep_qwen7Binstruct_K=3.jsonl"

	command="python scripts/rewrite.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model Qwen/Qwen2.5-7B-Instruct --method q-indep-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH --k 3"

elif [[ $MODE == "mds-oracle" ]] ; then
	# Oracle reranking
	#command="mkdir -p /data/sewonm/private-retrieval-lm/out && python scripts/rerank.py --retrieval_results $MMLU_MDS_RETRIEVAL_PATH --output_path $MMLU_MDS_ORACLE_RERANK_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_oracle_rerank.json --top_k_rerank 100"

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name oracle_rerank_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_ORACLE_RERANK_PATH --k $k --ctx_key oracle_reranked_ctxs"


elif [[ $MODE == "dclm" ]] ; then

	# NEW DATASTORE
	BASE_DIR=/data/sewonm/dense-retrieval/dclm_ft7percentile_fw3_shard00/retrieved_results
	FILENAME=0_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl

	# Contriever
	#RETRIEVED_FILE=$BASE_DIR/facebook/contriever-msmarco/$FILENAME

	# GTR
	#RETRIEVED_FILE=$BASE_DIR/sentence-transformers/gtr-t5-large/$FILENAME

	# E5
	#RETRIEVED_FILE=$BASE_DIR/intfloat/e5-large-v2/$FILENAME

	# Snowflake
	#RETRIEVED_FILE=$BASE_DIR/Snowflake/snowflake-arctic-embed-l-v2.0/$FILENAME

elif [[ $MODE == "lb" ]] ; then

	# LB DATASTORE
	BASE_DIR=/data/sewonm/dense-retrieval/lb_full/retrieved_results
	FILENAME=0_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl

	# Contriever
	RETRIEVED_FILE=$BASE_DIR/facebook/contriever-msmarco/$FILENAME #,$MMLU_MDS_RETRIEVAL_PATH

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $RETRIEVED_FILE --k $k"

elif [[ $MODE == "rubric" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_rubric_reranked_no_answer.jsonl --method rubric_annotate --rerank_config configs/rerank/mmlu_rubric_annotation_llama_3.1_0.4.json --top_k_rerank 1000"

elif [[ $MODE == "grit" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_grit_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/grit_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --top_k_rerank 1000"

elif [[ $MODE == "michael_grit" ]] ; then
	command="python scripts/rerank.py --retrieval_results s3://ai2-llm/pretraining-data/sources/ds-olmo-data/upperbound_retrieval/web_retrieval/results_for_grit_reranking/mmlu:mc::retrieval.jsonl --output_path s3://ai2-llm/pretraining-data/sources/ds-olmo-data/upperbound_retrieval/web_retrieval/results_for_grit_reranking/mmlu:mc::retrieval_grit_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/grit_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --top_k_rerank 100"

elif [[ $MODE == "oracle" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_oracle_reranked.jsonl --method oracle_rerank --rerank_config configs/rerank/olmes_oracle_rerank.json --top_k_rerank 100 --task_name mmlu:mc::retrieval"

elif [[ $MODE == "contriever" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_contriever_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/contriever_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --top_k_rerank 1000"


elif [[ $MODE == "rubric_over_grit" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_rubric_reranked_no_answer_70B.jsonl --method rubric_annotate --rerank_config configs/rerank/mmlu_rubric_annotation_llama_3.1_0.4.json --top_k_rerank 100 --retrieval_score_key 'grit score' --reranker_model $MODEL_NAME"

elif [[ $MODE == "rubric_over_grit_new" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_rubric_reranked_no_answer_new.jsonl --method rubric_annotate --rerank_config configs/rerank/mmlu_rubric_annotation_llama_3.1_0.5.json --top_k_rerank 100 --retrieval_score_key 'grit score' --reranker_model $MODEL_NAME"


elif [[ $MODE == "reasonIR" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_reasonIR_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/reasonIR_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --top_k_rerank 100"

elif [[ $MODE == "rubric-batch" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_reranked.jsonl --method rubric_annotate --rerank_config configs/rerank/mmlu_rubric_annotation_llama_3.1_0.3.json --top_k_rerank 5 --rank $RANK --world_size $WORLD_SIZE"

elif [[ $MODE == "grit-batch" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_grit_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/grit_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --rank $RANK --world_size $WORLD_SIZE"

elif [[ $MODE == "reasonIR-batch" ]] ; then
	command="python scripts/rerank.py --retrieval_results $RET_PATH.jsonl --output_path ${RET_PATH}_reasonIR_reranked.jsonl --method embedding_rerank --rerank_config configs/rerank/reasonIR_embed_rerank_no_chunk.json --intermediate_output_dir /tmp --rank $RANK --world_size $WORLD_SIZE"

elif [[ $MODE == "lb-oracle" ]] ; then
	# LB DATASTORE
	BASE_DIR=/data/sewonm/dense-retrieval/lb_full/retrieved_results
	FILENAME=0_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl

	# Contriever
	RETRIEVED_FILE=$BASE_DIR/facebook/contriever-msmarco/$FILENAME,$MMLU_MDS_RETRIEVAL_PATH

	#command="python scripts/rerank.py --retrieval_results $RETRIEVED_FILE --output_path $MMLU_LB_ORACLE_RERANK_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_oracle_rerank.json --top_k_rerank 100"

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name oracle_rerank_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_LB_ORACLE_RERANK_PATH --k $k --ctx_key oracle_reranked_ctxs"

	#command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name oracle_rerank_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_LB_ORACLE_RERANK_PATH,$MMLU_MDS_ORACLE_RERANK_PATH --k $k --ctx_key oracle_reranked_ctxs"

elif [[ $MODE == "openmathinstruct2" ]] ; then

	RETRIEVED_FILE=/data/sewonm/dense-retrieval/openmathinstruct2/retrieved_results/facebook/contriever-msmarco/openmathinstruct2_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl #,$MMLU_MDS_RETRIEVAL_PATH

	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $RETRIEVED_FILE --k $k"

elif [[ $MODE == "finemath" ]] ; then
	RETRIEVED_FILE=/data/sewonm/dense-retrieval/finemath/retrieved_results/facebook/contriever-msmarco/finemath_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl,$MMLU_MDS_RETRIEVAL_PATH
	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $RETRIEVED_FILE --k $k --sources_to_keep rpj_c4,finemath,rpj_stackexchange,pes2co"


elif [[ $MODE == "all" ]] ; then
	OPENMATH_RETRIEVED_FILE=/data/sewonm/dense-retrieval/openmathinstruct2/retrieved_results/facebook/contriever-msmarco/openmathinstruct2_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl
	FINEMATH_RETRIEVED_FILE=/data/sewonm/dense-retrieval/finemath/retrieved_results/facebook/contriever-msmarco/finemath_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl
	RETRIEVED_FILE=$OPENMATH_RETRIEVED_FILE,$FINEMATH_RETRIEVED_FILE,$MMLU_MDS_RETRIEVAL_PATH
	command="python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $RETRIEVED_FILE --k $k --sources_to_filter math"

elif [[ $MODE == "rewrite-math" ]] ; then
	MMLU_OPENMATH_RETRIEVAL_PATH=/data/sewonm/dense-retrieval/openmathinstruct2/retrieved_results/facebook/contriever-msmarco/openmathinstruct2_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl #,$MMLU_MDS_RETRIEVAL_PATH
	MMLU_FINEMATH_RETRIEVAL_PATH=/data/sewonm/dense-retrieval/finemath/retrieved_results/facebook/contriever-msmarco/finemath_datastore-256_chunk_size/top_100/cot_queries_retrieved_results.jsonl

	MMLU_MDS_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_mds_rewritten_q-dep_llama70Binstruct_K=3.jsonl"
	MMLU_OPENMATH_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_openmathinstruct2_rewritten_q-dep_llama70Binstruct_K=3.jsonl"
	MMLU_FINEMATH_REWRITTEN_PATH="/data/sewonm/private-retrieval-lm/out/mmlu_finemath_rewritten_q-dep_llama70Binstruct_K=3.jsonl"


	command="python scripts/rewrite.py --retrieval_results $MMLU_OPENMATH_RETRIEVAL_PATH --output_path $MMLU_OPENMATH_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-dep-no-option-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH,$MMLU_OPENMATH_REWRITTEN_PATH --k 3 --ctx_key rewritten_ctxs"

	####

	command="python scripts/rewrite.py --retrieval_results $MMLU_FINEMATH_RETRIEVAL_PATH --output_path $MMLU_FINEMATH_REWRITTEN_PATH --method oracle_rerank --rerank_config configs/rerank/mmlu_rubric_annotation.json --top_k_rerank 3 --reranker_model meta-llama/Llama-3.3-70B-Instruct --method q-dep-no-option-minimal"

	command="$command && python scripts/eval.py --config configs/eval/offline_retrieval.json --experiment_name offline_retrieval_mmlu_subset --model_name $MODEL_NAME --task_config_paths tasks/mmlu_subset.json --retrieval_results_path $MMLU_MDS_REWRITTEN_PATH,$MMLU_FINEMATH_REWRITTEN_PATH --k 3 --ctx_key rewritten_ctxs"

else
	echo "Invalid $MODE"
	exit
fi


gantry run \
    --task-name "Scaling-RAG_${MODE}" \
    --description "Scaling-RAG_${MODE} $MODEL_NAME" \
    --workspace ai2/ds-olmo \
    --allow-dirty \
    --beaker-image 'lucas/refine1' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --gpus 1 \
    --replicas $RANK \
    --preemptible \
    --cluster "${CLUSTER}" \
	--budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
	--env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env-secret AWS_ACCESS_KEY_ID=SEWONM_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=SEWONM_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
    --env-secret HF_TOKEN=SEWONM_HF_TOKEN \
    --shared-memory 10GiB \
	--no-logs \
    --yes \
	--cluster ai2/jupiter-cirrascale-2 \
	--install "pip install necessary simple_parsing smart-open pydantic vllm gritlm && pip install --force-reinstall -U --no-deps -v peft==0.14.0" \
	-- /bin/bash -c "PYTHONPATH=.:olmes $command"
	
	# --install "pip install -e olmes && pip install necessary simple_parsing smart-open pydantic vllm gritlm && pip install --force-reinstall -U --no-deps -v peft==0.14.0" \
	# --install "pip install necessary simple_parsing smart-open pydantic vllm gritlm && pip install --force-reinstall -U --no-deps -v peft==0.14.0" \
    # --weka oe-data-default:/weka_data \
	# --cluster ai2/jupiter-cirrascale-2 \
	# --retries 0 \
    
