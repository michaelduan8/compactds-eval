import numpy as np
import time

from transformers import AutoTokenizer

from modules.generator.generator import Generator
from src.utils import sort_dicts_by_key


# Character length to truncate retrieval docs to to avoid documents that are undertokenized.
# Same as character limit used in retrieval evaluation.
TRUNC_LEN = 5000


def generate(model_name, model_type, prompts_and_metadata, prompt_key, gen_output_key, mode=None, mode_args={}, num_gpus=1, llm=None, tokenizer=None, structured=False, **kwargs):
    """
    This function supports three generation modes:
    - mode=None: Regular generation; also supports structured output via vLLM interface
    - mode="prompt_logprobs": By providing a prompt_suffix_key in mode_args, return the likelihood of specified suffix in prompt per query. No generation is performed.
    - mode="mc": By providing single-token answer_choices in mode_args, return the logprobs for each answer choice. No generation is performed.  
    """

    # Instantiate generator
    print("Loading model...")
    if llm is None:
        if structured:
            params = {
                "guided_decoding_backend": "xgrammar"
            }
        else:
            params = {}
        starttime = time.time()
        llm = Generator(model_name, model_type)
        llm.get_model().load_model(**params)
        endtime = time.time()
        print(f"Loading model took {endtime - starttime}s.")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Extract prompts
    prompts = [p[prompt_key] for p in prompts_and_metadata]

    print("Performing generation...")
    
    if mode == 'prompt_logprobs':
        kwargs = kwargs | {"prompt_logprobs": 1, "return_metadata": True}
    elif mode == "mc":
        # Prepare mc token options
        assert "answer_choices" in mode_args
        answer_choices = mode_args["answer_choices"]
        tokenized_mc_options = tokenizer([f" {mc}" for mc in answer_choices], return_tensors="np").input_ids[:, -1]
        # Ensure we get topk logprobs for next token generation; only generate one token.
        # Use tokenized_mc_options to limit possible next token generations
        kwargs = kwargs | {
            "allowed_token_ids": tokenized_mc_options.tolist(),
            "logprobs": 20,
            "return_metadata": True
        }
    
    print(kwargs)
    starttime = time.time()
    raw_outputs = llm(prompts, **kwargs)
    endtime = time.time()
    print(f"Generation took {endtime - starttime}s.")
    

    outputs = []
    if mode == 'prompt_logprobs':
        # Tokenizer reference
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        assert "prompt_suffix_key" in mode_args
        prompt_suffix_key = mode_args["prompt_suffix_key"]

        for pm, out in zip(prompts_and_metadata, raw_outputs):
            # retokenize specified portion of the prompt
            # to determine which tokens to look at logprobs for
            qa = pm[prompt_suffix_key]

            tokenized_qa = tokenizer(qa, return_tensors="np").input_ids[0]
            num_qa_tokens = len(tokenized_qa)

            # Get the final n tokens from prompt_token_ids corresponding to q+a
            metadata = out["metadata"]
            relevant_tokens = metadata.prompt_token_ids[-num_qa_tokens:]
            assert len(relevant_tokens) == num_qa_tokens

            print(metadata)
            print(metadata.prompt_logprobs)

            # Get their associated logprobs, and compute mean to get likelihood                 
            relevant_token_logprobs = [-metadata.prompt_logprobs[-(num_qa_tokens - i)][tk].logprob for i, tk in enumerate(relevant_tokens)]

            outputs.append({
                "raw_qa_logprobs": relevant_token_logprobs,
                "mean_logprob": np.mean(relevant_token_logprobs)
            })
    elif mode == 'mc':
        for i, dt in enumerate(raw_outputs):
            assert len(dt["metadata"].outputs) > 0, dt
            assert len(dt["metadata"].outputs[0].logprobs) > 0, dt
            log_prob_metadata = dt["metadata"].outputs[0].logprobs[0]
            # Get log probs for each choice
            choice_log_probs = {}
            # Token output distribution should always be restricted to the allowed token ids;
            # VLLM_V1 seems to have an issue with allowed_token_ids currently, so using V0 is advised.
            for token in tokenized_mc_options:
                assert token in log_prob_metadata, f"{tokenizer.decode(token)}, {i}, {log_prob_metadata}"
                token_log_prob = log_prob_metadata[token]
                decoded_option = token_log_prob.decoded_token.replace('Ġ', ' ')
                choice_log_probs[decoded_option] = {
                    "rank": token_log_prob.rank,
                    "logprob": token_log_prob.logprob
                }

            outputs.append({
                "choice_log_probs": choice_log_probs,
                "output": dt["output"]
            })
    else:
        outputs = raw_outputs

    prepared_outputs = [pm | {gen_output_key: output} for pm, output in zip(prompts_and_metadata, outputs)]
    return prepared_outputs


def block_data(text, block_length=None, min_block_length=0, keep_last=True):
    if block_length is None:
        return [text]
    
    # TODO: For better splitting, maybe we use nltk word_tokenize?
    text = text.split()
    N = len(text) if keep_last else len(text) - len(text) % block_length
    blocks = [' '.join(text[i:i + block_length]) for i in range(0, N, block_length)]

    if len(blocks) > 1 and len(blocks[-1].split(' ')) < min_block_length:
        # merge the last min_block_length words to the previous block
        last_chunk = blocks.pop()
        blocks[-1] += ' ' + last_chunk

    return blocks


def pre_sort(retrieval_results, rc):
    # Pre-sort contexts, e.g., for applying top_k_reranking
    for dt in retrieval_results:
        dt[rc.ctx_key], num = sort_dicts_by_key(dt[rc.ctx_key], [rc.retrieval_score_key], reversed=True)
        if rc.top_k_rerank and num < rc.top_k_rerank and len(dt[rc.ctx_key]) >= rc.top_k_rerank:
            raise ValueError(f"Less than top_k_rerank retrieved contexts with {rc.retrieval_score_key} for reranking: {num}.")
    
    return retrieval_results
