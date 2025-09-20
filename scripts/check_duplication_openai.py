import asyncio
import backoff
import json
import random
import openai
import os

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from collections import Counter
import numpy as np

# PRICE_FACTOR = 1
PRICE_FACTOR = 5

NUM = -1
TOP_K = 10
MAX_CONCURRENT_REQUESTS = 20


def postprocess(question, context, response):
    answer = response.choices[0].message.content.strip().lower()
    if "yes" in answer:
        return {
            "question": question,
            "context": context,
            "duplication": True,
            "usage": response.usage.prompt_tokens * 0.05 / 1000000 + response.usage.completion_tokens * 0.4 / 1000000
        }
    elif "no" in answer:
        return {
            "question": question,
            "context": context,
            "duplication": False,
            "usage": response.usage.prompt_tokens * 0.05 / 1000000 + response.usage.completion_tokens * 0.4 / 1000000
        }
    else:
        print(f"Unclear answer: {answer}")
        return {
            "question": question,
            "context": context,
            "duplication": None,
            "usage": response.usage.prompt_tokens * 0.05 / 1000000 + response.usage.completion_tokens * 0.4 / 1000000
        }
    
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.APIConnectionError))
async def generate(client, prompt, model, semaphore, temperature, max_tokens):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=temperature,
            # max_completion_tokens=max_tokens
        )

        return response


async def batch_generate(client, queries, model, qc_pairs, temperature=0, max_tokens=None):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    raw_outputs = await tqdm_asyncio.gather(*[generate(client, query, model, semaphore, temperature=temperature, max_tokens=max_tokens) for query in queries])
    outputs = [postprocess(question, context, output) for (question, context), output in zip(qc_pairs, raw_outputs)]

    return outputs



if __name__ == "__main__":
    data_path = "data/minerva_math::retrieval_q_retrieved_results_k=1000_grit_reranked.jsonl"
    # Load datasets
    data = [json.loads(line) for line in open(data_path)]
    data = data[:NUM] if NUM > 0 else data
    
    print(f"Loaded {len(data)} data points from {data_path}")

    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-5-mini"
    temperature = 1.0
    max_tokens = 10
    total_cost = 0.0

    need_to_check_more = True

    round = 0
    while need_to_check_more:
        start_indices_in_original = []
        nums_need_to_be_labeled = []
        question_context_pairs = []
        for item in data:
            question = item["raw_query"]
            # Find nums of context that is not a duplicate
            num_needs_to_be_labeled = TOP_K - len([ctx for ctx in item["ctxs"] if "duplication" in ctx and ctx["duplication"] is False])            
            # Find the first context that is not labeled yet
            start_index = -1
            if num_needs_to_be_labeled > 0:
                for i, ctx in enumerate(item["ctxs"]):
                    if "duplication" not in ctx:
                        start_index = i
                        break
                
                # Add unlabled contexts to the list
                assert start_index != -1, "There should be at least one context that is not checked yet."
                for ctx in item["ctxs"][start_index: start_index + num_needs_to_be_labeled]:
                    context = ctx["retrieval text"]
                    question_context_pairs.append((question, context))

            start_indices_in_original.append(start_index)
            nums_need_to_be_labeled.append(num_needs_to_be_labeled)

        if len(question_context_pairs) == 0:
            need_to_check_more = False
            print("All data points have passed. Stopping.")
            break

        print("Round:", round)
        bins = np.arange(0, max(start_indices_in_original) + 11, 10)
        hist, bin_edges = np.histogram([i for i in start_indices_in_original if i >= 0], bins=bins)
        print("Distribution of start indices (bin size 10):")
        for i in range(len(hist)):
            print(f"{bin_edges[i]}-{bin_edges[i+1]-1}: {hist[i]}")
        dist = Counter(nums_need_to_be_labeled)
        print("Distribution of number of context that need to be checked:", dict(dist))
        print(f"{len([n for n in nums_need_to_be_labeled if n <= 0])} / {len(data)} data points are completed.")
        print(f"Total question-context pairs to evaluate this round: {len(question_context_pairs)}")

        ifcontinue = input("Continue? (y/n): ")
        if ifcontinue.lower() != "y":
            print("Stopping.")
            exit(0)
        
        print("Evaluating duplication...")
        prompts = [f"Below is a question and a retrieved passage. Does the passage contain text that is nearly identical to and clearly recognizable as the question? Answer with only 'Yes' or 'No'.\n\n<question>\n{question}\n</question>\n\n<passage>\n{context}\n</passage>\n\nYour answer: " for question, context in question_context_pairs]

        final_outputs = asyncio.run(batch_generate(client, prompts, model, question_context_pairs, temperature, max_tokens))
        cost = sum(item["usage"] for item in final_outputs) * PRICE_FACTOR
        total_cost += cost
        print(f"Round {round} cost: ${cost:.6f}")


        # Postprocess: assign the outputs back to the original data structure
        position_in_final_outputs = 0
        for i, item in enumerate(data):
            start_index = start_indices_in_original[i]
            num_need_to_be_labeled = nums_need_to_be_labeled[i]
            if start_index == -1 or num_need_to_be_labeled <= 0:
                continue

            assert "duplication" not in item["ctxs"][start_index], "The first context to be labeled should not have been labeled yet."
            assert "duplication" in item["ctxs"][start_index - 1] if start_index - 1 >= 0 else True, "The context before the first context to be labeled should have been labeled already."
            assert position_in_final_outputs + num_need_to_be_labeled <= len(final_outputs), "There should be enough outputs to assign."
            
            for j in range(num_need_to_be_labeled):
                assert item["raw_query"] == final_outputs[position_in_final_outputs + j]["question"], "The question should match."
                item["ctxs"][start_index + j]["duplication"] = final_outputs[position_in_final_outputs + j]["duplication"]
            
            position_in_final_outputs += num_need_to_be_labeled

        # Save intermediate results
        intermediate_output_path = data_path.replace(".jsonl", f"_duplication_checked_round{round}.jsonl")
        with open(intermediate_output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Intermediate results saved to {intermediate_output_path}")
        
        round += 1
    
    # Save results
    for item in data:
        ctxs = item["ctxs"]
        # Partition ctxs into two lists
        front = [ctx for ctx in ctxs if not ("duplication" not in ctx or ctx["duplication"] is None or ctx["duplication"] is True)]
        back = [ctx for ctx in ctxs if ("duplication" not in ctx or ctx["duplication"] is None or ctx["duplication"] is True)]
        item["ctxs"] = front + back

    output_path = data_path.replace(".jsonl", "_duplication_checked.jsonl")
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Results saved to {output_path}")
