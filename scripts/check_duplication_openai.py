import asyncio
import backoff
import json
import random
import openai
import os

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

NUM = 5
TOP_K = 10
MAX_CONCURRENT_REQUESTS = 20
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def postprocess(question, context, response):
    answer = response.choices[0].message.content.strip().lower()
    if "yes" in answer:
        return {
            "question": question,
            "context": context,
            "duplication": True
        }
    elif "no" in answer:
        return {
            "question": question,
            "context": context,
            "duplication": False
        }
    else:
        print(f"Unclear answer: {answer}\n\nQuestion: {question}\n\nContext: {context}")
        return {
            "question": question,
            "context": context,
            "duplication": None
        }
    
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.APIConnectionError))
async def generate(client, prompt, model, temperature, max_tokens):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )

        return response


async def batch_generate(client, queries, model, qc_pairs, temperature=0, max_tokens=None):
    raw_outputs = await tqdm_asyncio.gather(*[generate(client, query, model, temperature=temperature, max_tokens=max_tokens) for query in queries])
    outputs = [postprocess(question, context, output) for (question, context), output in zip(qc_pairs, raw_outputs)]

    return outputs



if __name__ == "__main__":
    # Load datasets
    agi_eval_data = [json.loads(line) for line in open("data/agi_eval_english::retrieval_q_retrieved_results_k=1000.jsonl")]
    math_data = [json.loads(line) for line in open("data/minerva_math::retrieval_q_retrieved_results_k=1000.jsonl")]
    mmlu_pro_data = [json.loads(line) for line in open("data/mmlu_pro:mc::retrieval_q_retrieved_results_k=1000.jsonl")]

    random.seed(2025)

    random.shuffle(agi_eval_data)
    agi_eval_data = agi_eval_data[:NUM]
    random.shuffle(math_data)
    math_data = math_data[:NUM]
    random.shuffle(mmlu_pro_data)
    mmlu_pro_data = mmlu_pro_data[:NUM]

    all_data = agi_eval_data + math_data + mmlu_pro_data
    question_context_pairs = []
    for item in all_data:
        question = item["raw_query"]
        for ctx in item["ctxs"][:TOP_K]:
            context = ctx["retrieval text"]
            question_context_pairs.append((question, context))

    print(f"Total question-context pairs to evaluate: {len(question_context_pairs)}")

    # Connect to vLLM via OpenAI-compatible API
    # Make sure you have started vLLM with: 
    #   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.environ["OPENROUTER_API_KEY"],
    # )
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    print("Evaluating duplication...")
    prompts = [f"Below is a question and a retrieved passage. Does the passage contain text that is nearly identical to and clearly recognizable as the question? Answer with only 'Yes' or 'No'.\n\n<question>\n{question}\n</question>\n\n<passage>\n{context}\n</passage>\n\nYour answer: " for question, context in question_context_pairs]

    # final_outputs = []


    model = "gpt-5-mini"
    temperature = 0.0
    max_tokens = 10

    final_outputs = asyncio.run(batch_generate(client, prompts, model, question_context_pairs, temperature, max_tokens))
    
    # for i, (prompt, (question, context)) in enumerate(tqdm(zip(prompts, question_context_pairs))):
        
    #     response = client.chat.completions.create(
    #         model="openai/gpt-5-nano",  # or another model loaded in vLLM
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=0.0,
    #         extra_body={},
    #     )

    #     answer = response.choices[0].message.content.strip().lower()
    #     if "yes" in answer:
    #         final_outputs.append({
    #             "question": question,
    #             "context": context,
    #             "duplication": True
    #         })
    #     elif "no" in answer:
    #         final_outputs.append({
    #             "question": question,
    #             "context": context,
    #             "duplication": False
    #         })
    #     else:
    #         print(f"Unclear answer for pair {i}: {answer}")
    #         final_outputs.append({
    #             "question": question,
    #             "context": context,
    #             "duplication": None
    #         })

    json.dump(final_outputs, open("duplication_results_gpt5_nano2.jsonl", "w"), indent=4)
