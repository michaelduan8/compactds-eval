import argparse
import os
import torch

from src.utils import load_jsonl, write

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _prepare_doc(doc, trace_key, mode="simple"):
    assert len(doc["ctxs"]) == 1, "Only support one retrieved doc for now"
    ctx = doc["ctxs"][0]
    retrieval_text = ctx[trace_key]
    question = retrieval_text.split("Reasoning Snippet:")[0].split("Question:")[1].strip()
    reasoning_snippet = retrieval_text.split("Reasoning Snippet:")[1].split("Answer:")[0].strip()
    answer = retrieval_text.split("Answer:")[1].strip()
 
    if mode == "simple":
        return [
            { "role": "user", "content": f'Here is a question with an accompanying reasoning trace and answer.\n\nQuestion: {question}\n\nReasoning Trace: {reasoning_snippet}\n\nAnswer: {answer}\n\nPlease rewrite the reasoning trace. Only output "New Trace:" followed by the rewritten reasoning trace.' }
        ]
    
    if mode == "cot":
        return [
            { "role": "user", "content": f'Here is a question with an accompanying reasoning trace and answer.\n\nQuestion: {question}\n\nReasoning Trace: {reasoning_snippet}\n\nAnswer: {answer}\n\nPlease rewrite the reasoning trace to be more clear and concise while ensuring that the answer remains the same. The rewritten reasoning trace should be easy to understand and follow. Think about how you will rewrite the trace to make it more clear and concise, then answer with "New Trace:" followed by the rewritten reasoning trace.' }
        ]
    
def prepare(retrieval_results_path, trace_key, mode):
    data = load_jsonl(retrieval_results_path)
    requests = []
    for datum in data:
        req = _prepare_doc(datum, trace_key, mode)
        requests.append(req)

    return requests, data

def rewrite(requests, model):
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, 
              tensor_parallel_size=torch.cuda.device_count(),
              gpu_memory_utilization=0.8,
              max_model_len=65536,
              rope_scaling={"rope_type": "yarn",
                            "factor": 2.0,
                            "original_max_position_embeddings": 32768})
    
    # TODO: not sure what happens with models without thinking mode
    requests = tokenizer.apply_chat_template(
        requests,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    print(requests[0])

    sampling_params = SamplingParams(temperature=0.7, 
                                     top_p=0.8, 
                                     top_k=20, 
                                     max_tokens=32768)
    

    outputs = llm.generate(requests, sampling_params)

    return [{"input": output.prompt, "output": [output.outputs[0].text.strip()]} for output in outputs]
    
     
def main(args):
    retrieved_results_path = args.retrieved_results_path
    trace_key = args.trace_key
    mode = args.mode

    model = args.model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    requests, data = prepare(retrieved_results_path, trace_key, mode)

    rewritten_traces = rewrite(requests, model)

    new_data = []
    for doc, rewritten_trace in zip(data, rewritten_traces):
        new_data.append(doc | {
            "rewritten_trace": rewritten_trace["output"]
        })

    output_path = os.path.join(output_dir, os.path.basename(retrieved_results_path).replace(".jsonl", f"_rewritten_{mode}.jsonl"))
    write(output_path, new_data)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved_results_path", 
        type=str,
        required=True,
        help="retrieval results"
    )
    # TODO: Variable to changing depending on file structure
    parser.add_argument(
        "--trace_key",
        type=str,
        default="retrieval text",
        help="Key to access the rewritten question",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "cot"],
        help="Mode for rewriting the trace",
    )
    parser.add_argument("--model", default=None, type=str, help="Which model to use for rewriting")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files")

    args = parser.parse_args()
    main(args)
