import json

file_path = "output_rewritten_trace/gpqa_diamond_0shot_cot__retrieval_q_retrieved_results_with_reasoning_trace_w_answer_rewritten_simple.jsonl"

total_length = 0
count = 0

outputs = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        assert "New Trace:" in data.get("rewritten_trace", "")[0], f"Unexpected format in rewritten_trace: {data.get('rewritten_trace', '')}"
        original_trace = data["ctxs"][0]["retrieval text"]
        if len(original_trace.split("Answer:")) < 2:
            assert False, f"Original trace does not contain 'Answer:': {original_trace}"
        new_trace = original_trace.split("Reasoning Snippet:")[0].strip() + "\nReasoning Snippet: " + data["rewritten_trace"][0].split("New Trace:")[1].strip() + "\nAnswer: " + original_trace.split("Answer:")[1].strip()
        
        data["ctxs"][0]["retrieval text"] = new_trace
        outputs.append(data)

open(file_path + "_converted.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(output) for output in outputs]))