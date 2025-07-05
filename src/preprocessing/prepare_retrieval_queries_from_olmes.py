import argparse
import copy
import random
from collections import defaultdict
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from olmes.oe_eval.configs.tasks import TASK_CONFIGS
from olmes.oe_eval.launch import resolve_task_suite
from olmes.oe_eval.utils import (
    get_dict_with_defaults,
    load_jsonl,
    parse_args_string,
    save_jsonl,
    concat_with_space
)
from olmes.oe_eval.run_eval import load_task

from modules.generator.generator import Generator
from src.preprocessing.structured_output import BreakDownResponse

def prepare_for_retrieval(task, doc, mode, retrieval_key, pre_gen=None):
    # Format the doc
    if mode == 'q':
        retrieval_text = [task.doc_to_text(doc)]
    elif mode == 'q+a':
        retrieval_text = [concat_with_space(task.doc_to_text(doc) + task.doc_to_target(doc))]
    elif mode == 'break_down':
        output = pre_gen['output']

        try:
            # TODO: Maybe add custom parsing here
            break_down = BreakDownResponse.model_validate_json(output)
            retrieval_text = break_down.search_queries
        except Exception:
            # Default to original query
            print("Parse failure")
            print(output)
            retrieval_text = [task.doc_to_text(doc)]
    
    else:
        raise NotImplementedError()
    
    prepared_docs = []
    for text in retrieval_text:
        # Replace retrieval query and store old value if exists
        prepared_doc = doc | {
            f'{retrieval_key}_olmes': doc[retrieval_key] if retrieval_key in doc else None,
            retrieval_key: text
        }

        # Additional metadata
        prepared_doc = prepared_doc | {
            'full_text_olmes': doc[retrieval_key] if retrieval_key in doc else None,
            'full_text': concat_with_space(task.doc_to_text(doc), task.doc_to_target(doc)),
            'target': task.doc_to_target(doc).strip()
        }
        
        # add id
        id_new = f"{task.task_config['metadata']['alias']}:{doc[task.task_config.get('native_id_field', 'id')]}"
        prepared_doc = prepared_doc | {
            'id_olmes': doc['id'] if 'id' in doc else None,
            'id': id_new
        }

        prepared_docs.append(prepared_doc)

    return prepared_docs

def generate_break_down(task, docs, model, use_langchain=False):
    # TODO: Consider more dynamic way to integrate Break down prompt template and structured output
    # Currently, they are hardcoded
    
    # Build questions
    queries = []
    for doc in docs:
        query = task.doc_to_text(doc)
        queries.append(query)

    outputs = []
    if use_langchain:
        # Decomposition
        template = """
            You are a helpful assistant that generates multiple sub-questions related to an input question. \n
            The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. The sub-questions should also be useful to find helpful relevant information to solve the original problem. \n
            Generate multiple search queries related to:\n\n
            {question}
            \n\n
            Output in json format with key "search_queries" pointing to a list of generated sub-questions (3 queries):
        """
        prompt_decomposition = ChatPromptTemplate.from_template(template)

        # LLM
        llm = ChatOpenAI(temperature=0.7, model=model, max_tokens=4096)
        llm = llm.with_structured_output(BreakDownResponse, method="json_schema")

        # Chain
        generate_queries_decomposition = ( prompt_decomposition | llm )
        for query in tqdm(queries):
            out = generate_queries_decomposition.invoke({
                "question": query
            })
            
            outputs.append({
                "output": out.model_dump_json()
            })
    else:
        prompts = []
        for query in queries:
            break_down_prompt = f'{query}\n\nRewrite the above question as up to three unique search queries to use with a search engine to find helpful relevant information to solve the above problem. Only output the generated search queries as a json dict with key "search_queries" pointing to the list of generated search queries. Do not exceed three search queries.'
            # TODO: hacky coding to chat templates, consider removing
            # or refactoring if we want to continue exploration with it

            # tokenizer = AutoTokenizer.from_pretrained(model.get_name()) 
            # system_prompt = """
            #     You are a helpful assistant that generates multiple sub-questions related to an input question. \n
            #     The goal is to break down the input into a set of sub-questions to search for helpful information to solve the original problem. \n. \
            #     Ensure the sub-questions can be answered without needing to refer to the original question
            #     Only output the generated sub-questions in the json format {"search_queries": list of generated sub-questions}.
            # """
            # template = """
            #     Generate exactly three independent sub-questions to search for relevant information to solve the following question\n\n
            #     {question}\n\n
            #     Do not exceed three sub-questions.
            # """

            # chat = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": template.format(question=query)}
            # ]

            # break_down_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            
            prompts.append(break_down_prompt)

        gen_kwargs = {
            "structured_output": BreakDownResponse.model_json_schema(),
            "max_output_length": 2048
        }
            
        print(f"Num prompts: {len(prompts)}")
        outputs = model(prompts, **gen_kwargs)
    
    return outputs

def main(args):
    ## Load OLMES task configs
    # task_config_shared: they can be set either globally (through --<arg>) or
    # individually (through --task {'<arg>':} for each task).
    task_config_shared = {}
    for key in ["split", "num_shots", "fewshot_seed"]:
        value = args[key]
        if value is not None:
            task_config_shared[key] = value
    task_config_shared["use_chat_format"] = False

    tasks = args["task"]
    # Borrowed from launch.py
    task_configs = []
    all_tasks = []
    task_suite_parent: dict = {}
    for task in tasks:
        all_tasks += resolve_task_suite(task, task_suite_parent)
    for task in all_tasks:
        if task.endswith(".jsonl"):
            task_configs += load_jsonl(task)
        elif task in TASK_CONFIGS:
            task_config = copy.deepcopy(TASK_CONFIGS[task])
            if "metadata" not in task_config:
                task_config["metadata"] = {}
            task_config["metadata"]["alias"] = task
            task_configs.append(task_config)
        elif task in task_suite_parent:
            print(
                f"No config found for task: {task} (from task suite {task_suite_parent[task]})"
            )
            task_configs.append({"task_name": task})
        else:
            task_config = parse_args_string(task, "task_name")
            # Allow updates to existing configured tasks
            if task_config["task_name"] in TASK_CONFIGS:
                new_task_config = task_config
                task_config = copy.deepcopy(TASK_CONFIGS[new_task_config["task_name"]])
                del new_task_config["task_name"]
                task_config.update(new_task_config)
            elif len(task_config) == 1:
                print(f"No config found for task: {task}, using raw task")
            task_configs.append(task_config)
        task_configs = [get_dict_with_defaults(task_config_shared, task_config) for task_config in task_configs]

    task_objects = [
        load_task(task_config, args['output_dir']) for task_config in task_configs
    ]

    model = None
    if args['model']:
        if 'gpt' in args['model'].lower():
            # We'll use langchain, so just keep model name
            model = args['model']
            use_langchain = True
        else:
            model = Generator(args['model']) if args['model'] else None
            model.get_model().load_model()
            use_langchain = False

    total = []
    for task in task_objects:
        # Download
        task.download()
        docs = task.get_eval_docs(limit=task.task_config.get("limit"), random_subsample_seed=2025)
        
        pre_gen_outputs = None
        if model and args['method'] == 'break_down':
            # Pre-generate break down in batch
            pre_gen_outputs = generate_break_down(task, docs, model, use_langchain=use_langchain)

        current = []
        for i, doc in tqdm(enumerate(docs)):
            current.extend(prepare_for_retrieval(task, doc, args['method'], args['retrieval_key'], pre_gen=pre_gen_outputs[i] if pre_gen_outputs else None))
        
        if args['n_samples']:
            random.shuffle(current)
            current = current[:args['n_samples']]

        print(f"{task.task_config['metadata']['alias']}: {len(current)}")

        if not args['combined']:
            save_jsonl(os.path.join(args['output_dir'], f"{task.task_config['metadata']['alias']}_{args['method']}.jsonl"), current)
        else:
            total.extend(current)
    
    if args['combined']:
        save_jsonl(os.path.join(args['output_dir'], f"{'+'.join(args['task'])}_{args['method']}.jsonl"), total)
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_key", 
        type=str,
        help="retrieval text", 
        default="query"
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        required=True,
        help="Task spec(s), or name(s) as in the Task registry",
    )
    parser.add_argument("--split", default=None, type=str, help="split from which to pull eval docs")
    parser.add_argument("--num_shots", type=int, default=None, help="Number of examples in prompt")
    parser.add_argument("--fewshot_seed", type=int, default=None, help="seed for fewshot example sampling")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files")
    parser.add_argument('--n_samples', type=int, default=None)
    methods = ['q', 'q+a', 'break_down'] # TODO: to be possibly added in the futur -> 'synth_doc_gen', 'cot'
    parser.add_argument('--method', type=str, default='q', choices=methods, help='Specify method to prepare data')
    parser.add_argument('--combined', action="store_true", help='Specify method to prepare data')
    parser.add_argument('--model', default=None, help='For certain preparation methods, specify a model to generate retrieval queries')
    args = parser.parse_args()
    main(vars(args))
