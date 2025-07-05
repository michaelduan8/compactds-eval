import argparse
import json
import pandas as pd
import numpy as np
import os

from collections import defaultdict
from pathlib import Path
from texttable import Texttable

from olmes.oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

# Map broader categories to contained subcategories
CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "engineering", "math"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}

# Reverse map of subcategories to categories
SUBCATEGORY_TO_CATEGORIES = {
    subcategory: category for category, subcategories in CATEGORIES.items() for subcategory in subcategories
}

TASK_TO_NAME = {
    "mmlu:mc::retrieval": "MMLU",
    "mmlu_pro:mc::retrieval": "MMLU Pro",
    "agi_eval_english::retrieval": "AGI Eval",
    "gpqa:0shot_cot::retrieval": "GPQA",
    "minerva_math::retrieval": "Minerva Math",
    "gpqa:0shot_cot::retrieval:long": "GPQA Diamond"
}

TASK_ORDER = ["MMLU", "MMLU:STEM", "MMLU:social sciences", "MMLU:humanities",
                "MMLU:other", "MMLU Pro", "AGI Eval", "GPQA", "GPQA:Biology", "GPQA:Physics", "GPQA:Chemistry", "Minerva Math", "Average w/o subcat", "Average with subcat"]

TASK_WO_CAT = ["MMLU", "MMLU Pro", "AGI Eval", "GPQA", "Minerva Math"]
TASK_W_CAT = ["MMLU:STEM", "MMLU:social sciences", "MMLU:humanities", "MMLU:other", "MMLU Pro", 
              "AGI Eval", "GPQA:Biology", "GPQA:Physics", "GPQA:Chemistry", "Minerva Math"]


def print_latex_table(df):
    column_order = ["index", 'MMLU:STEM', 'MMLU:humanities', 'MMLU:social sciences',
            'MMLU:other', 'MMLU Pro', 'AGI Eval', 'Minerva Math',
        'GPQA:Physics', 'GPQA:Biology',  'GPQA:Chemistry', 
        'Average w/o subcat']


    column_map = {
        'datastore': "Source",
        "index": "Method",
        ' MMLU': 'MMLU',
        'MMLU:humanities': 'MMLU:Human.' , 
        'MMLU:social sciences': 'MMLU:Social',
        'MMLU:other': 'MMLU:Others',
        'Minerva Math': 'Math',
        'GPQA': 'GPQA',
        'GPQA:Physics': 'GPQA:Phys',
        'GPQA:Biology': 'GPQA:Bio',
        'GPQA:Chemistry': 'GPQA:Chem',
        'Average w/o subcat': "AVG",
        'Average with subcat': "AVG",
    }


    # Function to bold the maximum value in each column
    def bold_max_in_column(df):
        for col in df.columns[1:]:  # Skip 'Method' column
            max_val = df[col].max()
            df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.1f}}}" if x == max_val else f"{x:.1f}")
        return df


    # Load the CSV file into a DataFrame
    df = df[[c for c in column_order if c in df.columns]]
    df = df.rename(columns=column_map)

    # Multiply all numeric columns by 100
    df.iloc[:, 1:] = df.iloc[:, 1:] * 100
    df = df.sort_values(by='AVG', ascending=False)

    # Escape underscores in the 'Method' column for LaTeX
    df['Method'] = df['Method'].apply(lambda x: x.replace('_', '\\_'))

    # Bold
    # df = bold_max_in_column(df)
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: f'\\avg{{{round(x, 1)}}}')
    # Convert to LaTeX format
    latex_table = df.to_latex(index=False, header=True, float_format="%.1f", escape=False)
    print(latex_table)


def print_table(data):
    headers = ["Task"] + list(next(iter(data.values())).keys())

    table = Texttable()
    table.header(headers)

    for name, info in data.items():
        row = [name] + [info.get(col, "") for col in headers[1:]]
        table.add_row(row)

    print(table.draw())


def main(args):
    result_file_dir = args.result_file_dir
    tasks = args.tasks

    # find all method name
    if args.method_names is None:
        method_names = [d for d in os.listdir(result_file_dir) if os.path.isdir(os.path.join(result_file_dir, d))]
    else:
        method_names = args.method_names
    
    data = {}
    for method_name in method_names:
        method_result_file_dir = os.path.join(args.result_file_dir, method_name)
        scores = {}
        grouped_mmlu_scores = defaultdict(list)
        grouped_gpqa_scores = defaultdict(list)

        metrics = []
        gpqa_prefixs = []
        for file_path in Path(method_result_file_dir).rglob('*-metrics.json'):
            metric = json.load(open(file_path, 'r'))
            if metric["task_config"]["metadata"]["alias"] == "gpqa:0shot_cot::retrieval" or \
                metric["task_config"]["metadata"]["alias"] == "gpqa:0shot_cot::retrieval:long":
                gpqa_primary_metric_key = metric["task_config"]["primary_metric"]
                gpqa_prefixs.append("-".join(str(file_path).split("-")[:-1]))
            metrics.append(metric)

        metrics = add_aggregate_tasks(metrics) + metrics

        for metric in metrics:
            if metric['task_config']['metadata']['alias'] in tasks:
                scores[TASK_TO_NAME[metric['task_config']['metadata']['alias']]] = metric['metrics']['primary_score']
            elif metric['task_config']['dataset_path'] == 'cais/mmlu' and 'mc' in metric['task_name']:
                grouped_mmlu_scores[SUBCATEGORY_TO_CATEGORIES[SUBCATEGORIES[metric['task_config']['dataset_name']][0]]].append(metric['metrics']['primary_score'])
                    
        if ("gpqa:0shot_cot::retrieval" in tasks or "gpqa:0shot_cot::retrieval:long" in tasks) and len(gpqa_prefixs) > 0:
            # Build id to domain mapping
            id_to_domain = {}
            for gpqa_prefix in gpqa_prefixs:
                with open(gpqa_prefix + "-requests.jsonl", 'r') as f:
                    for line in f:
                        dp = json.loads(line)['doc']
                        id_to_domain[dp['id']] = dp['domain']

                # Group scores by domain
                with open(gpqa_prefix + "-predictions.jsonl", 'r') as f:
                    for line in f:
                        dp = json.loads(line)
                        domain = id_to_domain[dp['native_id']]
                        grouped_gpqa_scores[domain].append(dp['metrics'][gpqa_primary_metric_key])

        # Add per-category scores
        for cat, score_group in grouped_mmlu_scores.items():
            scores[f"MMLU:{cat}"] = np.mean(score_group)
        
        total_score = []
        for cat, score_group in grouped_gpqa_scores.items():    
            scores[f"GPQA:{cat}"] = np.mean(score_group)
            total_score += score_group

        # Add average scores:
        scores["Average w/o subcat"] = np.mean([scores[task] for task in TASK_WO_CAT if task in scores])
        scores["Average with subcat"] = np.mean([scores[task] for task in TASK_W_CAT if task in scores])

        scores = {key: scores[key] for key in TASK_ORDER if key in scores}
        data[method_name] = scores

         # Output results to json format
        if not os.path.exists(os.path.join(args.output_dir, method_name)):
            os.makedirs(os.path.join(args.output_dir, method_name))
        with open(os.path.join(os.path.join(args.output_dir, method_name), "results.json"), 'w') as f:
            json.dump(scores, f, indent=4)

        print(f"Results for {method_name} saved to {os.path.join(os.path.join(args.output_dir, method_name), 'results.json')}")

    # Transpose data to have tasks as keys
    print(data)
    all_inner_keys = set()
    for inner_dict in data.values():
        all_inner_keys.update(inner_dict.keys())
    transposed = {key: {} for key in all_inner_keys}
    for outer_key, inner_dict in data.items():
        for key in all_inner_keys:
            transposed[key][outer_key] = inner_dict.get(key, '')  # Fill with blank if missing
    data = transposed

    data = {key: data[key] for key in TASK_ORDER if key in data}
    
    print_table(data)

    # Output results to csv    
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.T
    df = df.reset_index()

    print_latex_table(df)

    df.to_csv(os.path.join(args.output_dir, "results.csv"))
    print(f"All results saved to {os.path.join(args.output_dir, 'results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+" , 
                        default= ["mmlu:mc::retrieval", "mmlu_pro:mc::retrieval", "agi_eval_english::retrieval", 
                                  "gpqa:0shot_cot::retrieval", "minerva_math::retrieval"], help="Directory for output files")
    parser.add_argument("--result_file_dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--method_names", type=str, nargs="+", default=None, help="Directory for output files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    args = parser.parse_args()
    main(args)
