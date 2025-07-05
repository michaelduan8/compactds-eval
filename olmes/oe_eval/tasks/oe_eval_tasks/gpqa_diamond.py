"""
Title: GPQA: A Graduate-Level Google-Proof Q&A Benchmark

Abstract: https://arxiv.org/abs/2311.12022

We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, 
physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who 
have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes 
the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, 
despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are 
“Google-proof”). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4–
based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions—
for example, when developing new scientific knowledge—we need to develop scalable oversight methods that enable 
humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and 
knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable 
realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably 
get truthful information from AI systems that surpass human capabilities.

Homepage: https://github.com/idavidrein/gpqa/tree/main

Config for the diamond version of the GPQA benchmark.
"""

from oe_eval.tasks.oe_eval_tasks.gpqa import GPQA

_CITATION = """
@misc{rein2023gpqa,
      title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      year={2023},
      eprint={2311.12022},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
"""

class GPQADiamond(GPQA):
    """Most of the implementation is inherited from GPQA"""

    VERSION = 0.1
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "Idavidrein/gpqa",
        "dataset_name": "gpqa_diamond",
        "split": "train",
        "native_id_field": "id",  # "Dataset auto index"
        "primary_metric": "exact_match",
        "fewshot_source": "Original:GPQA",
        "generation_kwargs": {
            "max_gen_toks": 1024,
            "do_sample": False,
            "temperature": 0.0,
            "stop_sequences": ["</s>"],
        },
        "context_kwargs": {
            "answer_shuffling_seed": 111,
        },
        "metric_kwargs": {},
        "chat_overrides": {
            "context_kwargs": {
                "description": "Answer the following multiple choice question. The last line of your response should be of the following format: `The correct answer is: ($LETTER)` where LETTER is one of ABCD. Think step by step before answering.\n",
                "assistant_prefix": "\nLet's think step by step:\n",
                "fewshot_as_multiturn": True,
            },
        },
    }