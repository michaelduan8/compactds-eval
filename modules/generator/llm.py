# import easyapi
import os 
import time
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)    
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .model_utils import convert_model_to_int8_on_gpu

class LanguageModel(object):

    def __init__(self, 
                 model_name, 
                 model_type='vllm', 
                 device='cuda:0', 
                 max_input_len=None, 
                 enable_chunked_prefill=True, 
                 use_8bit=False):
        self.model_name = model_name
        self.model_type = model_type

        self.hf_access_token = os.environ.get('HF_TOKEN', '')
        self.device = device
        self.max_input_len = max_input_len  # TODO: Max context len and (left) truncation only supported for vllm currently
        self.enable_chunked_prefill = enable_chunked_prefill
        self.use_8bit = use_8bit

        self.llm = None

    def load_model_and_tokenizer(self):
        # TODO: support other models
        model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_access_token)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_access_token, padding_side='left')
        tokenizer.pad_token_id = model.config.eos_token_id
        if self.use_8bit:
            model = convert_model_to_int8_on_gpu(model, device=self.device)

        return tokenizer, model
    
    def load_model(self, **kwargs):
        if not self.llm:
            if self.model_type == 'vllm': 
                if self.model_name.startswith("allenai/"):
                    self.max_input_len = 4096
                if self.max_input_len:
                    kwargs = kwargs | {"max_model_len": self.max_input_len}
                if "gpu_memory_utilization" not in kwargs:
                    kwargs = kwargs | {"gpu_memory_utilization": 0.8}

                #max_model_len = 4096 if self.model_name.startswith("allenai/") else 32000
                self.llm = LLM(model=self.model_name, 
                               tensor_parallel_size=torch.cuda.device_count(),
                               **kwargs)
            elif self.model_type == "hf":
                tokenizer, model = self.load_model_and_tokenizer()
                self.llm = pipeline("text-generation", 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=self.device)
            # elif self.model_type == 'easyapi':
            #     self.llm = easyapi.Api('jupiter')
            #     print(f"Launching {self.model_name}...")
            #     self.llm.launch_model(self.model_name, gpus=1, hf_token=self.hf_access_token) # launch on jupiter
            #     while not self.llm.has_model(self.model_name):
            #         print(f"Waiting for {self.model_name} to be launched...")
            #         time.sleep(10)
                
            #     print(f"{self.model_name} loaded!")
            else:
                raise NotImplementedError()
                

    def __call__(self, queries, batch_size=32, max_output_length=32, temperature=0, structured_output=None, return_metadata=False, debug=False, **kwargs):

        # Move queries list to dataset if queries is not already a HF dataset
        # if type(queries) == list:
        #     queries = Dataset.from_list([{"query": query} for query in queries])
        # else:
        #     query_ds = queries
        inputs = queries #[datum["query"] for datum in queries]
        # print(inputs[:5])
        

        # for allenai
        if self.model_name.startswith("allenai/"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenized_inputs = tokenizer(inputs)["input_ids"]
            cnt = 0
            for i, (input_, tokens) in enumerate(zip(inputs, tokenized_inputs)):
                if len(tokens) >= 4096 - 10:
                    cnt += 1
                    while True:
                        input_ = " ".join(input_.split(" ")[10:])
                        tokenized_input = tokenizer([input_])["input_ids"][0]
                        if len(tokenized_input) >= 4096 - 10:
                            continue
                        else:
                            inputs[i] = input_
                            break
            print (f"{cnt}/{len(inputs)} inputs got truncated!")

        if debug:
            ############### generating samples
            import numpy as np
            random_indices = np.random.permutation(range(len(inputs)))[:5]
            random_inputs = [inputs[i] for i in random_indices]
            print ("-"*100)
            for random_input in random_inputs:
                print (random_input)
                print ("-"*20)
            print ("-"*100)


        if self.model_type == 'vllm':
            sampling_params_dict = {
                "temperature": temperature,
                "max_tokens": max_output_length
            }
            if self.model_name.startswith("allenai/"):
                sampling_params_dict["truncate_prompt_tokens"] = 4096 - 10
            if self.max_input_len:  #model_name.startswith("allenai/"):
                sampling_params_dict["truncate_prompt_tokens"] = self.max_input_len
            if structured_output:
                guided_decoding_params = GuidedDecodingParams(json=structured_output)
                sampling_params_dict["guided_decoding"] = guided_decoding_params
                
            sampling_params = SamplingParams(**sampling_params_dict, **kwargs)
            outputs = self.llm.generate(inputs, sampling_params, use_tqdm=True)
            outputs = [{"query": output.prompt, "output": output.outputs[0].text.strip(), "metadata": output if return_metadata else None} for output in outputs]
       
            if debug:
                random_outputs = [outputs[i] for i in random_indices]
                print ("-"*100)
                for random_output in random_outputs:
                    print (random_output["query"])
                    print (random_output["output"])
                    print ("-"*20)
                print ("-"*100)

            return outputs

        elif self.model_type == 'hf':
            inputs_ds = Dataset.from_list([{"input": input} for input in inputs])
            outputs = []
            for batch_out in tqdm(self.llm(KeyDataset(inputs_ds, "input"), 
                                           batch_size=batch_size, 
                                           max_new_tokens=max_output_length, 
                                           do_sample=False, 
                                           return_full_text=False)):
                outputs.extend([out["generated_text"].strip() for out in batch_out])
            
            return [{"query": input, "output": output} for input, output in zip(inputs, outputs)]
        
        # return [{"query": input, "output": output} for input, output in zip(inputs["query"], outputs)]
        # elif self.model_type == 'easyapi':
        #     outputs = []
        #     for i in range(0, len(inputs), batch_size):
        #         batch = inputs[i:i+batch_size]
        #         batch_outputs = self.llm.generate(batch,
        #                                           model=self.model_name, 
        #                                           temp=0.1, 
        #                                           max_tokens=max_output_length)
        #         outputs.extend(batch_outputs)

        #     # TODO: what is output format of easyapi?
        #     return [{"query": input, "output": output.strip()} for input, output in zip(inputs, outputs)]
        else:
            raise NotImplementedError()
