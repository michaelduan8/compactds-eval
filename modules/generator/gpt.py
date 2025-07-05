import backoff
import json
import openai
import os 

from tqdm import tqdm

class GPT(object):

    def __init__(self, model_name):
        self.model_name = model_name

        self.client = None

    def load_model(self):
        if not self.client:
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def postprocess_output(self, outputs, usage_tracking_dir=None):
        responses = [c.message.content.strip() for c in outputs.choices]

        if usage_tracking_dir:
            input_tokens = outputs.usage.prompt_tokens
            output_tokens = outputs.usage.completion_tokens
            usage_track_file = os.path.join(usage_tracking_dir, f"{self.model_name}_usage_generation.jsonl")
            with open(usage_track_file, "a") as f:
                f.write(json.dumps({'input': input_tokens, 'output': output_tokens}) + "\n")

        return responses[0]

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.APIConnectionError))
    #@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, prompt, temperature=None, max_tokens=None):
        completion = self.client.chat.completions.create(model=self.model_name,
                                                         messages=[{"role": "user", "content": f"{prompt}"}],
                                                         temperature=temperature,
                                                         max_tokens=max_tokens)

        return completion
    
    def __call__(self, queries, batch_size=8, max_output_length=32, temperature=0, usage_tracking_dir=None, **kwargs):
        # TODO: handle batching and/or async requests
        outputs = []
        for query in tqdm(queries):
            raw_out = self.generate(query, temperature=temperature, max_tokens=max_output_length)
            out = self.postprocess_output(raw_out, usage_tracking_dir)
            outputs.append(out)

        return outputs
