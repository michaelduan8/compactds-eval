from .llm import LanguageModel
try:
    from .gpt import GPT
except Exception:
    pass

from src.config import GeneratorConfig


class Generator(object):
    def __init__(self, model_name, model_type='vllm', max_input_len=None):
        self.model_name = model_name
        self.model_type = model_type
        self.max_input_len = max_input_len
        self.model = self.load_generator()

    @classmethod
    def from_config(cls, config: GeneratorConfig):
        model_name = config.model_name
        model_type = config.model_type
        max_input_len = config.max_input_length
        
        return cls(model_name, model_type, max_input_len)

    def get_name(self):
        return self.model_name
    
    def get_model(self):
        return self.model
    
    def load_generator(self, **kwargs):
        if 'gpt' in self.model_name.lower():
            return GPT(self.model_name)
        else:
            return LanguageModel(self.model_name, 
                                 self.model_type, 
                                 max_input_len=self.max_input_len, 
                                 **kwargs)

    # Try to use kwargs instead
    def __call__(self, queries, **kwargs):
        return self.model(queries=queries, **kwargs)
