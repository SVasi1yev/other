from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
)
import torch
from collections import deque
import os

TG_TOKEN = '1338832552:AAFQzZ_2f-TZqQ1eQSIDH9p1n_5cxq2vwBg'
DEVICE = 'cpu'
MAX_USERS = 10000
MAX_DIALOG_LEN = 20
MODELS_DIR = 'models'


models_dict = {}
tokenizer_dict = {}
# tokenizer_dict['dialogpt_small'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
tokenizer_dict['dialogpt_medium'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer_dict['dialogpt_large'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')


class User:
    def __init__(self, model_name='Homer_large', context=None, dialog_models=None, run_dialog=False):
        self.set_model(model_name)
        if context is None:
            self.context = deque()
        else:
            self.context = context
        if dialog_models is None:
            self.dialog_models = ['Homer_medium', 'Bart_medium']
        else:
            self.dialog_models = dialog_models
        self.run_dialog = run_dialog

    def set_model(self, model_name):
        self.model_name = model_name
        self.model = models_dict[self.model_name]

    def add_context(self, text):
        self.context.append(self.model.tokenizer.encode(text + self.model.tokenizer.eos_token, return_tensors='pt'))
        while len(self.context) > self.model.context_len:
            self.context.popleft()


class Model:
    def __init__(self, model_path, tokenizer, context_len):
        self.model = AutoModelWithLMHead.from_pretrained(model_path)
        self.tokenizer = tokenizer
        self.context_len = context_len

    def get_response(self, context):
        input_ids = torch.cat(list(context)[-min(len(context), self.context_len):], dim=-1)
        output_ids = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=200,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        response = self.tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        context.append(self.tokenizer.encode(response + self.tokenizer.eos_token, return_tensors='pt'))
        while len(context) > self.context_len:
            context.popleft()
        return response


# models_dict['Homer_small'] = Model(
#     MODELS_DIR + '/' + 'model_homer_small_7con_Wadditional',
#     tokenizer_dict['dialogpt_small'],
#     5
# )
# models_dict['Bart_small'] = Model(
#     MODELS_DIR + '/' + 'model_bart_small_7con_Wadditional',
#     tokenizer_dict['dialogpt_small'],
#     5
# )
models_dict['Homer_medium'] = Model(
    MODELS_DIR + '/' + 'model_homer_medium_7con_Wadditional',
    tokenizer_dict['dialogpt_medium'],
    4
)
models_dict['Homer_large'] = Model(
    MODELS_DIR + '/' + 'model_homer_large_1con',
    tokenizer_dict['dialogpt_large'],
    1
)
models_dict['Bart_medium'] = Model(
    MODELS_DIR + '/' + 'model_bart_medium_7con_Wadditional',
    tokenizer_dict['dialogpt_medium'],
    4
)
