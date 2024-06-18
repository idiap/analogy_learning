from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor

class Linear(nn.Module):


    def __init__(self):
        super(Linear, self).__init__()
        self.line = nn.Linear(768, 768)
        #No max_seq_length set. Try to infer from model

        self.config_keys = []

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())


    def forward(self, features: Dict[str, Tensor]):
        feat = features['token_embeddings']
        output_tokens = self.line(feat)
        features.update({'output_tokens': output_tokens})

        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Linear(**config)