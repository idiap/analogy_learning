import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling, see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_mean_tokens: bool = True,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_mean_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens


        self.pooling_output_dimension = word_embedding_dimension

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []

        if self.pooling_mode_mean_tokens:
            modes.append('mean')



        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        splits = features['sep_index']

        ## Pooling strategy
        output_vectors_1 = []
        output_vectors_2 = []

        for i in range(len(splits)):
            split = int(splits[i])

            input_mask_expanded1 = attention_mask[i][1:split].unsqueeze(-1).expand(token_embeddings[i][1:split].size()).float()
            input_mask_expanded2 = attention_mask[i][split+1:-1].unsqueeze(-1).expand(token_embeddings[i][split+1:-1].size()).float()

            sum_embeddings1 = torch.sum(token_embeddings[i][1:split] * input_mask_expanded1, 0)
            sum_embeddings2 = torch.sum(token_embeddings[i][split+1:-1] * input_mask_expanded2, 0)

            sum_mask1 = input_mask_expanded1.sum(0)
            sum_mask2 = input_mask_expanded2.sum(0)

            sum_mask1 = torch.clamp(sum_mask1, min=1e-9)
            sum_mask2 = torch.clamp(sum_mask2, min=1e-9)

            output_vectors_1.append(sum_embeddings1 / sum_mask1)
            output_vectors_2.append(sum_embeddings2 / sum_mask2)

        output_vector_1 = torch.stack(output_vectors_1)
        output_vector_2 = torch.stack(output_vectors_2)

        features.update({'sentence_embedding_a_c': output_vector_1})
        features.update({'sentence_embedding_b_d': output_vector_2})

        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)
