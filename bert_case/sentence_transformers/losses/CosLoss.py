import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformer import SentenceTransformer
import logging
import math

logger = logging.getLogger(__name__)

class CosLoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,

                 loss_fct: Callable = nn.CosineEmbeddingLoss()):
        super(CosLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels

        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [[self.model(sentence_feature)['sentence_embedding_a_c'], self.model(sentence_feature)['sentence_embedding_b_d']] for sentence_feature in sentence_features]

        anal1, anal2 = reps
        rep_a, rep_b = anal1
        rep_c, rep_d= anal2

        output1 = rep_a - rep_b
        output2 = rep_c - rep_d
#        output = torch.exp(-torch.mean(torch.abs((rep_a - rep_b) - (rep_c - rep_d)).squeeze(), 1))

        if labels is not None:
            loss = self.loss_fct(output1, output2, labels)
            return loss
        else:
            return reps