# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


###PATH = Add you directory here
import sys
sys.path.append(PATH)

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import csv
import numpy as np
import torch
import random
import os
os.mkdir('ab')

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-cased'

# Read the dataset
train_batch_size = 32
seed = 907

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model_save_path = 'output/training_nli_Abpairs'

for i in range(0,10):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)


    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_samples = []
    with open(os.path.join(PATH,'/data/data_full_remall_subtype.csv', newline='')) as csvfile:
        dataframe = csv.DictReader(csvfile)
        for row in dataframe:
            if row['split_10'] != '{0}'.format(i):
                score = float(row['y'])
                train_samples.append(InputExample(texts=[row['part1'], row['part2']], label=score))


    train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)
    train_loss = losses.CosLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

    dev_samples = []
    with open(os.path.join(PATH,'/data/data_full_remall_subtype.csv', newline='')) as csvfile:
        dataframe = csv.DictReader(csvfile)
        for row in dataframe:
            if row['split_10'] == '{0}'.format(i):
                score = float(row['y'])
                dev_samples.append(InputExample(texts=[row['part1'], row['part2']], label=score))

    #dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=531, name='sts-dev')

    # Configure the training
    num_epochs = 3

    model.fit(train_objectives=[(train_dataloader, train_loss)],
    #          evaluator=dev_evaluator,
              epochs=num_epochs,
    #          evaluation_steps=15,
              warmup_steps=0,
              save_best_model = True
              )

    if i == 0:
        i = 10

    torch.save(model.state_dict(), "ab/output_{0}remalldata".format(i))



