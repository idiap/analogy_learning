# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


import numpy as np
from sklearn.metrics import  accuracy_score
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)

from datasets import load_dataset
import random
from datasets import set_caching_enabled
set_caching_enabled(False)

import torch
torch.manual_seed(907)
seed = 907

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
    }

for i in range(1, 11):

    datasetall = load_dataset("csv", data_files={'train': os.path.join(PATH,'data/data_full_subtype_train{0}.csv'.format(i)), 'test': os.path.join(PATH,'data/data_full_subtype_test{0}.csv'.format(i))})


    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

    def tokenize_function(x):
        return tokenizer(x["part1"].replace("[SEP]", "</s>") + " </s> " + x["part2"].replace("[SEP]", "</s>"),  padding='max_length', max_length=40)
    datasetall_tok = datasetall.map(tokenize_function)

    train_dat = datasetall_tok['train']
    test_dat = datasetall_tok['test']

    modelconfig = AutoConfig.from_pretrained(pretrained_model_name_or_path='roberta-base', num_labels=2)
    modelconfig.vocab_size =tokenizer.vocab_size

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", config=modelconfig)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),
                    lr = 2e-5
                    )



    trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=f'modelsavebatchfullro', num_train_epochs=3, per_device_train_batch_size = 32, per_device_eval_batch_size=531, evaluation_strategy="steps",
                                   save_steps = 150, eval_steps = 100, save_strategy = "steps", report_to="tensorboard"),
            compute_metrics=compute_metrics, train_dataset=train_dat, eval_dataset= test_dat)
    trainer.place_model_on_device = False

    trainer.train()
    trainer.save_model('romodel{0}/model'.format(i))
    tokenizer.save_pretrained('romodel{0}/tokenizer'.format(i))


