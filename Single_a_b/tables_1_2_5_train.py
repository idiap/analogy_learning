# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


from sentence_transformers.evaluation import TestEvaluator
from sentence_transformers import SentenceTransformer, InputExample
import csv
import pandas as pd
from torch import nn
import torch
from sentence_transformers import models
model_name = 'bert-base-uncased'

dat = pd.read_csv(os.path.join(PATH,'data/data_full_remall_subtype.csv'))
dat['guess']=-9

for i in range(1,11):
    guess = []

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    for param in pooling_model.parameters():
        param.requires_grad = False

    new_lin = models.Linear()

    model = SentenceTransformer(modules=[word_embedding_model, new_lin, pooling_model])
    model.load_state_dict(torch.load("single/output_{0}remalldata".format(i)))
    model.eval()
    if i == 10:
        i = 0
    dev_samples = []
    with open(os.path.join(PATH,'data/data_full_remall_subtype.csv', newline='')) as csvfile:
        dataframe = csv.DictReader(csvfile)
        for row in dataframe:
            if row['split_10'] == '{0}'.format(i):
                score = float(row['y'])
                dev_samples.append(InputExample(texts=[row['part1'], row['part2']], label=score))

    test_evaluator = TestEvaluator.from_input_examples(dev_samples, batch_size=len(dev_samples), name='sts-test')
    guess = test_evaluator(model)
    dat.loc[dat['split_10']==i,'guess']=guess


dat['corr'] = dat['guess'] == dat['y']

name = []
acc_overall = []
acc_positive = []
acc_negative = []
overall_num = []
pos_num = []
neg_num = []


name.append('overall')
acc_overall.append(sum(dat['corr'])/len(dat['corr']))
acc_positive.append(sum(dat[dat['y'] == 1]['corr'])/len(dat[dat['y'] == 1]['corr']))
acc_negative.append(sum(dat[dat['y'] == -1]['corr'])/len(dat[dat['y'] == -1]['corr']))
overall_num.append(len(dat['corr']))
pos_num.append(len(dat[dat['y'] == 1]['corr']))
neg_num.append(len(dat[dat['y'] == -1]['corr']))

name.append('sci')
acc_overall.append(sum(dat[dat['type']=='sci']['corr'])/len(dat[dat['type']=='sci']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='sci']['corr'])/len(dat[dat['y'] == 1][dat['type']=='sci']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='sci']['corr'])/len(dat[dat['y'] == -1][dat['type']=='sci']['corr']))
overall_num.append(len(dat[dat['type']=='sci']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['type']=='sci']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['type']=='sci']['corr']))

name.append('science')
acc_overall.append(sum(dat[dat['subtype']=='science']['corr'])/len(dat[dat['subtype']=='science']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='science']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='science']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='science']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='science']['corr']))
overall_num.append(len(dat[dat['subtype']=='science']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='science']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='science']['corr']))

name.append('metaphor')
acc_overall.append(sum(dat[dat['subtype']=='metaphor']['corr'])/len(dat[dat['subtype']=='metaphor']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='metaphor']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='metaphor']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='metaphor']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='metaphor']['corr']))
overall_num.append(len(dat[dat['subtype']=='metaphor']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='metaphor']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='metaphor']['corr']))


name.append('sat')
acc_overall.append(sum(dat[dat['type']=='sat']['corr'])/len(dat[dat['type']=='sat']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='sat']['corr'])/len(dat[dat['y'] == 1][dat['type']=='sat']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='sat']['corr'])/len(dat[dat['y'] == -1][dat['type']=='sat']['corr']))
overall_num.append(len(dat[dat['type']=='sat']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['type']=='sat']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['type']=='sat']['corr']))

name.append('u2')
acc_overall.append(sum(dat[dat['type']=='u2']['corr'])/len(dat[dat['type']=='u2']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='u2']['corr'])/len(dat[dat['y'] == 1][dat['type']=='u2']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='u2']['corr'])/len(dat[dat['y'] == -1][dat['type']=='u2']['corr']))
overall_num.append(len(dat[dat['type']=='u2']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['type']=='u2']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['type']=='u2']['corr']))

name.append('grade4')
acc_overall.append(sum(dat[dat['subtype']=='grade4']['corr'])/len(dat[dat['subtype']=='grade4']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade4']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade4']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade4']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade4']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade4']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade4']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade4']['corr']))

name.append('grade5')
acc_overall.append(sum(dat[dat['subtype']=='grade5']['corr'])/len(dat[dat['subtype']=='grade5']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade5']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade5']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade5']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade5']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade5']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade5']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade5']['corr']))

name.append('grade6')
acc_overall.append(sum(dat[dat['subtype']=='grade6']['corr'])/len(dat[dat['subtype']=='grade6']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade6']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade6']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade6']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade6']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade6']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade6']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade6']['corr']))

name.append('grade7')
acc_overall.append(sum(dat[dat['subtype']=='grade7']['corr'])/len(dat[dat['subtype']=='grade7']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade7']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade7']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade7']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade7']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade7']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade7']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade7']['corr']))

name.append('grade8')
acc_overall.append(sum(dat[dat['subtype']=='grade8']['corr'])/len(dat[dat['subtype']=='grade8']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade8']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade8']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade8']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade8']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade8']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade8']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade8']['corr']))

name.append('grade9')
acc_overall.append(sum(dat[dat['subtype']=='grade9']['corr'])/len(dat[dat['subtype']=='grade9']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade9']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade9']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade9']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade9']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade9']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade9']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade9']['corr']))

name.append('grade10')
acc_overall.append(sum(dat[dat['subtype']=='grade10']['corr'])/len(dat[dat['subtype']=='grade10']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade10']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade10']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade10']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade10']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade10']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade10']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade10']['corr']))

name.append('grade11')
acc_overall.append(sum(dat[dat['subtype']=='grade11']['corr'])/len(dat[dat['subtype']=='grade11']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade11']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade11']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade11']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade11']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade11']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade11']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade11']['corr']))

name.append('grade12')
acc_overall.append(sum(dat[dat['subtype']=='grade12']['corr'])/len(dat[dat['subtype']=='grade12']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade12']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade12']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade12']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade12']['corr']))
overall_num.append(len(dat[dat['subtype']=='grade12']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='grade12']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='grade12']['corr']))

name.append('u4')
acc_overall.append(sum(dat[dat['type']=='u4']['corr'])/len(dat[dat['type']=='u4']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='u4']['corr'])/len(dat[dat['y'] == 1][dat['type']=='u4']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='u4']['corr'])/len(dat[dat['y'] == -1][dat['type']=='u4']['corr']))
overall_num.append(len(dat[dat['type']=='u4']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['type']=='u4']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['type']=='u4']['corr']))

name.append('high-beginning')
acc_overall.append(sum(dat[dat['subtype']=='high-beginning']['corr'])/len(dat[dat['subtype']=='high-beginning']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-beginning']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-beginning']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-beginning']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-beginning']['corr']))
overall_num.append(len(dat[dat['subtype']=='high-beginning']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='high-beginning']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='high-beginning']['corr']))

name.append('low-intermediate')
acc_overall.append(sum(dat[dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['subtype']=='low-intermediate']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='low-intermediate']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='low-intermediate']['corr']))
overall_num.append(len(dat[dat['subtype']=='low-intermediate']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='low-intermediate']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='low-intermediate']['corr']))

name.append('high-intermediate')
acc_overall.append(sum(dat[dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['subtype']=='high-intermediate']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-intermediate']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-intermediate']['corr']))
overall_num.append(len(dat[dat['subtype']=='high-intermediate']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='high-intermediate']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='high-intermediate']['corr']))

name.append('low-advanced')
acc_overall.append(sum(dat[dat['subtype']=='low-advanced']['corr'])/len(dat[dat['subtype']=='low-advanced']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='low-advanced']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='low-advanced']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='low-advanced']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='low-advanced']['corr']))
overall_num.append(len(dat[dat['subtype']=='low-advanced']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='low-advanced']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='low-advanced']['corr']))

name.append('high-advanced')
acc_overall.append(sum(dat[dat['subtype']=='high-advanced']['corr'])/len(dat[dat['subtype']=='high-advanced']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-advanced']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-advanced']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-advanced']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-advanced']['corr']))
overall_num.append(len(dat[dat['subtype']=='high-advanced']['corr']))
pos_num.append(len(dat[dat['y'] == 1][dat['subtype']=='high-advanced']['corr']))
neg_num.append(len(dat[dat['y'] == -1][dat['subtype']=='high-advanced']['corr']))

test_dat = pd.DataFrame([name, acc_overall, acc_positive, acc_negative, overall_num, pos_num, neg_num]).transpose()
test_dat = test_dat.rename(columns={0: "name", 1: "overall", 2: "pos", 3: "neg", 4: "overall_num", 5: "pos_num", 6: "neg_num"})
test_dat.to_csv("test_dat_results_splits.csv")


cos = nn.CosineSimilarity(dim=0)

dat_rank_test_pos = dat[dat['y'] == 1].sort_values('pair_ind').reset_index(drop=True)
dat_rank_test_neg = dat[dat['y'] == -1].sort_values('pair_ind').reset_index(drop=True)
dat_rank_test_pos["guess"] = "NaN"






for i in range(1,11):
    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    for param in pooling_model.parameters():
        param.requires_grad = False

    new_lin = models.Linear()

    model = SentenceTransformer(modules=[word_embedding_model, new_lin, pooling_model])
    model.load_state_dict(torch.load("single/output_{0}remalldata".format(i)))
    model.eval()
    dat_sub = pd.read_csv(os.path.join(PATH,'data/data_full_subtype_test{0}.csv'.format(i)))

    if i == 10:
        i = 0
    dat_sub_rank_test_pos = dat_sub[dat_sub['y'] == 1].sort_values('pair_ind').reset_index(drop=True)
    dat_sub_rank_test_neg = dat_sub[dat_sub['y'] == -1].sort_values('pair_ind').reset_index(drop=True)
    dat_sub_rank_test_pos["sim_right"] = "NaN"
    dat_sub_rank_test_neg["sim_wrong"] = "NaN"
    guess = []
    for j in range(0, len(dat_sub_rank_test_pos)):
        embeddings1 = model.encode(dat_sub_rank_test_pos['part1'][j])
        embeddings2 = model.encode(dat_sub_rank_test_pos['part2'][j])
        a = embeddings1[0][0]
        b = embeddings1[1][0]
        c = embeddings2[0][0]
        d = embeddings2[1][0]
        dat_sub_rank_test_pos["sim_right"][j] = cos((a - b), (c - d))
        embeddings2 = model.encode(dat_sub_rank_test_neg['part2'][j])
        c = embeddings2[0][0]
        d = embeddings2[1][0]
        dat_sub_rank_test_neg["sim_wrong"][j] = cos((a - b), (c - d))
        if dat_sub_rank_test_neg["sim_wrong"][j] > dat_sub_rank_test_pos["sim_right"][j]:
            guess.append(0)
        else:
            guess.append(1)

    dat_rank_test_pos.loc[dat_rank_test_pos['split_10']==i,'guess']=guess


name = []
acc_overall_rank = []
num = []
name.append('overall')
acc_overall_rank.append(sum(dat_rank_test_pos['guess'])/len(dat_rank_test_pos['guess']))
num.append(len(dat_rank_test_pos['guess']))

name.append('sci')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='sci']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='sci']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['type']=='sci']['guess']))

name.append('science')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='science']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='science']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='science']['guess']))

name.append('metaphor')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='metaphor']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='metaphor']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='metaphor']['guess']))

name.append('sat')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='sat']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='sat']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['type']=='sat']['guess']))

name.append('u2')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='u2']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='u2']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['type']=='u2']['guess']))

name.append('grade4')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade4']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade4']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade4']['guess']))

name.append('grade5')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade5']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade5']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade5']['guess']))

name.append('grade6')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade6']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade6']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade6']['guess']))

name.append('grade7')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade7']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade7']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade7']['guess']))

name.append('grade8')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade8']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade8']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade8']['guess']))

name.append('grade9')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade9']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade9']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade9']['guess']))

name.append('grade10')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade10']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade10']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade10']['guess']))

name.append('grade11')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade11']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade11']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade11']['guess']))

name.append('grade12')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade12']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade12']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade12']['guess']))

name.append('u4')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='u4']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='u4']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['type']=='u4']['guess']))

name.append('high-beginning')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-beginning']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-beginning']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-beginning']['guess']))

name.append('low-intermediate')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-intermediate']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-intermediate']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-intermediate']['guess']))

name.append('high-intermediate')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-intermediate']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-intermediate']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-intermediate']['guess']))

name.append('low-advanced')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-advanced']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-advanced']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-advanced']['guess']))

name.append('high-advanced')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-advanced']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-advanced']['guess']))
num.append(len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-advanced']['guess']))

test_dat_rank = pd.DataFrame([name, acc_overall_rank, num]).transpose()
test_dat_rank = test_dat_rank.rename(columns={0: "name", 1: "overall", 2:"num"})
test_dat_rank.to_csv("rank_results_splits.csv")


sim_dat = pd.read_csv(os.path.join(PATH,'data/similarity_analogy.csv'))

for i in range(1,11):
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    for param in pooling_model.parameters():
        param.requires_grad = False

    new_lin = models.Linear()

    model = SentenceTransformer(modules=[word_embedding_model, new_lin, pooling_model])
    model.load_state_dict(torch.load("single/output_{0}remalldata".format(i)))
    model.eval()



    if i == 10:
        i = 0
    sim_dat["sim_d_right{0}".format(i)] = "NaN"
    sim_dat["sim_d_high{0}".format(i)] = "NaN"
    sim_dat["sim_d_low{0}".format(i)] = "NaN"
    sim_dat["guess_high{0}".format(i)] = "NaN"
    sim_dat["guess_low{0}".format(i)] = "NaN"

    for j in range(0, len(sim_dat)):
        embeddings1 = model.encode(sim_dat['a_b_right'][j])
        embeddings2 = model.encode(sim_dat['c_d_right'][j])
        a = embeddings1[0][0]
        b = embeddings1[1][0]
        c = embeddings2[0][0]
        d = embeddings2[1][0]
        sim_dat["sim_d_right{0}".format(i)][j] = cos((a - b), (c - d))
        embeddings2 = model.encode(sim_dat['c_d_high'][j])
        c = embeddings2[0][0]
        d = embeddings2[1][0]
        sim_dat["sim_d_high{0}".format(i)][j] = cos((a - b), (c - d))
        embeddings2 = model.encode(sim_dat['c_d_low'][j])
        c = embeddings2[0][0]
        d = embeddings2[1][0]
        sim_dat["sim_d_low{0}".format(i)][j] = cos((a - b), (c - d))
        if sim_dat["sim_d_high{0}".format(i)][j] > sim_dat["sim_d_right{0}".format(i)][j]:
            sim_dat["guess_high{0}".format(i)][j] = 0
        else:
            sim_dat["guess_high{0}".format(i)][j] = 1
        if sim_dat["sim_d_low{0}".format(i)][j] > sim_dat["sim_d_right{0}".format(i)][j]:
            sim_dat["guess_low{0}".format(i)][j] = 0
        else:
            sim_dat["guess_low{0}".format(i)][j] = 1



sim_dat["guess_high"] = (sim_dat["guess_high1"]+sim_dat["guess_high2"]+sim_dat["guess_high3"]+sim_dat["guess_high4"]+sim_dat["guess_high5"]+sim_dat["guess_high6"]+sim_dat["guess_high7"]+sim_dat["guess_high8"]+sim_dat["guess_high9"]+sim_dat["guess_high0"])/10
sim_dat["guess_low"] = (sim_dat["guess_low1"] + sim_dat["guess_low2"] + sim_dat["guess_low3"] + sim_dat["guess_low4"] + sim_dat["guess_low5"] + sim_dat["guess_low6"] + sim_dat["guess_low7"] + sim_dat["guess_low8"] + sim_dat["guess_low9"] + sim_dat["guess_low0"])/10


name = []
acc_overall = []
acc_high_near = []
acc_low_near = []
acc_high_far = []
acc_low_far = []
overall_num = []
high_num = []
low_num = []

name.append('overall')
name.append('Near')
name.append('Categorical')
name.append('Causal')
name.append('Comp')
name.append('Far')
name.append('Categorical')
name.append('Causal')
name.append('Comp')

acc_overall.append((sum(sim_dat['guess_high']) + sum(sim_dat['guess_low']))/120)
acc_high_near.append(sum(sim_dat['guess_high'])/len(sim_dat['guess_high']))
acc_low_near.append(sum(sim_dat['guess_low'])/len(sim_dat['guess_low']))
overall_num.append(120)
high_num.append(len(sim_dat['guess_high']))
low_num.append(len(sim_dat['guess_low']))


acc_overall.append((sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high']) + sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low']))/60)
acc_high_near.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_low_near.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low']))
overall_num.append(60)
high_num.append(30)
low_num.append(30)



acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)


acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))

acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))
acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))
acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))
overall_num.append(20)
high_num.append(10)
low_num.append(10)
overall_num.append(20)
high_num.append(10)
low_num.append(10)
overall_num.append(20)
high_num.append(10)
low_num.append(10)


acc_overall.append((sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high']) + sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low']))/60)
acc_high_far.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_low_far.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low']))
overall_num.append(60)
high_num.append(30)
low_num.append(30)

acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)

acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))

acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))
acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))
acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))
overall_num.append(20)
high_num.append(10)
low_num.append(10)
overall_num.append(20)
high_num.append(10)
low_num.append(10)
overall_num.append(20)
high_num.append(10)
low_num.append(10)


acc_close = acc_high_near + acc_high_far
acc_far = acc_low_near + acc_low_far

test_dat = pd.DataFrame([name, acc_overall, acc_close, acc_far,overall_num,high_num,low_num]).transpose()
test_dat = test_dat.rename(columns={0: "name", 1: "overall", 2: "high", 3: "low", 4:'noverall', 5:'nnear', 6:'nfar'})
test_dat.to_csv("test_dat_results_semantic_split10.csv")

