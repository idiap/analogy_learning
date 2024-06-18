# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


from transformers import pipeline
import csv
import pandas as pd
from transformers import AutoTokenizer
from torch import nn
#HUG_PATH = ADD HUGGINGFACE USERNAME


import numpy as np


dat_full = pd.read_csv(os.path.join(PATH,'data/data_full_remall_subtype.csv'))
dat_full['guess']=-9
dat_full["text"] = dat_full['part1'].replace("[SEP]", "</s>") + " </s> " + dat_full["part2"].replace("[SEP]", "</s>")

for i in range(1,11):

    print(i)
    dat_sub = pd.read_csv(os.path.join(PATH,'data/data_full_subtype_test{0}.csv'.format(i)))

    dat_sub["text"] = dat_sub['part1'].replace("[SEP]", "</s>") + " </s> " + dat_sub["part2"].replace("[SEP]", "</s>")

    guess = []
    tokenizer = AutoTokenizer.from_pretrained('{0}/roberta_{1}'.format(HUG_PATH,i), use_fast=True)
    pipe = pipeline("text-classification", model='{0}/roberta_{1}'.format(HUG_PATH,i), tokenizer=tokenizer)
    if i == 10:
        i = 0
    for j in range(0, len(dat_sub)):

        if pipe(dat_sub['text'][j])[0].get("label") == 'LABEL_1':
            guess.append(1)
        if pipe(dat_sub['text'][j])[0].get("label") == 'LABEL_0':
            guess.append(-1)

    dat_full.loc[dat_full['split_10']==i,'guess']=guess


dat_full['corr'] = dat_full['guess'] == dat_full['y']
name = []
acc_overall = []
acc_positive = []
acc_negative = []

name.append('overall')
acc_overall.append(sum(dat_full['corr'])/len(dat_full['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1]['corr'])/len(dat_full[dat_full['y'] == 1]['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1]['corr'])/len(dat_full[dat_full['y'] == -1]['corr']))

name.append('sci')
acc_overall.append(sum(dat_full[dat_full['type']=='sci']['corr'])/len(dat_full[dat_full['type']=='sci']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['type']=='sci']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['type']=='sci']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['type']=='sci']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['type']=='sci']['corr']))

name.append('science')
acc_overall.append(sum(dat_full[dat_full['subtype']=='science']['corr'])/len(dat_full[dat_full['subtype']=='science']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='science']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='science']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='science']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='science']['corr']))

name.append('metaphor')
acc_overall.append(sum(dat_full[dat_full['subtype']=='metaphor']['corr'])/len(dat_full[dat_full['subtype']=='metaphor']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='metaphor']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='metaphor']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='metaphor']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='metaphor']['corr']))

name.append('sat')
acc_overall.append(sum(dat_full[dat_full['type']=='sat']['corr'])/len(dat_full[dat_full['type']=='sat']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['type']=='sat']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['type']=='sat']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['type']=='sat']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['type']=='sat']['corr']))

name.append('u2')
acc_overall.append(sum(dat_full[dat_full['type']=='u2']['corr'])/len(dat_full[dat_full['type']=='u2']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['type']=='u2']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['type']=='u2']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['type']=='u2']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['type']=='u2']['corr']))

name.append('grade4')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade4']['corr'])/len(dat_full[dat_full['subtype']=='grade4']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade4']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade4']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade4']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade4']['corr']))

name.append('grade5')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade5']['corr'])/len(dat_full[dat_full['subtype']=='grade5']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade5']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade5']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade5']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade5']['corr']))

name.append('grade6')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade6']['corr'])/len(dat_full[dat_full['subtype']=='grade6']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade6']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade6']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade6']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade6']['corr']))

name.append('grade7')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade7']['corr'])/len(dat_full[dat_full['subtype']=='grade7']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade7']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade7']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade7']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade7']['corr']))

name.append('grade8')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade8']['corr'])/len(dat_full[dat_full['subtype']=='grade8']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade8']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade8']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade8']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade8']['corr']))

name.append('grade9')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade9']['corr'])/len(dat_full[dat_full['subtype']=='grade9']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade9']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade9']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade9']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade9']['corr']))

name.append('grade10')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade10']['corr'])/len(dat_full[dat_full['subtype']=='grade10']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade10']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade10']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade10']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade10']['corr']))

name.append('grade11')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade11']['corr'])/len(dat_full[dat_full['subtype']=='grade11']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade11']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade11']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade11']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade11']['corr']))

name.append('grade12')
acc_overall.append(sum(dat_full[dat_full['subtype']=='grade12']['corr'])/len(dat_full[dat_full['subtype']=='grade12']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade12']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='grade12']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade12']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='grade12']['corr']))

name.append('u4')
acc_overall.append(sum(dat_full[dat_full['type']=='u4']['corr'])/len(dat_full[dat_full['type']=='u4']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['type']=='u4']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['type']=='u4']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['type']=='u4']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['type']=='u4']['corr']))

name.append('high-beginning')
acc_overall.append(sum(dat_full[dat_full['subtype']=='high-beginning']['corr'])/len(dat_full[dat_full['subtype']=='high-beginning']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-beginning']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-beginning']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-beginning']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-beginning']['corr']))

name.append('low-intermediate')
acc_overall.append(sum(dat_full[dat_full['subtype']=='low-intermediate']['corr'])/len(dat_full[dat_full['subtype']=='low-intermediate']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='low-intermediate']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='low-intermediate']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='low-intermediate']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='low-intermediate']['corr']))

name.append('high-intermediate')
acc_overall.append(sum(dat_full[dat_full['subtype']=='high-intermediate']['corr'])/len(dat_full[dat_full['subtype']=='high-intermediate']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-intermediate']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-intermediate']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-intermediate']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-intermediate']['corr']))

name.append('low-advanced')
acc_overall.append(sum(dat_full[dat_full['subtype']=='low-advanced']['corr'])/len(dat_full[dat_full['subtype']=='low-advanced']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='low-advanced']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='low-advanced']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='low-advanced']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='low-advanced']['corr']))

name.append('high-advanced')
acc_overall.append(sum(dat_full[dat_full['subtype']=='high-advanced']['corr'])/len(dat_full[dat_full['subtype']=='high-advanced']['corr']))
acc_positive.append(sum(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-advanced']['corr'])/len(dat_full[dat_full['y'] == 1][dat_full['subtype']=='high-advanced']['corr']))
acc_negative.append(sum(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-advanced']['corr'])/len(dat_full[dat_full['y'] == -1][dat_full['subtype']=='high-advanced']['corr']))

test_dat_full = pd.DataFrame([name, acc_overall, acc_positive, acc_negative]).transpose()
test_dat_full = test_dat_full.rename(columns={0: "name", 1: "overall", 2: "pos", 3: "neg"})
test_dat_full.to_csv("test_dat_results_10fold_roberta.csv")


dat_full = pd.read_csv(os.path.join(PATH,'data/data_full_remall_subtype.csv'))

dat_full_rank_test_pos = dat_full[dat_full['y'] == 1].sort_values('pair_ind').reset_index(drop=True)
dat_full_rank_test_pos["guess"] = "NaN"

for i in range(1,11):

    print(i)
    dat_sub = pd.read_csv(os.path.join(PATH,'data/data_full_subtype_test{0}.csv'.format(i)))

    dat_sub["text"] = dat_sub['part1'].replace("[SEP]", "</s>") + " </s> " + dat_sub["part2"].replace("[SEP]", "</s>")

    guess = []
    tokenizer = AutoTokenizer.from_pretrained('{0}/roberta_{1}'.format(HUG_PATH,i), use_fast=True)
    pipe = pipeline("text-classification", model='{0}/roberta_{1}'.format(HUG_PATH,i), tokenizer=tokenizer)
    if i == 10:
        i = 0
    dat_sub_rank_test_pos = dat_sub[dat_sub['y'] == 1].sort_values('pair_ind').reset_index(drop=True)
    dat_sub_rank_test_neg = dat_sub[dat_sub['y'] == -1].sort_values('pair_ind').reset_index(drop=True)
    dat_sub_rank_test_pos["sim_right"] = "NaN"
    dat_sub_rank_test_neg["sim_wrong"] = "NaN"
    for j in range(0, len(dat_sub_rank_test_pos)):
        dat_sub_rank_test_pos["sim_right"] = pipe(dat_sub_rank_test_pos['text'][j])[0].get('score')
        dat_sub_rank_test_neg["sim_wrong"] = pipe(dat_sub_rank_test_neg['text'][j])[0].get('score')

        if dat_sub_rank_test_neg["sim_wrong"][j] > dat_sub_rank_test_pos["sim_right"][j]:
            guess.append(0)
        else:
            guess.append(1)

    dat_full_rank_test_pos.loc[dat_full_rank_test_pos['split_10']==i,'guess']=guess


name = []
acc_overall_rank = []

name.append('overall')
acc_overall_rank.append(sum(dat_full_rank_test_pos['guess'])/len(dat_full_rank_test_pos['guess']))

name.append('sci')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='sci']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='sci']['guess']))

name.append('science')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='science']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='science']['guess']))

name.append('metaphor')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='metaphor']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='metaphor']['guess']))

name.append('sat')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='sat']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='sat']['guess']))

name.append('u2')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='u2']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='u2']['guess']))

name.append('grade4')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade4']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade4']['guess']))

name.append('grade5')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade5']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade5']['guess']))

name.append('grade6')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade6']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade6']['guess']))

name.append('grade7')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade7']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade7']['guess']))

name.append('grade8')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade8']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade8']['guess']))

name.append('grade9')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade9']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade9']['guess']))

name.append('grade10')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade10']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade10']['guess']))

name.append('grade11')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade11']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade11']['guess']))

name.append('grade12')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade12']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='grade12']['guess']))

name.append('u4')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='u4']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['type']=='u4']['guess']))

name.append('high-beginning')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-beginning']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-beginning']['guess']))

name.append('low-intermediate')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='low-intermediate']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='low-intermediate']['guess']))

name.append('high-intermediate')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-intermediate']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-intermediate']['guess']))

name.append('low-advanced')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='low-advanced']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='low-advanced']['guess']))

name.append('high-advanced')
acc_overall_rank.append(sum(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-advanced']['guess'])/len(dat_full_rank_test_pos[dat_full_rank_test_pos['subtype']=='high-advanced']['guess']))

test_dat_full_rank = pd.DataFrame([name, acc_overall_rank]).transpose()
test_dat_full_rank = test_dat_full_rank.rename(columns={0: "name", 1: "overall"})
test_dat_full_rank.to_csv("rank_results_10fold_roberta.csv")


sim_dat = pd.read_csv(os.path.join(PATH,'data/similarity_analogy.csv'))

for i in range(1,11):

    print(i)


    guess = []
    tokenizer = AutoTokenizer.from_pretrained('{0}/roberta_{1}'.format(HUG_PATH,i), use_fast=True)
    pipe = pipeline("text-classification", model='{0}/roberta_{1}'.format(HUG_PATH,i), tokenizer=tokenizer)
    if i == 10:
        i = 0
    sim_dat["sim_d_right{0}".format(i)] = "NaN"
    sim_dat["sim_d_high{0}".format(i)] = "NaN"
    sim_dat["sim_d_low{0}".format(i)] = "NaN"
    sim_dat["guess_high{0}".format(i)] = "NaN"
    sim_dat["guess_low{0}".format(i)] = "NaN"

    for j in range(0, len(sim_dat)):
        sim_dat["sim_d_right{0}".format(i)][j] = pipe(sim_dat['a_b_right'][j].replace("[SEP]", "</s>") + ' </s> ' + sim_dat["c_d_right"][j].replace("[SEP]", "</s>"))[0].get('score')
        sim_dat["sim_d_high{0}".format(i)][j] = pipe(sim_dat['a_b_right'][j].replace("[SEP]", "</s>") + ' </s> ' + sim_dat["c_d_high"][j].replace("[SEP]", "</s>"))[0].get('score')
        sim_dat["sim_d_low{0}".format(i)][j] = pipe(sim_dat['a_b_right'][j].replace("[SEP]", "</s>") + ' </s> ' + sim_dat["c_d_low"][j].replace("[SEP]", "</s>"))[0].get('score')
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


acc_overall.append((sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high']) + sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low']))/60)
acc_high_near.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_low_near.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Near']['guess_low']))



acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))/20)


acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))
acc_high_near.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_high']))

acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))
acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))
acc_low_near.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Near']['guess_low']))


acc_overall.append((sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high']) + sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low']))/60)
acc_high_far.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_low_far.append(sum(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Distance_analogy'] == 'Far']['guess_low']))


acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)
acc_overall.append((sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])+sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))/20)

acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))
acc_high_far.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_high']))

acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Categorical'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))
acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Causal'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))
acc_low_far.append(sum(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low'])/len(sim_dat[sim_dat['Type'] == 'Comp'][sim_dat['Distance_analogy'] == 'Far']['guess_low']))

acc_close = acc_high_near + acc_high_far
acc_far = acc_low_near + acc_low_far

test_dat = pd.DataFrame([name, acc_overall, acc_close, acc_far]).transpose()
test_dat = test_dat.rename(columns={0: "name", 1: "overall", 2: "high", 3: "low"})
test_dat.to_csv("test_dat_results_semantic_split10_roberta.csv")

