# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


from gensim.models import FastText
import gensim.downloader as api
from torch import nn
import csv
import pandas as pd
import torch
cos = nn.CosineSimilarity(dim=0)

fast_text_vectors = api.load('fasttext-wiki-news-subwords-300')

dat = pd.read_csv(os.path.join(PATH,'data/data_full_remall_subtype.csv'))
word1 = [word.split(' [SEP] ')[0].strip().lower() for word in dat['part1']]
word2 = [word.split(' [SEP] ')[1].strip().lower() for word in dat['part1']]
word3 = [word.split(' [SEP] ')[0].strip().lower() for word in dat['part2']]
word4 = [word.split(' [SEP] ')[1].strip().lower() for word in dat['part2']]

cos_similarity = []


for i in range(0,len(dat)):
    if ((len(word1[i].split(' '))==1) & (len(word2[i].split(' '))==1) & (len(word3[i].split(' '))==1) & (len(word4[i].split(' '))==1)):
        if (word1[i] in fast_text_vectors) & (word2[i] in fast_text_vectors) & (word3[i] in fast_text_vectors) & (word4[i] in fast_text_vectors):
            cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-fast_text_vectors[word2[i]]), torch.Tensor(fast_text_vectors[word3[i]]-fast_text_vectors[word4[i]])))
        else:
            cos_similarity.append('NaN')
    elif(len(word1[i].split(' '))!=1):
        words = word1[i].split(' ')
        cos_similarity.append(cos(torch.Tensor((fast_text_vectors[words[0]]+fast_text_vectors[words[1]])/2-fast_text_vectors[word2[i]]), torch.Tensor(fast_text_vectors[word3[i]]-fast_text_vectors[word4[i]])))
    elif ((len(word2[i].split(' ')) != 1)&(len(word4[i].split(' ')) == 1)):
        words = word2[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-(fast_text_vectors[words[0]]+fast_text_vectors[words[1]])/2), torch.Tensor(fast_text_vectors[word3[i]]-fast_text_vectors[word4[i]])))
    elif ((len(word2[i].split(' ')) != 1)&(len(word3[i].split(' ')) != 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word2[i].split(' ')
        words2 = word3[i].split(' ')
        words3 = word4[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-(fast_text_vectors[words1[0]]+fast_text_vectors[words1[1]])/2), torch.Tensor((fast_text_vectors[words2[0]]+fast_text_vectors[words2[1]])/2-(fast_text_vectors[words3[0]]+fast_text_vectors[words3[1]])/2)))
    elif ((len(word2[i].split(' ')) != 1)&(len(word3[i].split(' ')) == 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word2[i].split(' ')
        words2 = word4[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-(fast_text_vectors[words1[0]]+fast_text_vectors[words1[1]])/2), torch.Tensor(fast_text_vectors[word3[i]]-(fast_text_vectors[words2[0]]+fast_text_vectors[words2[1]])/2)))
    elif ((len(word2[i].split(' ')) == 1)&(len(word3[i].split(' ')) != 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word3[i].split(' ')
        words2 = word4[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-fast_text_vectors[word2[i]]), torch.Tensor((fast_text_vectors[words1[0]]+fast_text_vectors[words1[1]])/2-(fast_text_vectors[words2[0]]+fast_text_vectors[words2[1]])/2)))

    elif (len(word3[i].split(' ')) != 1):
        words = word3[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-fast_text_vectors[word2[i]]), torch.Tensor((fast_text_vectors[words[0]]+fast_text_vectors[words[1]])/2-fast_text_vectors[word4[i]])))
    elif (len(word4[i].split(' ')) != 1):
        words = word4[i].split(' ')
        cos_similarity.append(cos(torch.Tensor(fast_text_vectors[word1[i]]-fast_text_vectors[word2[i]]), torch.Tensor(fast_text_vectors[word3[i]]-(fast_text_vectors[words[0]]+fast_text_vectors[words[1]])/2)))
    else:
        cos_similarity.append('NaN')

dat['cos'] = cos_similarity
dat = dat[dat['cos']!='NaN']
guess = [1 if cos >0.5 else -1 for cos in dat['cos']]
dat['guess'] = guess
dat['corr'] = dat['guess'] == dat['y']

name = []
acc_overall = []
acc_positive = []
acc_negative = []

name.append('overall')
acc_overall.append(sum(dat['corr'])/len(dat['corr']))
acc_positive.append(sum(dat[dat['y'] == 1]['corr'])/len(dat[dat['y'] == 1]['corr']))
acc_negative.append(sum(dat[dat['y'] == -1]['corr'])/len(dat[dat['y'] == -1]['corr']))



name.append('sci')
acc_overall.append(sum(dat[dat['type']=='sci']['corr'])/len(dat[dat['type']=='sci']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='sci']['corr'])/len(dat[dat['y'] == 1][dat['type']=='sci']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='sci']['corr'])/len(dat[dat['y'] == -1][dat['type']=='sci']['corr']))



name.append('science')
acc_overall.append(sum(dat[dat['subtype']=='science']['corr'])/len(dat[dat['subtype']=='science']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='science']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='science']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='science']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='science']['corr']))

name.append('metaphor')
acc_overall.append(sum(dat[dat['subtype']=='metaphor']['corr'])/len(dat[dat['subtype']=='metaphor']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='metaphor']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='metaphor']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='metaphor']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='metaphor']['corr']))

name.append('sat')
acc_overall.append(sum(dat[dat['type']=='sat']['corr'])/len(dat[dat['type']=='sat']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='sat']['corr'])/len(dat[dat['y'] == 1][dat['type']=='sat']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='sat']['corr'])/len(dat[dat['y'] == -1][dat['type']=='sat']['corr']))

name.append('u2')
acc_overall.append(sum(dat[dat['type']=='u2']['corr'])/len(dat[dat['type']=='u2']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='u2']['corr'])/len(dat[dat['y'] == 1][dat['type']=='u2']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='u2']['corr'])/len(dat[dat['y'] == -1][dat['type']=='u2']['corr']))

name.append('grade4')
acc_overall.append(sum(dat[dat['subtype']=='grade4']['corr'])/len(dat[dat['subtype']=='grade4']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade4']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade4']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade4']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade4']['corr']))

name.append('grade5')
acc_overall.append(sum(dat[dat['subtype']=='grade5']['corr'])/len(dat[dat['subtype']=='grade5']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade5']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade5']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade5']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade5']['corr']))

name.append('grade6')
acc_overall.append(sum(dat[dat['subtype']=='grade6']['corr'])/len(dat[dat['subtype']=='grade6']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade6']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade6']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade6']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade6']['corr']))

name.append('grade7')
acc_overall.append(sum(dat[dat['subtype']=='grade7']['corr'])/len(dat[dat['subtype']=='grade7']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade7']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade7']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade7']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade7']['corr']))

name.append('grade8')
acc_overall.append(sum(dat[dat['subtype']=='grade8']['corr'])/len(dat[dat['subtype']=='grade8']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade8']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade8']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade8']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade8']['corr']))

name.append('grade9')
acc_overall.append(sum(dat[dat['subtype']=='grade9']['corr'])/len(dat[dat['subtype']=='grade9']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade9']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade9']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade9']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade9']['corr']))

name.append('grade10')
acc_overall.append(sum(dat[dat['subtype']=='grade10']['corr'])/len(dat[dat['subtype']=='grade10']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade10']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade10']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade10']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade10']['corr']))

name.append('grade11')
acc_overall.append(sum(dat[dat['subtype']=='grade11']['corr'])/len(dat[dat['subtype']=='grade11']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade11']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade11']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade11']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade11']['corr']))

name.append('grade12')
acc_overall.append(sum(dat[dat['subtype']=='grade12']['corr'])/len(dat[dat['subtype']=='grade12']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='grade12']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='grade12']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='grade12']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='grade12']['corr']))

name.append('u4')
acc_overall.append(sum(dat[dat['type']=='u4']['corr'])/len(dat[dat['type']=='u4']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['type']=='u4']['corr'])/len(dat[dat['y'] == 1][dat['type']=='u4']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['type']=='u4']['corr'])/len(dat[dat['y'] == -1][dat['type']=='u4']['corr']))

name.append('high-beginning')
acc_overall.append(sum(dat[dat['subtype']=='high-beginning']['corr'])/len(dat[dat['subtype']=='high-beginning']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-beginning']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-beginning']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-beginning']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-beginning']['corr']))

name.append('low-intermediate')
acc_overall.append(sum(dat[dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['subtype']=='low-intermediate']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='low-intermediate']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='low-intermediate']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='low-intermediate']['corr']))

name.append('high-intermediate')
acc_overall.append(sum(dat[dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['subtype']=='high-intermediate']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-intermediate']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-intermediate']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-intermediate']['corr']))

name.append('low-advanced')
acc_overall.append(sum(dat[dat['subtype']=='low-advanced']['corr'])/len(dat[dat['subtype']=='low-advanced']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='low-advanced']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='low-advanced']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='low-advanced']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='low-advanced']['corr']))

name.append('high-advanced')
acc_overall.append(sum(dat[dat['subtype']=='high-advanced']['corr'])/len(dat[dat['subtype']=='high-advanced']['corr']))
acc_positive.append(sum(dat[dat['y'] == 1][dat['subtype']=='high-advanced']['corr'])/len(dat[dat['y'] == 1][dat['subtype']=='high-advanced']['corr']))
acc_negative.append(sum(dat[dat['y'] == -1][dat['subtype']=='high-advanced']['corr'])/len(dat[dat['y'] == -1][dat['subtype']=='high-advanced']['corr']))

test_dat = pd.DataFrame([name, acc_overall, acc_positive, acc_negative]).transpose()
test_dat = test_dat.rename(columns={0: "name", 1: "overall", 2: "pos", 3: "neg"})
test_dat.to_csv("test_dat_results_splits.csv")





dat_rank_test_pos = dat[dat['y'] == 1].sort_values('pair_ind_all').reset_index(drop=True)
dat_rank_test_neg = dat[dat['y'] == -1].sort_values('pair_ind_all').reset_index(drop=True)


dat_rank_test_pos["sim_right"] = "NaN"
dat_rank_test_neg["sim_wrong"] = "NaN"
dat_rank_test_pos["guess"] = "NaN"

word1 = [word.split(' [SEP] ')[0].strip().lower() for word in dat_rank_test_pos['part1']]
word2 = [word.split(' [SEP] ')[1].strip().lower() for word in dat_rank_test_pos['part1']]
word3 = [word.split(' [SEP] ')[0].strip().lower() for word in dat_rank_test_pos['part2']]
word4 = [word.split(' [SEP] ')[1].strip().lower() for word in dat_rank_test_pos['part2']]

word5 = [word.split(' [SEP] ')[0].strip().lower() for word in dat_rank_test_neg['part2']]
word6 = [word.split(' [SEP] ')[1].strip().lower() for word in dat_rank_test_neg['part2']]

for i in range(0, len(dat_rank_test_pos)):

    if ((len(word1[i].split(' '))==1) & (len(word2[i].split(' '))==1) & (len(word3[i].split(' '))==1) & (len(word4[i].split(' '))==1)& (len(word5[i].split(' '))==1) & (len(word6[i].split(' '))==1)):
        if (word1[i] in fast_text_vectors) & (word2[i] in fast_text_vectors) & (word3[i] in fast_text_vectors) & (word4[i] in fast_text_vectors):
            a = torch.Tensor(fast_text_vectors[word1[i]])
            b = torch.Tensor(fast_text_vectors[word2[i]])
            c = torch.Tensor(fast_text_vectors[word3[i]])
            d = torch.Tensor(fast_text_vectors[word4[i]])
            dat_rank_test_pos["sim_right"][i] = cos((a-b), (c-d))
            c = torch.Tensor(fast_text_vectors[word5[i]])
            d = torch.Tensor(fast_text_vectors[word6[i]])
            dat_rank_test_neg["sim_wrong"][i] = cos((a-b), (c-d))
            if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
                dat_rank_test_pos["guess"][i] = 0
            else:
                dat_rank_test_pos["guess"][i] = 1
        else:
            dat_rank_test_pos["guess"][i] = 'NaN'

    elif(len(word1[i].split(' '))!=1):
        words = word1[i].split(' ')
        a = (torch.Tensor(fast_text_vectors[words[0]]) +torch.Tensor(fast_text_vectors[words[1]]))/2
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a-b), (c-d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a-b), (c-d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word2[i].split(' ')) != 1)&(len(word4[i].split(' ')) == 1)&(len(word5[i].split(' ')) == 1)):
        words = word2[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = (torch.Tensor(fast_text_vectors[words[0]]) +torch.Tensor(fast_text_vectors[words[1]]))/2
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a-b), (c-d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a-b), (c-d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word2[i].split(' ')) != 1)&(len(word5[i].split(' ')) != 1)&(len(word6[i].split(' ')) != 1)):

        words1 = word2[i].split(' ')
        words2 = word5[i].split(' ')
        words3 = word6[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = (torch.Tensor(fast_text_vectors[words1[0]]) +torch.Tensor(fast_text_vectors[words1[1]]))/2
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = (torch.Tensor(fast_text_vectors[words2[0]]) +torch.Tensor(fast_text_vectors[words2[1]]))/2
        d = (torch.Tensor(fast_text_vectors[words3[0]]) +torch.Tensor(fast_text_vectors[words3[1]]))/2
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word2[i].split(' ')) != 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word2[i].split(' ')
        words2 = word4[i].split(' ')

        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = (torch.Tensor(fast_text_vectors[words1[0]]) +torch.Tensor(fast_text_vectors[words1[1]]))/2
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = (torch.Tensor(fast_text_vectors[words2[0]]) +torch.Tensor(fast_text_vectors[words2[1]]))/2
        dat_rank_test_pos["sim_right"][i] = cos((a-b), (c-d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a-b), (c-d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1
    elif ((len(word3[i].split(' ')) != 1)&(len(word4[i].split(' ')) == 1)):
        words = word3[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = (torch.Tensor(fast_text_vectors[words[0]]) + torch.Tensor(fast_text_vectors[words[1]])) / 2
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1
    elif ((len(word3[i].split(' ')) == 1)&(len(word4[i].split(' ')) != 1)):
        words = word4[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = (torch.Tensor(fast_text_vectors[words[0]]) + torch.Tensor(fast_text_vectors[words[1]])) / 2
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1
    elif ((len(word2[i].split(' ')) != 1)&(len(word3[i].split(' ')) != 1)&(len(word4[i].split(' ')) != 1)):

        words1 = word2[i].split(' ')
        words2 = word3[i].split(' ')
        words3 = word4[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = (torch.Tensor(fast_text_vectors[words1[0]]) + torch.Tensor(fast_text_vectors[words1[1]])) / 2
        c = (torch.Tensor(fast_text_vectors[words2[0]]) + torch.Tensor(fast_text_vectors[words2[1]])) / 2
        d = (torch.Tensor(fast_text_vectors[words3[0]]) + torch.Tensor(fast_text_vectors[words3[1]])) / 2
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word2[i].split(' ')) != 1)&(len(word3[i].split(' ')) == 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word2[i].split(' ')
        words2 = word4[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = (torch.Tensor(fast_text_vectors[words1[0]]) + torch.Tensor(fast_text_vectors[words1[1]])) / 2
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = (torch.Tensor(fast_text_vectors[words2[0]]) + torch.Tensor(fast_text_vectors[words2[1]])) / 2
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word2[i].split(' ')) == 1)&(len(word3[i].split(' ')) != 1)&(len(word4[i].split(' ')) != 1)):
        words1 = word3[i].split(' ')
        words2 = word4[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = (torch.Tensor(fast_text_vectors[words1[0]]) + torch.Tensor(fast_text_vectors[words1[1]])) / 2
        d = (torch.Tensor(fast_text_vectors[words2[0]]) + torch.Tensor(fast_text_vectors[words2[1]])) / 2
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word5[i].split(' ')) != 1) & (len(word6[i].split(' ')) == 1)):
        words = word5[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c =(torch.Tensor(fast_text_vectors[words[0]]) + torch.Tensor(fast_text_vectors[words[1]])) / 2
        d = torch.Tensor(fast_text_vectors[word6[i]])
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1
    elif ((len(word5[i].split(' ')) == 1) & (len(word6[i].split(' ')) != 1)):
        words = word6[i].split(' ')
        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c = torch.Tensor(fast_text_vectors[word5[i]])
        d = (torch.Tensor(fast_text_vectors[words[0]]) + torch.Tensor(fast_text_vectors[words[1]]))/2
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1

    elif ((len(word5[i].split(' ')) != 1) & (len(word6[i].split(' ')) != 1)):
        words1 = word5[i].split(' ')
        words2 = word6[i].split(' ')

        a = torch.Tensor(fast_text_vectors[word1[i]])
        b = torch.Tensor(fast_text_vectors[word2[i]])
        c = torch.Tensor(fast_text_vectors[word3[i]])
        d = torch.Tensor(fast_text_vectors[word4[i]])
        dat_rank_test_pos["sim_right"][i] = cos((a - b), (c - d))
        c =(torch.Tensor(fast_text_vectors[words1[0]]) + torch.Tensor(fast_text_vectors[words1[1]])) / 2
        d = (torch.Tensor(fast_text_vectors[words2[0]]) + torch.Tensor(fast_text_vectors[words2[1]])) / 2
        dat_rank_test_neg["sim_wrong"][i] = cos((a - b), (c - d))
        if dat_rank_test_neg["sim_wrong"][i] > dat_rank_test_pos["sim_right"][i]:
            dat_rank_test_pos["guess"][i] = 0
        else:
            dat_rank_test_pos["guess"][i] = 1


    else:
        print(i)



dat_rank_test_pos = dat_rank_test_pos[dat_rank_test_pos['guess']!='NaN']

name = []
acc_overall_rank = []

name.append('overall')
acc_overall_rank.append(sum(dat_rank_test_pos['guess'])/len(dat_rank_test_pos['guess']))

name.append('sci')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='sci']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='sci']['guess']))

name.append('science')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='science']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='science']['guess']))

name.append('metaphor')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='metaphor']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='metaphor']['guess']))

name.append('sat')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='sat']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='sat']['guess']))

name.append('u2')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='u2']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='u2']['guess']))

name.append('grade4')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade4']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade4']['guess']))

name.append('grade5')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade5']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade5']['guess']))

name.append('grade6')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade6']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade6']['guess']))

name.append('grade7')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade7']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade7']['guess']))

name.append('grade8')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade8']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade8']['guess']))

name.append('grade9')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade9']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade9']['guess']))

name.append('grade10')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade10']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade10']['guess']))

name.append('grade11')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade11']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade11']['guess']))

name.append('grade12')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade12']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='grade12']['guess']))

name.append('u4')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['type']=='u4']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['type']=='u4']['guess']))

name.append('high-beginning')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-beginning']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-beginning']['guess']))

name.append('low-intermediate')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-intermediate']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-intermediate']['guess']))

name.append('high-intermediate')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-intermediate']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-intermediate']['guess']))

name.append('low-advanced')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-advanced']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='low-advanced']['guess']))

name.append('high-advanced')
acc_overall_rank.append(sum(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-advanced']['guess'])/len(dat_rank_test_pos[dat_rank_test_pos['subtype']=='high-advanced']['guess']))

test_dat_rank = pd.DataFrame([name, acc_overall_rank]).transpose()
test_dat_rank = test_dat_rank.rename(columns={0: "name", 1: "overall"})
test_dat_rank.to_csv("rank_results_splits.csv")



sim_dat = pd.read_csv(os.path.join(PATH,'data/similarity_analogy.csv'))


sim_dat["sim_d_right"] = "NaN"
sim_dat["sim_d_high"] = "NaN"
sim_dat["sim_d_low"] = "NaN"
sim_dat["guess_high"] = "NaN"
sim_dat["guess_low"] = "NaN"

for i in range(0, len(sim_dat)):
    a = torch.Tensor(fast_text_vectors[sim_dat['a_b_right'][i].split(' [SEP] ')[0].lower()])
    b = torch.Tensor(fast_text_vectors[sim_dat['a_b_right'][i].split(' [SEP] ')[1].lower()])
    c = torch.Tensor(fast_text_vectors[sim_dat['c_d_right'][i].split(' [SEP] ')[0].lower()])
    d = torch.Tensor(fast_text_vectors[sim_dat['c_d_right'][i].split(' [SEP] ')[1].lower()])
    sim_dat["sim_d_right"][i] = cos((a-b), (c-d))
    c = torch.Tensor(fast_text_vectors[sim_dat['c_d_high'][i].split(' [SEP] ')[0].lower()])
    d = torch.Tensor(fast_text_vectors[sim_dat['c_d_high'][i].split(' [SEP] ')[1].lower()])
    sim_dat["sim_d_high"][i] = cos((a-b), (c-d))
    c = torch.Tensor(fast_text_vectors[sim_dat['c_d_low'][i].split(' [SEP] ')[0].lower()])
    d = torch.Tensor(fast_text_vectors[sim_dat['c_d_low'][i].split(' [SEP] ')[1].lower()])
    sim_dat["sim_d_low"][i] = cos((a-b), (c-d))
    if sim_dat["sim_d_high"][i] > sim_dat["sim_d_right"][i]:
        sim_dat["guess_high"][i] = 0
    else:
        sim_dat["guess_high"][i] = 1
    if sim_dat["sim_d_low"][i] > sim_dat["sim_d_right"][i]:
        sim_dat["guess_low"][i] = 0
    else:
        sim_dat["guess_low"][i] = 1


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
test_dat.to_csv("test_dat_results_semantic.csv")
