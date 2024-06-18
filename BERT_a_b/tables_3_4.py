# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from sentence_transformers.evaluation import TestEvaluator
from sentence_transformers import SentenceTransformer, InputExample
import csv
import pandas as pd
import torch
from sentence_transformers import models
model_name = 'bert-base-uncased'
import random
import numpy as np
import sys
import seaborn as sns


#PATH = Add you directory here
import sys
sys.path.append(PATH)


dat = pd.read_csv(os.path.join(PATH,'data/data_full_remall_subtype.csv'))
freq = pd.read_csv(os.path.join(PATH,'data/bert_word_frequencies.csv'))

dat['guess']=-9
dat['guess_train']=-9

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    guess = []
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#    model.load_state_dict(torch.load("ab/output_{0}remalldata".format(i)))

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

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    guess = []
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.load_state_dict(torch.load("ab/output_{0}remalldata".format(i)))

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
    dat.loc[dat['split_10']==i,'guess_train']=guess


dat['train'] = dat['guess_train'] == dat['y']



words = []
for i in range(len(dat)):
    words.append(dat['analogy'][i].split(','))

words = pd.DataFrame(words)
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

words['len1'] = -9
words['len2'] = -9
words['len3'] = -9
words['len4'] = -9
words['total'] = -9
words['ave'] = -9.9
words['num_wordpiece'] = -9
words['freq1'] = -9
words['freq2'] = -9
words['freq3'] = -9
words['freq4'] = -9
words['total_freq'] = -9
words['ave_freq'] = -9.9

for i in range(len(dat)):
    for j in range(0,4):
        words.iloc[i, j] = words.iloc[i, j].strip(' ')
        words.iloc[i, j + 4] = len(model.tokenizer(words.iloc[i, j])['input_ids']) - 2
        if(len(freq[freq['index'] == words.iloc[i, j]]['freq'])==0):
            words.iloc[i, j + 11] = 0
        else:
            words.iloc[i, j + 11] = freq[freq['index'] == words.iloc[i, j]]['freq']


    words['total'][i] = sum(words.iloc[i, 4:8])
    words['ave'][i] = sum(words.iloc[i, 4:8])/4
    words['num_wordpiece'][i] = sum(words.iloc[i, 4:8]>1)
    words['total_freq'][i] = sum(words.iloc[i, 11:15])
    words['ave_freq'][i] = sum(words.iloc[i, 11:15])/4



dat['total'] = words['total']
dat['ave'] = words['ave']
dat['num_wordpiece'] = words['num_wordpiece']
dat['total_freq'] = words['total_freq']
dat['ave_freq'] = words['ave_freq']

print(sum(dat[(dat['y']==1)]['ave_freq'])/len(dat[(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['y']==-1)]['ave_freq'])/len(dat[(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['guess']==1)]['ave_freq'])/len(dat[(dat['guess']==1)]['ave_freq']))
print(sum(dat[(dat['guess']==-1)]['ave_freq'])/len(dat[(dat['guess']==-1)]['ave_freq']))
print(sum(dat[(dat['guess_train']==1)]['ave_freq'])/len(dat[(dat['guess_train']==1)]['ave_freq']))
print(sum(dat[(dat['guess_train']==-1)]['ave_freq'])/len(dat[(dat['guess_train']==-1)]['ave_freq']))


print(sum(dat[(dat['guess']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['guess']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['guess']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['guess']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['guess']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['guess']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['guess']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['guess']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['guess_train']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['guess_train']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='u4')&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='u4')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u4')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u4')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='u2')&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==-1)]['ave_freq']))


print(sum(dat[(dat['type']=='u2')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='u2')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='u2')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='sat')&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==-1)]['ave_freq']))


print(sum(dat[(dat['type']=='sat')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sat')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sat')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq']))



print(sum(dat[(dat['type']=='sci')&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==-1)]['ave_freq']))


print(sum(dat[(dat['type']=='sci')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess']==1)&(dat['y']==-1)]['ave_freq']))

print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==-1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==1)&(dat['y']==1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==-1)&(dat['y']==-1)]['ave_freq']))
print(sum(dat[(dat['type']=='sci')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq'])/len(dat[(dat['type']=='sci')&(dat['guess_train']==1)&(dat['y']==-1)]['ave_freq']))


freq = words['freq1'].values.tolist() +words['freq2'].values.tolist() + words['freq3'].values.tolist()+ words['freq4'].values.tolist()
len = words['len1'].values.tolist() +words['len2'].values.tolist() + words['len3'].values.tolist()+ words['len4'].values.tolist()
type = dat['type'].values.tolist() +dat['type'].values.tolist() + dat['type'].values.tolist()+ dat['type'].values.tolist()
df = pd.DataFrame(list(zip(freq, len, type)),
               columns =['freq', 'len', 'type'])



p = sns.boxplot( x=df['len'], y=df[df['freq']<100000]['freq'],)
p.set_xlabel('# of subwords', fontsize= 14, fontweight='bold')
p.set_ylabel('# of times seen in training data', fontsize= 14, fontweight='bold')

