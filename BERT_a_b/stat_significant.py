# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

train_class = pd.read_csv('test_dat_results_splits.csv')

nontrain_class = pd.read_csv('test_dat_results_splits_untrained.csv')

for col in zip(['overall','pos','neg'], ["overall_num", "pos_num", "neg_num"]):
    print(col)
    for i in range(21):
        print(train_class['name'][i],nontrain_class[col[1]][i],proportions_ztest(np.array([train_class[col[0]][i]*nontrain_class[col[1]][i],nontrain_class[col[0]][i]*nontrain_class[col[1]][i]]), [nontrain_class[col[1]][i],nontrain_class[col[1]][i]])[1])


train_class = pd.read_csv('rank_results_splits.csv')

nontrain_class = pd.read_csv('rank_results_splits_untrained.csv')
print('rank')
for i in range(21):
    print(train_class['name'][i],proportions_ztest(np.array([train_class['overall'][i]*nontrain_class['num'][i],nontrain_class['overall'][i]*nontrain_class['num'][i]]), [nontrain_class['num'][i],nontrain_class['num'][i]])[1])



train_class = pd.read_csv('test_dat_results_semantic_split.csv')

nontrain_class = pd.read_csv('test_dat_results_semantic_split_untrained.csv')
print('semantic')

for col in zip(['overall','high','low'],["noverall","nnear","nfar"]):
    print(col)
    for i in range(9):
        print(train_class['name'][i],nontrain_class[col[1]][i],proportions_ztest(np.array([train_class[col[0]][i]*nontrain_class[col[1]][i],nontrain_class[col[0]][i]*nontrain_class[col[1]][i]]), [nontrain_class[col[1]][i],nontrain_class[col[1]][i]])[1])
              
              

