# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856

from benchmarks.web.analogy import *

import gensim.downloader as api


tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

fast_text_vectors = api.load('fasttext-wiki-news-subwords-300')

w_fast = fast_text_vectors




A = np.vstack(w_fast[word] for word in tasks["SIMLEX999"].X[:, 0])
B = np.vstack(w_fast[word] for word in tasks["SIMLEX999"].X[:, 1])
scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
print(scipy.stats.spearmanr(scores, tasks['SIMLEX999'].y).correlation)

A = np.vstack(w_fast[word] for word in tasks["MEN"].X[:, 0])
B = np.vstack(w_fast[word] for word in tasks["MEN"].X[:, 1])
scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
print(scipy.stats.spearmanr(scores, tasks['MEN'].y).correlation)

A = np.vstack(w_fast[word] for word in tasks["WS353"].X[:, 0])
B = np.vstack(w_fast[word] for word in tasks["WS353"].X[:, 1])
scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
print(scipy.stats.spearmanr(scores, tasks['WS353'].y).correlation)

