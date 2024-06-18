# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Molly Petersen <molly.petersen@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only


#PATH = Add you directory here
import sys
sys.path.append(PATH)


from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999

from benchmarks.web.analogy import *
import torch
from sentence_transformers import SentenceTransformer
import random
from sentence_transformers import models
import numpy as np
model_name = 'bert-base-uncased'
seed = 907
import scipy
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

corr_simlex = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["SIMLEX999"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["SIMLEX999"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_simlex.append(scipy.stats.spearmanr(scores, tasks['SIMLEX999'].y).correlation)
print("nontuned", sum(corr_simlex)/10)



corr_men = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["MEN"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["MEN"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_men.append(scipy.stats.spearmanr(scores, tasks['MEN'].y).correlation)
print("nontuned", sum(corr_men)/10)



corr_w = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["WS353"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["WS353"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_w.append(scipy.stats.spearmanr(scores, tasks['WS353'].y).correlation)
print("nontuned", sum(corr_w)/10)




corr_simlex = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.load_state_dict(torch.load("ab/output_{0}remalldata".format(i)))
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["SIMLEX999"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["SIMLEX999"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_simlex.append(scipy.stats.spearmanr(scores, tasks['SIMLEX999'].y).correlation)
print("tuned", sum(corr_simlex)/10)



corr_men = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.load_state_dict(torch.load("ab/output_{0}remalldata".format(i)))
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["MEN"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["MEN"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_men.append(scipy.stats.spearmanr(scores, tasks['MEN'].y).correlation)
print("tuned", sum(corr_men)/10)



corr_w = []

for i in range(1,11):
    seed = 907

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.load_state_dict(torch.load("ab/output_{0}remalldata".format(i)))
    model.eval()

    A = np.vstack(model.encode(word)[0][0] for word in tasks["WS353"].X[:, 0])
    B = np.vstack(model.encode(word)[0][0] for word in tasks["WS353"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_w.append(scipy.stats.spearmanr(scores, tasks['WS353'].y).correlation)
print("tuned", sum(corr_w)/10)
