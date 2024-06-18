

#PATH = Add you directory here
import sys
sys.path.append(PATH)

#HUG_PATH = ADD HUGGINGFACE USERNAME


from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from benchmarks.web.analogy import *

from transformers import pipeline
from transformers import AutoTokenizer
import numpy as np
import scipy
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

corr_simlex = []
corr_men = []
corr_w = []

for i in range(1,11):
    tokenizer = AutoTokenizer.from_pretrained('{0}/largemodel_{1}'.format(HUG_PATH,i), use_fast=True)
    pipe = pipeline("text-classification", model='{0}/largemodel_{1}'.format(HUG_PATH,i), tokenizer=tokenizer)

    #mean_vector = np.mean(w_fast.vectors, axis=0, keepdims=True)
    A = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["SIMLEX999"].X[:, 0])
    B = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["SIMLEX999"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_simlex.append(scipy.stats.spearmanr(scores, tasks['SIMLEX999'].y).correlation)


    #mean_vector = np.mean(w_fast.vectors, axis=0, keepdims=True)
    A = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["MEN"].X[:, 0])
    B = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["MEN"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_men.append(scipy.stats.spearmanr(scores, tasks['MEN'].y).correlation)

    #mean_vector = np.mean(w_fast.vectors, axis=0, keepdims=True)
    A = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["WS353"].X[:, 0])
    B = np.vstack(np.array(pipe(word,return_tensors = "pt")).squeeze()[1:-1].mean(axis= 0) for word in tasks["WS353"].X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    corr_w.append(scipy.stats.spearmanr(scores, tasks['WS353'].y).correlation)


print(sum(corr_simlex)/10)
print(sum(corr_men)/10)
print(sum(corr_w)/10)
