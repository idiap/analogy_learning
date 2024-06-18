# Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance

# Requirements

To install dependencies:

```pip install -r requirements.txt```

Additionally please clone the [Word Embedding Benchmark (web) package](https://github.com/kudkudak/word-embeddings-benchmarks) into this folder, and rename "**benchmarks**"


# Datasets

For training, we use a combination of 4 datasets. You must retreive all datasets before running the scripts.

SCAN: Please refer [here](https://github.com/taczin/SCAN_analogies) to github for [Czinczoll et. al.](https://arxiv.org/abs/2211.15268)

SAT: This dataset can be requested from [Peter Turney](https://www.apperceptual.com/)

U2/U4: Please refer [here](https://github.com/taczin/SCAN_analogies) to github for [Ushio et. al.](https://arxiv.org/abs/2211.15268) 

Distractor Dataset: Please refer [here](https://osf.io/cd7b9/) for the data from [Jones et. al.](https://link.springer.com/article/10.3758/s13423-022-02062-8#Sec13)

Word Frequencies: Please refer [here](https://github.com/katezhou/cosine_and_frequency) for the data from [Zhou et. al.](https://arxiv.org/abs/2205.05092)


In order to set up the data, add all the datasets to the "data" folder, and run ```format_data.py```.


# Training models

Files to train models and get the results publiches in the tables are all in their respective folders.

For example 'BERT_a_b' contains all files to get the numbers for all "BERT a-b" results presented in the paper. In order to get the results for this particular model you will need to run:
1) ```train_model.py```
2) ```tables_1_2_5_notrain.py``` and ```tables_1_2_5_train.py```
3) ```stat_significant.py```
4) ```table_6.py```

# Simple classification models

For the simple classification models, you will need to upload your trained models (or push them) to [huggingface](https://huggingface.co/) to preproduce the tables.


# Authors and acknowledgment

The following repositories were used and/or edited to produce this research:

- [SBERT](https://www.sbert.net/) (original paper [here](https://arxiv.org/abs/1908.10084)). All model directories contain a folder ```sentence-transformers```, that contain versions of the [SBERT repo](https://github.com/UKPLab/sentence-transformers/tree/master), with modifications made by us relevant to each model, as well as files in the original repo that are not relevant to our experiments removed.

- [Word Embedding Benchmark (web) package](https://github.com/kudkudak/word-embeddings-benchmarks)

# Support

Please contact the first author with any questions

