# Text Classification Repository

This repository contains various implementations of text classification models using different techniques and libraries such as NLTK, Gensim, GloVe, BERT, and Hugging Face's Transformers. The models are trained and tested on a spam detection dataset.

## Contents

1. `text_classification_word2vec.py`: This script uses Word2Vec for word embeddings and trains a Logistic Regression and Naive Bayes model for spam detection.
2. `text_classification_glove.py`: This script uses GloVe for word embeddings and trains a Logistic Regression and Naive Bayes model for spam detection.
3. `text_classification_bert.py`: This script uses BERT for word embeddings and trains a Logistic Regression and Naive Bayes model for spam detection.
4. `text_classification_transformers.py`: This script uses Hugging Face's Transformers library to train a DistilBERT model for spam detection.

## Requirements

Python 3.6 or later is required with the following packages:

- pandas
- numpy
- sklearn
- nltk
- gensim
- transformers
- torch

## Usage

Each script can be run independently using Python. For example, to run the script that uses Word2Vec for word embeddings, use the following command:

```bash
python text_classification_word2vec.py

## Data

The scripts use a spam detection dataset that is expected to be in a CSV file named `spam.csv` in a directory named `data`. The CSV file should have the following columns:

- `v1`: The label column, which should contain the string 'ham' for non-spam messages and 'spam' for spam messages.
- `v2`: The text of the message.
