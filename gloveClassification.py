import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Assuming you've downloaded the pre-trained GloVe embeddings
# and stored them in a file named 'glove.6B.100d.txt'

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

raw_data = pd.read_csv("data/spam.csv", encoding="latin1")\
        .rename(columns={"v1":"label", "v2":"text"})\
        .drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    tokens = [token.lower() for token in tokens]
    
    punctuations = ['.', ',', '?', '!', ';', ':']
    tokens = [token for token in tokens if token not in punctuations]
    
    tokens = [token for token in tokens if not token.isdigit()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    tokens = [token.strip() for token in tokens if token.strip() != '']
    
    return tokens

raw_data['text_tokens'] = raw_data['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(raw_data["text_tokens"], raw_data["label"], test_size=0.2, random_state=42)

# Load GloVe embeddings
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('data/glove/glove.6B.100d.txt')

# Function to convert document to vector using GloVe embeddings
def document_vector(glove_embeddings, doc, vector_size=100):
    doc_vector = np.zeros(vector_size)
    vectors = [glove_embeddings[word] for word in doc if word in glove_embeddings]
    if vectors:
        doc_vector = np.mean(vectors, axis=0)
    return doc_vector.reshape(vector_size,)

# Using the pre-trained GloVe embeddings
X_train = X_train.apply(lambda x: document_vector(glove_embeddings, x))
X_train = np.vstack(X_train.values)

X_test = X_test.apply(lambda x: document_vector(glove_embeddings, x))
X_test = np.vstack(X_test.values)

# Create a logistic regression model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Create a logistic regression model
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of LR Model is:", accuracy)

# Make predictions on the test set
y_pred = model_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of NB Model is:", accuracy)