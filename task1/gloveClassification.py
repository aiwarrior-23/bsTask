import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Constants
PUNCTUATIONS = ['.', ',', '?', '!', ';', ':']
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesses the text by tokenizing, removing punctuations, digits, stopwords, and lemmatizing."""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in PUNCTUATIONS and not token.isdigit() and token not in STOP_WORDS]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token.strip() for token in tokens if token.strip() != '']
    return tokens

def load_glove_embeddings(path):
    """Loads GloVe embeddings from the given path."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def document_vector(embeddings, doc, vector_size=100):
    """Returns the mean vector for the given document using the provided embeddings."""
    doc_vector = np.zeros(vector_size)
    vectors = [embeddings[word] for word in doc if word in embeddings]
    if vectors:
        doc_vector = np.mean(vectors, axis=0)
    return doc_vector.reshape(vector_size,)

def load_and_preprocess_data():
    """Loads and preprocesses the data."""
    raw_data = pd.read_csv("data/spam.csv", encoding="latin1")\
            .rename(columns={"v1":"label", "v2":"text"})\
            .drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    raw_data['text_tokens'] = raw_data['text'].apply(preprocess_text)
    return raw_data

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates the logistic regression and naive bayes models."""
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of LR Model is:", accuracy)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred = model_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of NB Model is:", accuracy)

def main():
    """Main function to run the script."""
    raw_data = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(raw_data["text_tokens"], raw_data["label"], test_size=0.2, random_state=42)
    glove_embeddings = load_glove_embeddings('data/glove/glove.6B.100d.txt')
    X_train = X_train.apply(lambda x: document_vector(glove_embeddings, x))
    X_train = np.vstack(X_train.values)
    X_test = X_test.apply(lambda x: document_vector(glove_embeddings, x))
    X_test = np.vstack(X_test.values)
    train_and_evaluate_models(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
