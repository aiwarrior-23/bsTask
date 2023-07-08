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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Print train and test data
print("Training data:")
print(len(X_train), len(y_train))
print()
print("Testing data:")
print(len(X_test), len(y_test))

model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)

# Getting the vector representation of each token and averaging them
def document_vector(model, doc):
    # Start with a vector of zeros
    doc_vector = np.zeros(model.vector_size)
    
    # List of vectors for words in doc that are in the word2vec vocabulary
    vectors = [model.wv[word] for word in doc if word in model.wv]
    
    # If there are word vectors, calculate their mean
    if vectors:
        doc_vector = np.mean(vectors, axis=0)
    
    # Ensure that the result is of shape (model.vector_size,)
    return doc_vector.reshape(model.vector_size,)

X_train = X_train.apply(lambda x: document_vector(model, x))
X_train = np.vstack(X_train.values)

# Getting the vector representation of each token and averaging them
def document_vector(model, doc):
    # Start with a vector of zeros
    doc_vector = np.zeros(model.vector_size)
    
    # List of vectors for words in doc that are in the word2vec vocabulary
    vectors = [model.wv[word] for word in doc if word in model.wv]
    
    # If there are word vectors, calculate their mean
    if vectors:
        doc_vector = np.mean(vectors, axis=0)
    
    # Ensure that the result is of shape (model.vector_size,)
    return doc_vector.reshape(model.vector_size,)

X_test = X_test.apply(lambda x: document_vector(model, x))
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

