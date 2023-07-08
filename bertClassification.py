import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a given text
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Get BERT embeddings for training data
X_train_embeddings = np.vstack([get_bert_embeddings(text, tokenizer, model) for text in X_train])

# Get BERT embeddings for test data
X_test_embeddings = np.vstack([get_bert_embeddings(text, tokenizer, model) for text in X_test])

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
