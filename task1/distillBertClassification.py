import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Constants
MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"ham": 0, "spam": 1}

def load_and_preprocess_data():
    """Loads and preprocesses the data."""
    raw_data = pd.read_csv("data/spam.csv", encoding="latin1")\
            .rename(columns={"v1":"labels", "v2":"text"})\
            .drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    raw_data["labels"] = raw_data["labels"].map(LABEL_MAP)
    train_df, test_df = train_test_split(raw_data, test_size=0.2)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, test_dataset

def preprocess_function(examples):
    """Preprocesses the examples using the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

def compute_metrics(eval_pred):
    """Computes the accuracy of the predictions."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

def train_model(train_dataset, test_dataset):
    """Trains the model."""
    training_args = TrainingArguments(
        output_dir="test_model",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def main():
    """Main function to run the script."""
    train_dataset, test_dataset = load_and_preprocess_data()
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    train_model(train_dataset, test_dataset)

if __name__ == "__main__":
    main()