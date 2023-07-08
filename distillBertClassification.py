import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

raw_data = pd.read_csv("data/spam.csv", encoding="latin1")\
        .rename(columns={"v1":"labels", "v2":"text"})\
        .drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Map string labels to integers
label_map = {"ham": 0, "spam": 1}
raw_data["labels"] = raw_data["labels"].map(label_map)

train_df, test_df = train_test_split(raw_data, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# Preprocess the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="test_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define the compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
