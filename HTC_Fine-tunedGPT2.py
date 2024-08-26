#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:16:21 2024

@author: eedisgpu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:41:56 2024

@author: Bouchiha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset

# Load dataset
data = pd.read_csv('train_40k_Adapted.csv')

# Preprocess dataset
data['text_category'] = data['text'] + " [CATEGORY] " + data['category']
data['text_category_supercategory'] = data['text_category'] + " [SUPERCATEGORY] " + data['super_category']

# Tokenizer and Model initialization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Encode labels
category_labels = {label: i for i, label in enumerate(data['category'].unique())}
super_category_labels = {label: i for i, label in enumerate(data['super_category'].unique())}

# Map labels
labels_category = data['category'].map(category_labels).tolist()
labels_super_category = data['super_category'].map(super_category_labels).tolist()

# Split dataset
train_texts, test_texts, train_labels_category, test_labels_category, train_labels_super_category, test_labels_super_category = train_test_split(
    data['text'].tolist(), labels_category, labels_super_category, test_size=0.3, random_state=42)

# Encode train and test sets with padding
train_encodings_category = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings_category = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

train_encodings_super_category = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings_super_category = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Create datasets
train_dataset_category = CustomDataset(train_encodings_category, train_labels_category)
test_dataset_category = CustomDataset(test_encodings_category, test_labels_category)

train_dataset_super_category = CustomDataset(train_encodings_super_category, train_labels_super_category)
test_dataset_super_category = CustomDataset(test_encodings_super_category, test_labels_super_category)

# Initialize model
model_category = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(category_labels))
model_category.resize_token_embeddings(len(tokenizer))
model_category.to('cuda')  # Move model to GPU

model_super_category = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(super_category_labels))
model_super_category.resize_token_embeddings(len(tokenizer))
model_super_category.to('cuda')  # Move model to GPU

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args_category = TrainingArguments(
    output_dir='./results_category',
    num_train_epochs=5,  # Increase the number of epochs
    per_device_train_batch_size=1,  # Set batch size to 1 to avoid padding issues
    per_device_eval_batch_size=1,  # Set batch size to 1 to avoid padding issues
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_category',
    logging_steps=10,
    evaluation_strategy="epoch",
    #device='cuda'  # Specify GPU usage
)

training_args_super_category = TrainingArguments(
    output_dir='./results_super_category',
    num_train_epochs=3,  # Increase the number of epochs
    per_device_train_batch_size=1,  # Set batch size to 1 to avoid padding issues
    per_device_eval_batch_size=1,  # Set batch size to 1 to avoid padding issues
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_super_category',
    logging_steps=10,
    evaluation_strategy="epoch",
    #device='cuda'  # Specify GPU usage
)

# Initialize Trainer
trainer_category = Trainer(
    model=model_category,
    args=training_args_category,
    train_dataset=train_dataset_category,
    eval_dataset=test_dataset_category,
    data_collator=data_collator
)

trainer_super_category = Trainer(
    model=model_super_category,
    args=training_args_super_category,
    train_dataset=train_dataset_super_category,
    eval_dataset=test_dataset_super_category,
    data_collator=data_collator
)

# Train models
trainer_category.train()
trainer_super_category.train()

# Evaluate models
category_predictions = trainer_category.predict(test_dataset_category)
super_category_predictions = trainer_super_category.predict(test_dataset_super_category)

# Convert predictions to labels
y_pred_cat = np.argmax(category_predictions.predictions, axis=1)
y_true_cat = np.array(test_labels_category)

y_pred_super_cat = np.argmax(super_category_predictions.predictions, axis=1)
y_true_super_cat = np.array(test_labels_super_category)

# Evaluation metrics
def hierarchical_f1_score(y_true_cat, y_pred_cat, y_true_super_cat, y_pred_super_cat):
    correct_predictions = (y_true_cat == y_pred_cat) & (y_true_super_cat == y_pred_super_cat)
    hierarchical_f1 = f1_score(correct_predictions, [True] * len(correct_predictions), average='weighted')
    return hierarchical_f1

category_accuracy = accuracy_score(y_true_cat, y_pred_cat)
category_f1 = f1_score(y_true_cat, y_pred_cat, average='weighted')
super_category_accuracy = accuracy_score(y_true_super_cat, y_pred_super_cat)
super_category_f1 = f1_score(y_true_super_cat, y_pred_super_cat, average='weighted')
hierarchical_f1 = hierarchical_f1_score(y_true_cat, y_pred_cat, y_true_super_cat, y_pred_super_cat)

print("\n\n*******      Fine-tuned GPT2   -GPU  ******")
print(f'Category Accuracy: {category_accuracy}')
print(f'Category F1 Score: {category_f1}')
print(f'Super-Category Accuracy: {super_category_accuracy}')
print(f'Super-Category F1 Score: {super_category_f1}')
print(f'Hierarchical F1 Score: {hierarchical_f1}')
