#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 23:51:24 2024

@author: eedisgpu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:41:30 2024

@author: Bouchiha
"""
import pandas as pd
import numpy as np
import nltk
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, TFBertModel

# Check if GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
data = pd.read_csv('train_40k_Adapted.csv')

# Preprocess texts
data['processed_text'] = data['text'].apply(preprocess_text)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize texts and convert them to BERT input format
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(), 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='tf'
    )

encoded_texts = encode_texts(data['processed_text'], tokenizer)

# Split dataset
X_train, X_test, y_train_cat, y_test_cat, y_train_super_cat, y_test_super_cat = train_test_split(
    encoded_texts['input_ids'].numpy(), data['category'], data['super_category'], test_size=0.3, random_state=42)

# Create label mappings
category_labels = list(data['category'].unique())
super_category_labels = list(data['super_category'].unique())

# Map labels to indices
y_train_cat_idx = y_train_cat.apply(lambda x: category_labels.index(x))
y_test_cat_idx = y_test_cat.apply(lambda x: category_labels.index(x) if x in category_labels else -1)

y_train_super_cat_idx = y_train_super_cat.apply(lambda x: super_category_labels.index(x))
y_test_super_cat_idx = y_test_super_cat.apply(lambda x: super_category_labels.index(x) if x in super_category_labels else -1)

# Remove rows with unseen labels in the test set
valid_test_indices = (y_test_cat_idx != -1) & (y_test_super_cat_idx != -1)
X_test = X_test[valid_test_indices]
y_test_cat_idx = y_test_cat_idx[valid_test_indices]
y_test_super_cat_idx = y_test_super_cat_idx[valid_test_indices]

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train_cat_idx, num_classes=len(category_labels))
y_test_cat = tf.keras.utils.to_categorical(y_test_cat_idx, num_classes=len(category_labels))

y_train_super_cat = tf.keras.utils.to_categorical(y_train_super_cat_idx, num_classes=len(super_category_labels))
y_test_super_cat = tf.keras.utils.to_categorical(y_test_super_cat_idx, num_classes=len(super_category_labels))

# Fine-tune BERT model
def create_fine_tuned_bert_model(num_categories, num_super_categories):
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state

    # Add a BiLSTM layer on top of the BERT output
    lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(bert_output)
    lstm_out = tf.keras.layers.GlobalMaxPooling1D()(lstm_out)

    # Output layers
    category_output = tf.keras.layers.Dense(num_categories, activation='softmax', name='category_output')(lstm_out)
    super_category_output = tf.keras.layers.Dense(num_super_categories, activation='softmax', name='super_category_output')(lstm_out)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[category_output, super_category_output])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

num_categories = len(category_labels)
num_super_categories = len(super_category_labels)
model = create_fine_tuned_bert_model(num_categories, num_super_categories)

# Create attention masks
attention_masks = (X_train != 0).astype(int)

# Train model
history = model.fit(
    [X_train, attention_masks], 
    {'category_output': y_train_cat, 'super_category_output': y_train_super_cat},
    validation_split=0.2,
    epochs=3,
    batch_size=16
)

# Evaluate model
attention_masks_test = (X_test != 0).astype(int)
y_pred_cat, y_pred_super_cat = model.predict([X_test, attention_masks_test])
y_pred_cat = np.argmax(y_pred_cat, axis=1)
y_pred_super_cat = np.argmax(y_pred_super_cat, axis=1)

y_test_cat = np.argmax(y_test_cat, axis=1)
y_test_super_cat = np.argmax(y_test_super_cat, axis=1)

# Evaluation metrics
def hierarchical_f1_score(y_true_cat, y_pred_cat, y_true_super_cat, y_pred_super_cat):
    correct_predictions = (y_true_cat == y_pred_cat) & (y_true_super_cat == y_pred_super_cat)
    hierarchical_f1 = f1_score(correct_predictions, [True] * len(correct_predictions), average='weighted')
    return hierarchical_f1

category_accuracy = accuracy_score(y_test_cat, y_pred_cat)
category_f1 = f1_score(y_test_cat, y_pred_cat, average='weighted')
super_category_accuracy = accuracy_score(y_test_super_cat, y_pred_super_cat)
super_category_f1 = f1_score(y_test_super_cat, y_pred_super_cat, average='weighted')
hierarchical_f1 = hierarchical_f1_score(y_test_cat, y_pred_cat, y_test_super_cat, y_pred_super_cat)

print("\n\n*******      BERT-BiLSTM - gpu    ******")
print(f'Category Accuracy: {category_accuracy}')
print(f'Category F1 Score: {category_f1}')
print(f'Super-Category Accuracy: {super_category_accuracy}')
print(f'Super-Category F1 Score: {super_category_f1}')
print(f'Hierarchical F1 Score: {hierarchical_f1}')
