import winsound
import numpy as np
import pandas as pd
from parse import parse_text
from matplotlib import pyplot as plt

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
df = pd.read_csv('word_freq.csv')

num_features = 100
top_words = list(df['Word'][:num_features])

def encode_review(review):
    parsed_review = parse_text(review)
    
    feature_dict = {}
    
    for i in range(num_features):
        if top_words[i] in parsed_review:
            feature_dict[top_words[i]] = 1
        else:
            feature_dict[top_words[i]] = 0
    return feature_dict


X_train_encoded = pd.DataFrame(columns=top_words)
for i in range(len(X_train.index)):
    X_train_encoded = X_train_encoded.append(
        encode_review(X_train['description'][i]),
        ignore_index=True)    
X_train_encoded.to_csv('X_train_encoded.csv', index=False)

print('Training Data Encoded')

X_test_encoded = pd.DataFrame(columns=top_words)

for i in range(len(X_test.index)):
    X_test_encoded = X_test_encoded.append(
        encode_review(X_test['description'][i]),
        ignore_index=True)

X_test_encoded.to_csv('X_test_encoded.csv', index=False)

winsound.MessageBeep()
