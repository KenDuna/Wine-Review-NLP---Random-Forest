import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('wine_data.csv')

X = data['description']
y = data['points']

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=69)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

def points_to_category(n):
    if n in range(80,86):
        return 0
    elif n in range(86,88):
        return 1
    elif n in range(88,90):
        return 2
    elif n in range(90,92):
        return 3
    elif n in range(92,101):
        return 4

y_train = y_train.apply(points_to_category)
y_test = y_test.apply(points_to_category)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
