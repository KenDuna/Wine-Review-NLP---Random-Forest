import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical

# Load data
X_train = np.array(pd.read_csv('X_train_encoded.csv'))
X_test = np.array(pd.read_csv('X_test_encoded.csv'))
y_train = to_categorical(pd.read_csv('y_train.csv'))
y_test = to_categorical(pd.read_csv('y_test.csv'))



# Create the model
num_trees = 1000
model = RandomForestClassifier(n_estimators= num_trees,
                               bootstrap = True,
                               max_features = 'sqrt')

# Fit on training data
model.fit(X_train, y_train)

print('Training Accuracy: {}'.format(model.score(X_train, y_train)))

print('Test Accuracy: {}'.format(model.score(X_test, y_test)))

# Make predictions
#rf_predictions = model.predict(X_test)
#rf_probs = model.predict_proba(X_test)

# Calculate roc_auc
#roc_value = roc_auc_score(y_test, rf_probs, multi_class='ovr')
