# dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# hyperparameters = {'randomforestregressor__criterion': ['mse', 'mae'],
#                   'randomforestregressor__max_depth': [None, 5, 3, 1],
#                   'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']}
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# loading data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# features - label split
y = data.quality
X = data.drop(['quality'], axis=1)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# preprocessing
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=300))

# declare hyperparameters
hyperparameters = {'randomforestregressor__criterion': ['mse', 'mae'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1],
                  'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']}

# cross-validation - 10 fold
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# fit and tune model
clf.fit(X_train, y_train)

# print best params
print('Best params:', clf.best_params_)

# refit model
clf.refit

# prediction
y_pred = clf.predict(X_test)

# evaluation
print('mse:', mean_squared_error(y_test, y_pred))
print('r2 score:', r2_score(y_test, y_pred))
print('mse must be low, r2 score must be high')