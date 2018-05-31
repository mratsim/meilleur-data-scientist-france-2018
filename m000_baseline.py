# coding: utf-8
# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.xgb_processing import xgb_validate, xgb_cross_val, xgb_output
from m110_feat_eng import select_feat
from src.star_command import feat_selection
from m110_feat_eng import pipe_transforms

import logging
import time
import os
from timeit import default_timer as timer
from src.instrumentation import setup_logs

# Log
str_timerun = time.strftime("%Y-%m-%d_%H%M")
tmp_logfile = os.path.join('./outputs/', f'{str_timerun}--run-in-progress.log')
logger = setup_logs(tmp_logfile)

# Globals
cache_file = './cache.db'

# Import data
X = pd.read_csv("./data/X_train.csv", error_bad_lines=False)
print('Input training data has shape: ', X.shape)

y = pd.read_csv("./data/y_train.csv", index_col=0, error_bad_lines=False)

print("############ Preprocessing test data ######################")
X_test = pd.read_csv("./data/X_test.csv", error_bad_lines=False)
id_test = X_test['id']

le = LabelEncoder()
y = le.fit_transform(y)

##############################
# Setup basic XGBoost and validation
# Validation is used to get an unique name only
# Model performance will be measured by proper Cross-Validation

xgb_params                     = {}
xgb_params['num_class']        = 3
xgb_params['objective']        = 'multi:softprob'
xgb_params['eta']              = 0.1
xgb_params['max_depth']        = 4
xgb_params['silent']           = 1
xgb_params['eval_metric']      = "mlogloss"
xgb_params['min_child_weight'] = 1
xgb_params['subsample']        = 0.7
xgb_params['colsample_bytree'] = 0.7
xgb_params['seed']             = 1337
xgb_params['tree_method']      = 'gpu_hist'

xgb_params = list(xgb_params.items())

###############################
# Create folds

cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1337)
folds = list(cv.split(X,y))

# Pipeline processing
logger.info("   ===> Preprocessing")
X, X_test, _, _, _ = pipe_transforms(X, X_test, y, folds, cache_file)

logger.info(f'After preprocessing data shape is: {X.shape}')

# Quick validation to get a unique name
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and validate
print("############ Validation ######################")
x_trn, x_val = feat_selection(select_feat, x_trn, x_val, y_trn)
val_score = xgb_validate(x_trn, x_val, y_trn, y_val, xgb_params, seed_val = 0)

# Selection
X_train, X_test = feat_selection(select_feat, X, X_test, y)

# print("############ Cross - Validation ######################")
# n_stop = xgb_cross_val(xgb_params, X_train, y, folds)
# n_stop = np.int(n_stop * 1.1) # Full dataset is 25% bigger, so we want a bit of leeway on stopping round to avoid overfitting.

n_stop = 300

print("############ Training ######################")
xgtrain = xgb.DMatrix(X_train, y)
classifier = xgb.train(xgb_params, xgtrain, n_stop)

print("############ Prediction ######################")
xgb_output(X_test, id_test, classifier, n_stop, val_score, le)
