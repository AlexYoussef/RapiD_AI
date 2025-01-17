from sys import path as pylib
import os
pylib += [os.path.abspath('....')] # Add your path here

from pytorch_tabnet.tab_model import TabNetClassifier
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score

from configs.features_sets import ALL_FEATURES
from configs.training import TAB_NET_EPOCHS, BIN_CLASSIFICATION_WEIGHTS
import numpy as np

print(f'[{datetime.now()}] -- Reading data for pre-training')
train_data = pd.read_csv('../data/pretraining/train_sample.csv')
val_data = pd.read_csv('../data/pretraining/val_sample.csv')
test_data = pd.read_csv('../data/pretraining/test_sample.csv')


used_features = ALL_FEATURES

X_train, y_train = train_data[used_features].to_numpy(), train_data['label'].to_numpy()
X_val, y_val = val_data[used_features].to_numpy(), val_data['label'].to_numpy()
X_test, y_test = test_data[used_features].to_numpy(), test_data['label'].to_numpy()

ind=np.where(y_train>0)[0]
y_train[ind]=1
ind=np.where(y_val>0)[0]
y_val[ind]=1
ind=np.where(y_test>0)[0]
y_test[ind]=1

params = {'n_d': 128, 'n_a': 128, 'n_steps': 5, 'gamma': 1.5, 'verbose': True, 'epochs': TAB_NET_EPOCHS, 'patience': 25,
          'fromPath': None, 'saved_model': './tab_net_model_clf.pkl'}
eval_metrics = ['logloss']
saving_path_name = "./pretrained_models/TabNet"


clf = TabNetClassifier(n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'], gamma=params['gamma'],
                       verbose=params['verbose'])

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_name=['val'],  eval_metric=eval_metrics, max_epochs=6,
        weights=BIN_CLASSIFICATION_WEIGHTS, patience=params['patience'], batch_size=1280, virtual_batch_size=128)

print(f'[{datetime.now()}] -- Saving model...')
clf.save_model(saving_path_name)

print(f'[{datetime.now()}] -- Loading trained model...')
clf.load_model(f'{saving_path_name}.zip')

print(f"[{datetime.now()}] -- Predicting...")
A=clf.predict_proba(X_test)[:,1]
test_auc=roc_auc_score(y_test,A)
print(test_auc)


