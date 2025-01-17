import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from sys import path as pylib
import os
pylib += [os.path.abspath('....')] # Add your path here

from configs.training import MLP_N_EPOCHS, MLP_PATIENCE
from transfer_learning.transfer_utils import MLP, PatientDeteriorationDataset_PD, mlp_train, mlp_predict,  custom_print, \
    get_device

from configs.features_sets import ALL_FEATURES

# read file here
custom_print(f'INFO: Loading data')
train_data = pd.read_csv('../data/pretraining/train_sample.csv')
val_data = pd.read_csv('../data/pretraining/val_sample.csv')
test_data = pd.read_csv('../data/pretraining/test_sample.csv')

# define features
used_features= ALL_FEATURES

train_dataset = PatientDeteriorationDataset_PD(train_data, used_features, 'label')
val_dataset = PatientDeteriorationDataset_PD(val_data, used_features, 'label')
test_dataset = PatientDeteriorationDataset_PD(test_data, used_features, 'label')
MLP_BATCH_SIZE=64
# del train_data, val_data, test_data
train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=MLP_BATCH_SIZE)

# define model
custom_print(f'INFO: Defining model')
mlp_model = MLP(input_size=len(used_features), output_size=1)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
custom_print(f'INFO: Training model')
train_steps, train_losses, val_steps, val_losses, mlp_model, best_auc = mlp_train(mlp_model, train_loader, val_loader,
                                                                                  loss_fn, optimizer, MLP_N_EPOCHS,
                                                                                  MLP_PATIENCE, get_device())

torch.save(mlp_model, './pretrained_models/EXP_1_MLP.pkl')
mlp_model = torch.load('./pretrained_models/EXP_1_MLP.pkl')
P,L=mlp_predict(mlp_model, test_loader, get_device())
test_auc=roc_auc_score(L,P[:,0])
print(test_auc)
