from sys import path as pylib
import os
pylib += [os.path.abspath('....')] # Add your path here

from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from configs.data import LABEL_COL
from configs.training import MLP_BATCH_SIZE, MLP_LR, MLP_PATIENCE

from transfer_learning.transfer_utils import MLP, PatientDeteriorationDataset_PD, mlp_train, mlp_predict, custom_print,\
    get_device
from utils.data_processor import LabelProcessor, FeatureSelector

from configs.features_sets import ALL_FEATURES
from tqdm import tqdm
import gc

# define features
used_features = ALL_FEATURES

# define label processor
lp = LabelProcessor({0: 0, 4: 1, 5: 1})
# Define feature selector
feature_selector = FeatureSelector(features=used_features)

# Load Validation data
custom_print(f'INFO: Loading Validationg data')
val_data = pd.read_csv('../data/Scenario_C/ROME_ready_part_4_val.csv').dropna()
val_data = lp.process(val_data)
val_dataset = PatientDeteriorationDataset_PD(val_data, used_features, LABEL_COL)
val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
del val_data
gc.collect()

# Train file is split for 3 parts (to fit in small memory settings)
train_1_path = '../data/Scenario_C/ROME_ready_part_1.csv'
train_2_path = '../data/Scenario_C/ROME_ready_part_2.csv'
train_3_path = '../data/Scenario_C/ROME_ready_part_3.csv'
train_paths = [train_1_path, train_2_path, train_3_path]

# define model
custom_print(f'INFO: Defining model')
mlp_model = MLP(input_size=len(used_features), output_size=1)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=MLP_LR)
loss_fn = nn.BCELoss()

custom_print(f'INFO: Training model')

train_steps_all = []
train_losses_all = []
val_steps_all = []
val_losses_all = []

n_epochs = 15
mini_epoch_size = 5

prog_bar = tqdm(range(n_epochs // mini_epoch_size))
best_auc = 0
for ep in prog_bar:
    for tr_idx, train_path in enumerate(train_paths):
        prog_bar.set_description(f"[{datetime.now()}] -- Epoch {ep} -- Loading train file #{tr_idx + 1}")
        prog_bar.refresh()  # to show immediately the update
        train_data = pd.read_csv(train_path).dropna()
        train_data = lp.process(train_data)
        train_dataset = PatientDeteriorationDataset_PD(train_data, used_features, LABEL_COL)
        del train_data
        gc.collect()

        prog_bar.set_description(f"Epoch {ep} -- training #{tr_idx + 1}")
        prog_bar.refresh()  # to show immediately the update
        train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
        train_steps, train_losses, val_steps, val_losses, mlp_model, best_auc = mlp_train(mlp_model, train_loader,
                                                                                          val_loader,
                                                                                          loss_fn, optimizer,
                                                                                          mini_epoch_size,
                                                                                          MLP_PATIENCE, get_device(),
                                                                                          best_auc)
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=MLP_LR)
        train_steps_all = train_steps_all + train_steps
        train_losses_all = train_losses_all + train_losses
        val_steps_all = val_steps_all + val_steps
        val_losses_all = val_losses_all + val_losses

torch.save(mlp_model, './pretrained_models/MLP_Scenario_C.pkl')
torch.save([train_steps_all, train_losses_all, val_steps_all, val_losses_all], './logging/losses_MLP_Scenario_C.pkl')

# Testing on downstream Task
mlp_model = torch.load('./pretrained_models/MLP_Scenario_C.pkl')
test_data = pd.read_csv('../data/TEST_EXP1.csv')
test_dataset = PatientDeteriorationDataset_PD(test_data, used_features, LABEL_COL)
del test_data


gc.collect()
test_loader = DataLoader(test_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)

P, L = mlp_predict(mlp_model, test_loader, get_device())
test_auc = roc_auc_score(L, P[:, 0])
print(test_auc)
