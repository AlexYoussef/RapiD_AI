import copy
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import Dataset

# from configs import MLP_RESULTS_FILE, MLP_DEBUG_FILE, MLP_BEST_MODEL
from utils import evaluation_metrics
from utils.evaluation_metrics import get_best_prec_recall_threshold

VERBOSE = True


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def init_output_file(filepath):
    handler = open(filepath, 'w')
    handler.close()


def custom_print(msg, paused=False):
    if paused:
        return
    msg = str(msg)
    print(f'[{datetime.now()}] -- {msg}')


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size * 4, input_size * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size * 3, output_size),
        )
        self.sm = nn.Sigmoid()

    def forward(self, x):
        return self.sm(self.layers(x))


# Here dataset is numpy array
class PatientDeteriorationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(dtype=np.float32)
        self.labels = labels.astype(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# here dataset is pandas DF
class PatientDeteriorationDataset_PD(Dataset):
    def __init__(self, data, used_features, target_label):
        self.features = data[used_features].to_numpy(dtype=np.float32)
        self.labels = data[target_label].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def mlp_train(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, patience, device, best_auc=0):
    # Run the training loop
    model = model.to(device)
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    step_size = 10
    last_improvement = 0
    best = best_auc
    best_model = copy.deepcopy(model)
    for epoch in range(n_epochs):
        # Print epoch
        custom_print(f'INFO: Starting epoch {epoch + 1}')
        # Set current loss value
        train_loss = 0.0
        # Iterate over the DataLoader for training data
        model = model.train()
        for i, data in enumerate(train_loader):
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            ind = torch.where(targets > 0)
            targets[ind] = 1
            # print(targets.shape)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)[:, 0]
            # print(outputs.shape)
            # Compute loss
            loss = loss_fn(outputs, targets.to(torch.float32))
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # Print statistics
            train_loss += loss.item()
            if i % step_size == step_size - 1:
                cur_step = epoch * train_loader.__len__() + i + 1
                train_steps.append(cur_step)
                train_losses.append(train_loss / step_size)
                train_loss = 0.0

        model.eval()
        val_loss = 0.0
        val_steps_count = 0
        preds = []
        labels = []
        for i, data in enumerate(val_loader):
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            ind = torch.where(targets > 0)
            targets[ind] = 1
            # Perform forward pass
            outputs = model(inputs)[:, 0]
            # Compute loss
            loss = loss_fn(outputs, targets.to(torch.float32))
            # Print statistics
            val_loss += loss.item()
            val_steps_count += 1
            preds.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())

        cur_step = (epoch + 1) * val_loader.__len__()
        val_loss = val_loss / val_steps_count
        val_steps.append(cur_step)
        val_losses.append(val_loss)

        P = np.concatenate(preds)
        L = np.concatenate(labels)
        # print(np.unique(L))
        # print(P.shape)
        val_auc = roc_auc_score(L, P)
        custom_print(f'INFO: Val Loss: {val_loss:.6f} VAL AUC: {val_auc:.6f}')

        if val_auc > best:
            best = val_auc
            last_improvement = 0
            best_model = copy.deepcopy(model)
        else:
            last_improvement += 1
            if last_improvement == patience:
                break

    # Process is complete.
    if last_improvement == patience:
        custom_print(f'INFO: Early stopping occurred after {epoch + 1} epochs.')
        return train_steps, train_losses, val_steps, val_losses, best_model, best_auc
    else:
        custom_print(f'INFO: Training process has finished.')
        return train_steps, train_losses, val_steps, val_losses, best_model, best_auc


def mlp_predict(model, data_loader, device):
    model = model.to(device)
    model.eval()
    pred_prob = []
    true_labels = []
    for x, y in data_loader:
        x = x.to(device)
        pred_part = model(x)
        pred_prob.append(pred_part.cpu().detach().numpy())
        ind = torch.where(y > 0)
        y[ind] = 1
        true_labels.append(y.detach().numpy())

    return np.concatenate(pred_prob), np.concatenate(true_labels)


def validate(train_true, train_probs, val_true, val_probs, test_true, test_probs):
    chosen_threshold, metric_prec = get_best_prec_recall_threshold(train_true, train_probs[:, 1])
    train_preds = (train_probs[:, 1] > chosen_threshold).astype('float')
    val_preds = (val_probs[:, 1] > chosen_threshold).astype('float')
    test_preds = (test_probs[:, 1] > chosen_threshold).astype('float')

    train_metrics = evaluation_metrics.compute_basic_metrics(train_true, train_probs, train_preds)
    val_metrics = evaluation_metrics.compute_basic_metrics(val_true, val_probs, val_preds)
    test_metrics = evaluation_metrics.compute_basic_metrics(test_true, test_probs, test_preds)

    return chosen_threshold, train_preds, val_preds, train_metrics, test_preds, val_metrics, test_metrics


def print_results(chosen_threshold=None, train_metrics=None, val_metrics=None, test_metrics=None, train_true=None,
                  train_preds=None, val_true=None, val_preds=None, test_true=None, test_preds=None):
    custom_print(f'Chosen threshold = {chosen_threshold}')
    custom_print('Train metrics')
    custom_print(train_metrics)
    custom_print('Validation metrics')
    custom_print(val_metrics)
    custom_print('Test metrics')
    custom_print(test_metrics)
    custom_print('Train confusion matrix')
    tn, fp, fn, tp = confusion_matrix(train_true, train_preds, labels=[0, 1]).ravel()
    custom_print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    custom_print('Val confusion matrix')
    tn, fp, fn, tp = confusion_matrix(val_true, val_preds, labels=[0, 1]).ravel()
    custom_print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    custom_print('Test confusion matrix')
    tn, fp, fn, tp = confusion_matrix(test_true, test_preds, labels=[0, 1]).ravel()
    custom_print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
