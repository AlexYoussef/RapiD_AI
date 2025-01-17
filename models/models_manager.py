import random
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from torch import nn
from torch.utils.data import DataLoader

from sklearn.utils import class_weight
from xgboost import XGBClassifier
from configs.training import MLP_N_EPOCHS, MLP_PATIENCE, MLP_LR, MLP_BATCH_SIZE, TABNET_BATCH_SIZE, TAB_NET_EPOCHS
from transfer_learning.transfer_utils import MLP, mlp_train, PatientDeteriorationDataset, get_device


class BasicModelManager:
    def __init__(self, params=None, eval_metric=None, class_weights=None, from_pretrained=False, pretrained_path=None,
                 device=None):
        self.params = params
        self.eval_metric = eval_metric
        self.class_weights = class_weights  # dict of label:weight
        self.from_pretrained = from_pretrained
        self.pretrained_path = pretrained_path
        self.device = device

    def basic_model_params(self):
        pass

    def get_new_instance(self):
        pass

    def fit_classifier(self, clf, x_train, y_train, x_val, y_val, train_verbose):
        """
        How the model is trained, differs by model. This is abstract class.
        """
        pass

    def get_class_weights(self, y_train):
        return class_weight.compute_sample_weight(class_weight=self.class_weights, y=y_train)

    def set_seed(self, seed):
        self.params['seed'] = seed

    def get_seed(self):
        return self.params['seed']

    def set_eval_metric(self, eval_metric):
        self.eval_metric = eval_metric

    def set_class_weights(self, weights):
        self.class_weights = weights

    def set_params(self, params):
        self.params = params

    def predict_proba(self, clf, x_val):
        pass


class BasicXGBoostModelManager(BasicModelManager):
    def __init__(self):
        super(BasicXGBoostModelManager, self).__init__()
        self.params = self.basic_xgboost_params()

    def get_params_from_grid(self, seed=0):
        for max_depth in [2,4,6]:
            for n_estimators in [5,7,9]:
                for lr in [0.01, 0.1, 0.2, 0.3]:
                    yield {'seed': seed, 'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': lr,
                           'colsample_bynode': 1}

    def get_default_params(self):
        return [self.basic_xgboost_params()]


    @staticmethod
    def basic_xgboost_params(seed=0, max_depth=4, n_estimators=13, learning_rate=0.1, subsample=0.8):
        return {'seed': seed, 'max_depth': max_depth, 'n_estimators': n_estimators,
                'learning_rate': learning_rate,  'colsample_bynode': 1, 'subsample': subsample}

    def get_new_instance(self):
        clf = XGBClassifier(use_label_encoder=False, seed=self.params['seed'], max_depth=self.params['max_depth'],
                            n_estimators=self.params['n_estimators'], colsample_bynode=self.params['colsample_bynode'],
                            learning_rate=self.params['learning_rate'])
        if self.from_pretrained is True:
            raise NotImplementedError
        else:
            return clf

    def fit_classifier(self, clf, x_train, y_train, x_val, y_val, train_verbose):
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=self.eval_metric,
                verbose=train_verbose, sample_weight=self.get_class_weights(y_train))
        return clf

    def predict_proba(self, clf, x_val):
        return clf.predict_proba(x_val)[:, 1]


class BasicMLPModelManager(BasicModelManager):
    def __init__(self, from_pretrained=False, pretrained_path=None, basic_layer_dim=77, device=None):
        super(BasicMLPModelManager, self).__init__(from_pretrained=from_pretrained,
                                                   pretrained_path=pretrained_path,
                                                   device=device)
        self.params = self.basic_MLP_params()
        self.optimizer = None
        self.loss_fn = None
        self.basic_layer_dim = basic_layer_dim

    def get_params_from_grid(self):
        for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
            yield self.basic_MLP_params(lr=lr)

    def get_default_params(self):
        return self.basic_MLP_params()

    @staticmethod
    def basic_MLP_params(seed=1, epochs=MLP_N_EPOCHS, patience=MLP_PATIENCE, lr=MLP_LR, batch_size=MLP_BATCH_SIZE,
                         device=get_device()):
        return {'seed': seed, 'n_epochs': epochs, 'patience': patience, 'lr': lr, 'batch_size': batch_size,
                'device': device}

    def get_new_instance(self):
        # define network
        # For efficiency in getting results we want, we will set input and output dims here
        torch.manual_seed(self.params['seed'])
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        clf = MLP(input_size=self.basic_layer_dim, output_size=1)
        self.optimizer = torch.optim.Adam(clf.parameters(), lr=self.params['lr'])
        self.loss_fn = nn.BCELoss()
        return clf

    def fit_classifier(self, clf, x_train, y_train, x_val, y_val, train_verbose):
        clf = clf.to(self.device)
        train_dataset = PatientDeteriorationDataset(x_train, y_train)
        val_dataset = PatientDeteriorationDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=True)
        if self.from_pretrained is True:
            clf = torch.load(self.pretrained_path)
            self.optimizer = torch.optim.Adam(clf.parameters(), lr=self.params['lr'])
            _, _, _, _, clf, _ = mlp_train(clf, train_loader, val_loader,
                                           self.loss_fn, self.optimizer, self.params['n_epochs'],
                                           self.params['patience'], self.params['device'])
        else:
            _, _, _, _, clf, _ = mlp_train(clf, train_loader, val_loader,
                                           self.loss_fn, self.optimizer, self.params['n_epochs'],
                                           self.params['patience'], self.params['device'])
        return clf

    def predict_proba(self, clf, x_val):
        clf = clf.to(self.device)
        x_val = torch.from_numpy(x_val).float().to(self.device)
        y_prob = clf(x_val)
        return y_prob.detach().cpu().numpy()


class BasicTabNetModelManager(BasicModelManager):
    def __init__(self, from_pretrained=False, pretrained_path=None):
        super(BasicTabNetModelManager, self).__init__(from_pretrained=from_pretrained,
                                                      pretrained_path=pretrained_path)
        self.params = self.basic_tabnet_params()

    def get_params_from_grid(self, seed=0, verbose=0):
        if self.from_pretrained:
            for nd_na in [128]:
                for n_steps in [5]:
                    for gamma in [1.5, 1.75]:
                        for epochs in [4]:
                            yield {'seed': seed, 'n_d': nd_na, 'n_a': nd_na, 'n_steps': n_steps, 'gamma': gamma,
                                   'verbose': verbose, 'epochs': epochs, 'patience': 10, 'batch_size':TABNET_BATCH_SIZE}
        else:
            for nd_na in [32, 64, 128]:
                for n_steps in [5, 7, 9]:
                    for gamma in [1.5, 1.75]:
                        for epochs in [4]:
                            yield {'seed': seed, 'n_d': nd_na, 'n_a': nd_na, 'n_steps': n_steps, 'gamma': gamma,
                                   'verbose': verbose, 'epochs': epochs, 'patience': 10, 'batch_size':TABNET_BATCH_SIZE}


    def get_default_params(self):
        return [self.basic_tabnet_params()]

    @staticmethod
    def basic_tabnet_params(seed=0, n_d=128, n_a=128, n_steps=5, gamma=1.5, epochs=TAB_NET_EPOCHS, patience=10,
                            verbose=0, device=get_device()):
        return {'seed': seed, 'n_d': n_d, 'n_a': n_a, 'n_steps': n_steps, 'gamma': gamma, 'verbose': verbose,
                'epochs': epochs, 'patience': patience, 'device': device, 'batch_size':TABNET_BATCH_SIZE}

    def get_new_instance(self):
        clf = TabNetClassifier(seed=self.params['seed'], n_d=self.params['n_d'], n_a=self.params['n_a'],
                               n_steps=self.params['n_steps'], gamma=self.params['gamma'],
                               verbose=self.params['verbose'])
        clf.batch_size = self.params['batch_size']

        return clf

    def fit_classifier(self, clf, x_train, y_train, x_val, y_val, train_verbose):
        clf.verbose = train_verbose
        if self.from_pretrained is True:
            clf.load_model(self.pretrained_path)
            clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_name=['train', 'val'],
                    eval_metric=self.eval_metric, max_epochs=self.params['epochs'], weights=self.class_weights,
                    patience=self.params['patience'], drop_last=True)
        else:
            clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_name=['val'],
                    eval_metric=self.eval_metric, max_epochs=self.params['epochs'], weights=self.class_weights,
                    patience=self.params['patience'], batch_size=self.params['batch_size'], drop_last=True)
        return clf

    def predict_proba(self, clf, x_val):
        return clf.predict_proba(x_val)[:, 1]

