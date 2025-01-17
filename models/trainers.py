import pandas as pd
import numpy as np
from configs.data import TIMESTAMP_FORMAT, TIMESTAMP_COL, LABEL_COL
from models.wrapper import BasicModelsWrapper
from utils.data_processor import FeatureSelector
from utils.temporal_splitter import TemporalSplitter


def train_with_grid_search(model_manager, train_data_path, test_data_path, features, interval, class_weights,
                           eval_metric, threshold='auto'):
    # Read data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Define temporal splitter
    temporal_splitter = TemporalSplitter(data=train_data, timestamp_loc=TIMESTAMP_COL,
                                         timestamp_format=TIMESTAMP_FORMAT,
                                         interval=interval)
    # Define feature selector
    feature_selector = FeatureSelector(features=features)
    # prepare testing data
    test_x = feature_selector.process(test_data).to_numpy()
    test_y = np.array(test_data.loc[:, LABEL_COL])
    GS_results = {'splits': [], 'params': [], 'AUC': []}
    weekly_best_params = []
    for split_id, split in enumerate(temporal_splitter.get_splits()):
        print(f"Processing split #{split_id + 1:02d}")
        # Prepare split data for training
        train_x = feature_selector.process(split).to_numpy()
        train_y = np.array(split.loc[:, LABEL_COL])

        # Prepare model manager
        model_manager.set_class_weights(class_weights)
        model_manager.set_eval_metric(eval_metric)

        # Define model
        cur_best_param = None
        best_AUC = 0.0
        for params in model_manager.get_params_from_grid():
            model_manager.set_params(params)
            model = BasicModelsWrapper(model_manager)
            # run exp
            _, val_metrics, test_metrics = model.train_with_no_confidence_estimation(train_x, train_y, test_x, test_y,
                                                                                     threshold=threshold)
            GS_results['splits'].append(split_id)
            GS_results['params'].append(str(list(params.values())))
            GS_results['AUC'].append(test_metrics['AUC'])

            if val_metrics['AUC'] > best_AUC:
                cur_best_param = params
                best_AUC = val_metrics['AUC']
        weekly_best_params.append((cur_best_param, best_AUC))

    return pd.DataFrame.from_dict(GS_results), weekly_best_params


def train_with_CI(model_manager, train_data_path, test_data_path, features, interval, params, class_weights,
                  eval_metric, n=5, confidence=0.95, threshold='auto'):
    # Read data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Define temporal splitter
    temporal_splitter = TemporalSplitter(data=train_data, timestamp_loc=TIMESTAMP_COL,
                                         timestamp_format=TIMESTAMP_FORMAT,
                                         interval=interval)
    # Define feature selector
    feature_selector = FeatureSelector(features=features)
    # prepare testing data
    test_x = feature_selector.process(test_data).to_numpy()
    test_y = np.array(test_data.loc[:, LABEL_COL])

    # Define output
    CI_results = {'splits': [], 'params': [],
                  'AUC_mean': [], 'AUC_min': [], 'AUC_max': [],
                  'Acc_mean': [], 'Acc_min': [], 'Acc_max': [],
                  'Sen_mean': [], 'Sen_min': [], 'Sen_max': [],
                  'Sps_mean': [], 'Sps_min': [], 'Sps_max': [],
                  'Prs_mean': [], 'Prs_min': [], 'Prs_max': []}

    for split_id, split in enumerate(temporal_splitter.get_splits()):
        print(f"Processing split #{split_id + 1:02d}")
        # Prepare split data for training
        train_x = feature_selector.process(split).to_numpy()
        train_y = np.array(split.loc[:, LABEL_COL])

        # Prepare model manager
        model_manager.set_class_weights(class_weights)
        model_manager.set_eval_metric(eval_metric)
        if type(params) == list:
            split_params = params[split_id]
        elif type(params) == dict:
            split_params = params
        else:
            split_params = model_manager.get_default_params()

        model_manager.set_params(split_params)
        # Define model
        model = BasicModelsWrapper(model_manager)
        # run exp
        _, _, test_confidence = model.train_with_confidence_estimation(train_x, train_y, test_x, test_y, n=n,
                                                                       confidence=confidence, threshold=threshold)

        CI_results['splits'].append(split_id + 1)
        CI_results['params'].append(str(list(split_params.values())))
        CI_results['Acc_mean'].append(test_confidence['Accuracy']['Mean'])
        CI_results['Acc_min'].append(test_confidence['Accuracy']['Left bound'])
        CI_results['Acc_max'].append(test_confidence['Accuracy']['Right bound'])

        CI_results['Sen_mean'].append(test_confidence['sensitivity']['Mean'])
        CI_results['Sen_min'].append(test_confidence['sensitivity']['Left bound'])
        CI_results['Sen_max'].append(test_confidence['sensitivity']['Right bound'])

        CI_results['Sps_mean'].append(test_confidence['specificity']['Mean'])
        CI_results['Sps_min'].append(test_confidence['specificity']['Left bound'])
        CI_results['Sps_max'].append(test_confidence['specificity']['Right bound'])

        CI_results['Prs_mean'].append(test_confidence['precision']['Mean'])
        CI_results['Prs_min'].append(test_confidence['precision']['Left bound'])
        CI_results['Prs_max'].append(test_confidence['precision']['Right bound'])

        CI_results['AUC_mean'].append(test_confidence['AUC']['Mean'])
        CI_results['AUC_min'].append(test_confidence['AUC']['Left bound'])
        CI_results['AUC_max'].append(test_confidence['AUC']['Right bound'])

    return pd.DataFrame.from_dict(CI_results)
