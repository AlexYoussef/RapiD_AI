from sklearn.model_selection import train_test_split, StratifiedKFold

from models.models_manager import BasicModelManager
from utils import evaluation_metrics
from utils.evaluation_metrics import get_best_roc_threshold


class BasicModelsWrapper:
    """
    Basic model is the high level wrapper of how the models where we control the training process
    """

    def __init__(self, model_manager: BasicModelManager, train_verbose=False):
        self.model_manager = model_manager
        self.clf = model_manager.get_new_instance()
        self.train_verbose = train_verbose
        self.initial_seed = model_manager.get_seed()

    def train(self, x_train, y_train, x_val, y_val):
        self.clf = self.model_manager.fit_classifier(self.clf, x_train, y_train, x_val, y_val, self.train_verbose)

    def predict_proba(self, x_val):
        return self.model_manager.predict_proba(self.clf, x_val)

    def __get_best_threshold(self, x_val, y_val, criterion='AUROC'):
        y_prob = self.predict_proba(x_val)
        if criterion:
            chosen_threshold, _ = get_best_roc_threshold(y_val, y_prob)
        else:
            print("Criterion not found, setting threshold to 0.5")
            chosen_threshold = 0.5
        return chosen_threshold

    def train_with_no_confidence_estimation(self, x_data, y_data, x_test, y_test, threshold='auto'):
        # create train/val splits
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.20, random_state=1)
        # train model
        self.clf = self.model_manager.get_new_instance()
        self.train(x_train, y_train, x_val, y_val)
        # find best threshold, this includes
        if threshold == 'auto':
            chosen_threshold = self.__get_best_threshold(x_val, y_val)
        else:
            chosen_threshold = threshold
        # find metrics
        train_metrics = self.test(x_train, y_train, chosen_threshold)
        val_metrics = self.test(x_val, y_val, chosen_threshold)
        test_metrics = self.test(x_test, y_test, chosen_threshold)

        return train_metrics, val_metrics, test_metrics

    def train_with_confidence_estimation(self, x_data, y_data, x_test, y_test, n=5, confidence=0.95,
                                         threshold='auto'):
        metrics_all_train = []
        metrics_all_val = []
        metrics_all_test = []

        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        # enumerate the splits and summarize the distributions
        for cv_train_idx, cv_val_idx in kfold.split(x_data, y_data):
            # get fold
            x_train, x_val = x_data[cv_train_idx], x_data[cv_val_idx]
            y_train, y_val = y_data[cv_train_idx], y_data[cv_val_idx]
            # train model
            self.clf = self.model_manager.get_new_instance()
            self.train(x_train, y_train, x_val, y_val)
            # set threshold
            if threshold == 'auto':
                chosen_threshold = self.__get_best_threshold(x_val, y_val)
            else:
                chosen_threshold = threshold
            # find metrics
            metrics_all_train.append(self.test(x_train, y_train, chosen_threshold))
            metrics_all_val.append(self.test(x_val, y_val, chosen_threshold))
            metrics_all_test.append(self.test(x_test, y_test, chosen_threshold))

        # calculate confidence
        train_confidence = evaluation_metrics.compute_metrics_with_confidence_estimation(metrics_all_train,
                                                                                         confidence=confidence)
        val_confidence = evaluation_metrics.compute_metrics_with_confidence_estimation(metrics_all_val,
                                                                                       confidence=confidence)

        test_confidence = evaluation_metrics.compute_metrics_with_confidence_estimation(metrics_all_test,
                                                                                        confidence=confidence)
        return train_confidence, val_confidence, test_confidence

    def test(self, x_test, y_test, threshold):
        """
        Test the model using the given threshold and return metrics
        """
        y_score = self.predict_proba(x_test)
        y_pred = (y_score > threshold).astype('int')
        return evaluation_metrics.compute_basic_metrics(y_test, y_score, y_pred)
