"""
This file defines different types of data preprocessor
currently we have 2 simple types: one for mapping labels (e.g. for binariztion), and the other to select features
"""
from typing import List

import pandas as pd


class DatasetProcessor:
    def __init__(self):
        pass

    def process(self, dataset: pd.DataFrame):
        pass


class FeatureSelector(DatasetProcessor):
    def __init__(self, features=[]):
        super(FeatureSelector, self).__init__()
        self.features = features

    def process(self, dataset: pd.DataFrame):
        return self.select(dataset)

    def select(self, dataset: pd.DataFrame):
        return dataset[self.features]


class LabelProcessor(DatasetProcessor):
    def __init__(self, labels_map: {}, labels_col='label'):
        super(LabelProcessor, self).__init__()
        self.labels_map = labels_map
        self.labels_col = labels_col

    def process(self, dataset: pd.DataFrame):
        dataset.loc[:, self.labels_col] = dataset[self.labels_col].map(self.labels_map)
        return dataset


