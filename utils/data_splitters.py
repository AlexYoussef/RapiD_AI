from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

"""
This file defines the data splitter, it also guarantees the minimal intersection between train/ val/ test splits.
"""


class PreventDataSplitter:

    def __init__(self, train_p, val_p, test_p, data, label_col, timestamp_col, patient_col):
        assert train_p + val_p + test_p == 1, f'Sum of splits pct must be 1 got {train_p + val_p + test_p}'
        self.train_p = train_p
        self.val_p = val_p
        self.test_p = test_p
        self.data = data
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        self.patient_col_idx = data.columns.get_loc(patient_col)
        self.patient_col = patient_col

    def split(self, show_distribution=True, solve_intersection=True, resolver='trivial'):
        # sort data to avoid temporal intersection
        self.data.loc[:, self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
        self.data = self.data.sort_values(axis=0, by=[self.timestamp_col])
        splits = {'train': [], 'val': [], 'test': []}
        # group by label because we want same ratio for each label in the splits
        classes = self.data.groupby(self.label_col)
        split_size = {'train': 0, 'val': 0, 'test': 0}
        # distribute classes on each split
        for label, subdf in classes:
            total = subdf.shape[0]  # Number of samples in each split (split was done )
            split_size['train'] = int(total * self.train_p)
            split_size['val'] = int(total * self.val_p)
            split_size['test'] = total - split_size['train'] - split_size['val']

            start, end = 0, 0
            for split in ['train', 'test', 'val']:
                end = min(end + split_size[split], total)
                splits[split].append(subdf.iloc[start:end, :])
                start = min(start + split_size[split], total)
                intersection = 0
                while True:  # Make sure patient exists only in one split
                    if end == total:
                        break
                    if subdf.iloc[end - 1, self.patient_col_idx] == subdf.iloc[end, self.patient_col_idx]:
                        intersection += 1
                        end += 1
                    else:
                        break
                if intersection > 0:
                    splits[split].append(subdf.iloc[start:end, :])
                    start += intersection

        # concatenate splits' classes
        for split in ['train', 'val', 'test']:
            splits[split] = pd.concat(splits[split], axis=0)
        # remove patients intersection
        if solve_intersection is True:
            splits['train'], splits['val'], splits['test'] = self.remove_intersection(splits['train'], splits['val'],
                                                                                      splits['test'], resolver=resolver)

        if show_distribution:
            print('Train distribution:')
            print('    Train%: {:.2f}% ({} samples)'.
                  format(100 * splits['train'].shape[0] / self.data.shape[0],
                         splits['train'].shape[0]
                         )
                  )
            print('    Classes distribution in train:')
            for label, pct in self.calculate_classes_distribution(splits['train']).items():
                print('        label: {}: {:.2f}({} samples)'.format(label, pct[0] * 100, pct[1]))
            print('----- ****** ------')
            print('Val distribution:')
            print('      Val%: {:.2f}% ({} samples)'.
                  format(100 * splits['val'].shape[0] / self.data.shape[0],
                         splits['val'].shape[0]
                         )
                  )
            print('    Classes distribution in val:')
            for label, pct in self.calculate_classes_distribution(splits['val']).items():
                print('        label: {}: {:.2f}% ({} samples)'.format(label, pct[0] * 100, pct[1]))
            print('----- ****** ------')
            print('Test distribution:')
            print('    Test%: {:.2f}% ({} samples)'.
                  format(100 * splits['test'].shape[0] / self.data.shape[0],
                         splits['test'].shape[0]
                         )
                  )
            print('    Classes distribution in test:')
            for label, pct in self.calculate_classes_distribution(splits['test']).items():
                print('        label: {}: {:.2f}% ({} samples)'.format(label, pct[0] * 100, pct[1]))

        return splits['train'], splits['val'], splits['test']

    def remove_intersection(self, train_split, val_split, test_split, resolver='trivial'):
        """
        Remove intersection between 2 sets splitting the intersected data between the two sets
        """
        val_split, train_split = self.remove_intersection_2_sets(splitA=val_split, splitB=train_split,
                                                                 resolver=resolver)
        test_split, val_split = self.remove_intersection_2_sets(splitA=test_split, splitB=val_split, resolver=resolver)
        test_split, train_split = self.remove_intersection_2_sets(splitA=test_split, splitB=train_split,
                                                                  resolver=resolver)
        return train_split, val_split, test_split

    def remove_intersection_2_sets(self, splitA, splitB, resolver='trivial'):
        # Define intersection between patients
        intersection = np.intersect1d(splitA['hadm_id'], splitB['hadm_id'])
        to_splitA = True
        # patients in intersection must be moved to one split
        if resolver == 'trivial':
            '''
                trivial resolver assign even-numbered patients to train, and the remaining to test
            '''
            for patient_id in intersection:
                if to_splitA:
                    row_to_move = splitB.loc[splitB[self.patient_col] == patient_id]
                    splitA = pd.concat([splitA, row_to_move.copy()])
                    splitB.drop(row_to_move.index, inplace=True)
                    to_splitA = False
                else:
                    row_to_move = splitA.loc[splitA[self.patient_col] == patient_id]
                    splitB = pd.concat([splitB, row_to_move.copy()])
                    splitA.drop(row_to_move.index, inplace=True)
                    to_splitA = True
        else:
            raise Exception('Unimplemented resolver')

        return splitA, splitB

    def calculate_classes_distribution(self, split):
        split_total = split.shape[0]
        classes = split.groupby(self.label_col)
        distribution = {}
        for label, subdf in classes:
            class_total = subdf.shape[0]
            distribution[label] = (class_total / split_total, class_total)
        return distribution

    def is_acceptable_split(self, train_y, val_y, test_y):
        # all splits should contain samples from all classes
        classes_train = np.unique(train_y)
        classes_val = np.unique(val_y)
        classes_test = np.unique(test_y)
        classes = np.unique(np.concatenate((classes_train, classes_val, classes_test)))
        if classes.shape[0] * 3 == classes_train.shape[0] + classes_val.shape[0] + classes_test.shape[0]:
            return True
        else:
            return False
