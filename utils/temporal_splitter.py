"""
Name: Temporal splitter
Description:    In this file, we are building a temporal splitter. A temporal splitter splits and yields the dataset
                chronologically in a form of batches. For batch I it yields the dataset rows with deviation less than
                I * INTERVAL. The earliest timestamp in the dataset is considered the moment 0, and the difference
                between a timestamp X of a row R and the earliest timestamp is considered the deviation of the row.
                INTERVAL is the deviation widening ratio between batches measured in days.

Class definition:   we need the following parameters to initialize the splitter:
                        1) data: Pandas data frame with first column represents 0-based row indices.
                        2) timestamp_loc: string represents the name of column which contains the timestamps of each row
                        3) interval: integer represents the deviation widening ratio (in days)
"""
import pandas as pd


class TemporalSplitter:
    def __init__(self, data, timestamp_loc, timestamp_format, interval=7):
        self.timestamp_loc = timestamp_loc
        self.timestamp_format = timestamp_format
        self.len_data = data.shape[0]
        self.interval = interval
        self.interval_col = '__interval__'
        self.data = self.prepare_data(data)

    def prepare_data(self, data):
        """
        data preprocessor: sorts the data according to timestamps, and calculate the deviation from moment 0.
        """
        data[self.timestamp_loc] = pd.to_datetime(data[self.timestamp_loc])
        data.sort_values([self.timestamp_loc], inplace=True)
        time_diff = [0]
        for i in range(1, self.len_data):
            time_diff.append(abs(pd.Timedelta(data.loc[i, self.timestamp_loc] - data.loc[0, self.timestamp_loc]).days))
        data[self.interval_col] = pd.DataFrame(time_diff, dtype='int32')
        return data

    def get_splits(self):
        """
        data generator: at time step i returns data with deviation within the range [0, i * interval[
        """
        batch_id = 0
        max_interval = self.data[self.interval_col].max()
        while True:
            batch_min_interval = batch_id * self.interval
            batch_max_interval = (batch_id + 1) * self.interval
            if batch_min_interval <= max_interval:
                batch = (self.data.loc[self.data[self.interval_col] < batch_max_interval]).loc[:,
                        self.data.columns != self.interval_col]
                yield batch
                batch_id += 1
            else:
                break
