import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import importlib
import pandas
from datetime import datetime
from collections import OrderedDict

from libs.utils import get_time_stamp


class Logger():
    SUPPERTED_MODES = set('train val test'.split())

    def __init__(self, path: str, mode: str, flags=None):
        if mode not in self.SUPPERTED_MODES:
            raise ValueError

        self.path = path
        self.mode = mode
        self.df = pandas.DataFrame()
        self._save()

        self.row_index = 0

    def log(self, metrics: dict, step=None):
        """
        take new log to csv file.

        Args:
        - metrics: dict of metrics to log. like, {'loss': 0.32, 'acc': 89.6}
        - step: step of logging.

        logged csv file will become following

            step | time stamp | val01 | val02 | val03
        0
        -
        1
        -
        """

        self.df = pandas.read_csv(self.path, index_col=0)
        time_stamp = get_time_stamp()

        # create data dict for adding new data to csv file
        datadict = OrderedDict()
        if (self.mode == 'train') or (self.mode == 'val'):
            datadict['step'] = int(step)
        datadict['time stamp'] = time_stamp

        for k, v in metrics.items():
            datadict[k] = v

        # adding new dataframe to self.df and save
        new_df = pandas.DataFrame(datadict, index=[self.row_index])
        self.df = self.df.append(new_df, sort=False)
        self._save()

        self.row_index += 1

    def _save(self):
        try:
            self.df.to_csv(self.path)
        except ValueError:
            print('self.path is invalid path')


if __name__ == '__main__':
    log_path_root = '../logs/logger_test'
    log_basename = 'log_test_' + get_time_stamp('short') + '.csv'
    log_path = os.path.join(log_path_root, log_basename)
    os.makedirs(log_path_root, exist_ok=True)

    logger = Logger(log_path, mode='test')

    metrics = {'loss01': 1.0, 'loss02': 2.0}
    metrics_ = {'loss01': 1.0, 'loss03': 3.0}
    logger.log(metrics, 1)
    logger.log(metrics, 2)
    logger.log(metrics, 3)
    logger.log(metrics_, 4)
