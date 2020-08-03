import os
import glob
from omegaconf import OmegaConf
import pandas as pd


def convert_to_dataframe(target_dir):
    """
    convert test results to dataframe.
    results are placed under 'target_dir' like follows,

    hoge/target_dir/**/hogehoge01/test
                                 /transfer
                   /**/hogehoge02/test
                                 /transfer
                   /**/hogehoge-3/test
                                 /train
    """
    def _parse_config(test_path: str):
        """
        """
        train_config_path = os.path.join(test_path, '..', 'train', 'config.yaml')
        transfer_config_path = os.path.join(test_path, '..', 'transfer', 'config.yaml')

        if os.path.exists(train_config_path):
            loaded_cofig = OmegaConf.load(train_config_path)
        elif os.path.exists(transfer_config_path):
            loaded_cofig = OmegaConf.load(transfer_config_path)
        else:
            print('"config.yaml" for "{}" is not found.'.format(test_path))
            return None

        series_config = pd.Series()
        for k, v in OmegaConf.to_container(loaded_cofig).items():

            if type(v) == dict:
                for kk, vv in v.items():
                    if type(vv) == dict:
                        raise ValueError('Nest of config is too deep.')
                    else:
                        series_config['{k}.{kk}'.format(k=k, kk=kk)] = vv
            else:
                series_config[k] = v

        return series_config

    def _parse_acc(test_path: str):
        """
        """
        acc_path = os.path.join(test_path, 'acc', 'local_log.csv')

        if os.path.exists(acc_path):
            loaded_acc = pd.read_csv(acc_path, index_col=0)
        else:
            print('"{}/acc/local_log.csv" is not found.'.format(test_path))
            return None

        return loaded_acc.drop('time stamp', axis=1)

    def _parse_fourier(test_path: str):
        """
        """
        fourier_path = os.path.join(test_path, 'fourier', 'fhmap.png')

        if os.path.exists(fourier_path):
            fhmap_path = pd.DataFrame([pd.Series([os.path.abspath(fourier_path)], index=['fhmap'])])
        else:
            print('"{}/fourier/fhmap.png" is not found.'.format(test_path))
            return None

        return fhmap_path

    if not os.path.exists(target_dir):
        raise FileNotFoundError('target_dir: "{}" is not found'.format(target_dir))

    test_paths = glob.glob(os.path.join(target_dir, '**', 'test'), recursive=True)
    df = pd.DataFrame()

    for test_path in test_paths:
        df_test = pd.DataFrame()

        # parse config
        parsed_config = _parse_config(test_path)
        if parsed_config is not None:
            df_test = df_test.append([parsed_config], ignore_index=True, sort=False)

        # parse acc
        parsed_acc = _parse_acc(test_path)
        if parsed_acc is not None:
            df_test = pd.concat([df_test, parsed_acc], axis=1)

        # parse fourier
        fhmap_path = _parse_fourier(test_path)
        if fhmap_path is not None:
            print(fhmap_path)
            df_test = pd.concat([df_test, fhmap_path], axis=1)

        df = df.append(df_test, ignore_index=True, sort=False)

    print(df)
    df.to_csv('../logs/test_csv.csv')


if __name__ == '__main__':
    target_dir = '../logs/viewer_test'
    convert_to_dataframe(target_dir)