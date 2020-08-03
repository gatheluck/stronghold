import glob
import os

import pandas as pd
from omegaconf import OmegaConf

HTML_TEMPLATE = """
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
  </head>
  <body>
    <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
    <div class="container">
        {table}
    </div>
  </body>
</html>
"""


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

    def _parse_source_option(soruce: str):
        """
        parse source option like follows
        fbdb_metric-balance_norm-l2_basis-0031_size-0032_cls-0496_standard
        """
        parsed_source = dict()
        split_by_underbar = soruce.split('_')

        if len(split_by_underbar) == 0:
            return pd.Series()

        for term in split_by_underbar:
            split_by_dash = term.split('-')
            if split_by_dash[0] == 'metric':
                parsed_source['source.metric'] = split_by_dash[1]
            elif split_by_dash[0] == 'basis':
                parsed_source['source.num_basis'] = int(split_by_dash[1])

        parsed_source['source.aug'] = split_by_underbar[-1]

        return pd.Series(parsed_source)

    def _parse_config(test_path: str):
        """
        """
        DROP_CONFIGS = [
            "normalize",
            "num_workers",
            "gpus",
            "prefix",
            "savedir",
            "num_nodes",
            "distributed_backend",
            "checkpoint_monitor",
            "checkpoint_mode",
            "unfreeze_params",
            "online_logger.name",
            "online_logger.activate",
            "dataset.mean",
            "dataset.std",
            "dataset.num_classes",
            "dataset.input_size",
            "optimizer.name",
            "optimizer.lr",
            "optimizer.momentum",
            "optimizer.weight_decay",
            "scheduler.name",
            "scheduler.milestones",
            "scheduler.gamma",
            "resume_ckpt_path"
        ]

        train_config_path = os.path.join(test_path, "..", "train", "config.yaml")
        transfer_config_path = os.path.join(test_path, "..", "transfer", "config.yaml")

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
                        raise ValueError("Nest of config is too deep.")
                    else:
                        series_config["{k}.{kk}".format(k=k, kk=kk)] = vv
            else:
                series_config[k] = v

            if (k == 'source') and (v is not None):
                series_config = series_config.append(_parse_source_option(v))

        return series_config.drop(labels=DROP_CONFIGS, errors='ignore')

    def _parse_acc(test_path: str):
        """
        """
        DROP_CONFIGS = ['advacc1', 'advacc5', 'time stamp']
        acc_path = os.path.join(test_path, "acc", "local_log.csv")

        if os.path.exists(acc_path):
            loaded_acc = pd.read_csv(acc_path, index_col=0)
        else:
            print('"{}/acc/local_log.csv" is not found.'.format(test_path))
            return None

        return loaded_acc.drop(DROP_CONFIGS, axis=1)

    def _parse_corruption(test_path: str):
        """
        """
        corruption_path = os.path.join(test_path, "corruption", "corruption_result.csv")
        corruption_img_path = os.path.join(test_path, "corruption", "plot_result.png")

        if os.path.exists(corruption_path):
            loaded_acc = pd.read_csv(corruption_path, index_col=0)
            mean_acc = pd.DataFrame([loaded_acc.mean()]).rename(columns={'accuracy': 'coracc'})
            print(mean_acc)
        else:
            print('"{}/corruption/corruption_result.csv" is not found.'.format(test_path))
            return None

        if os.path.exists(corruption_img_path):
            mean_acc['corruption'] = os.path.abspath(corruption_img_path)
        else:
            print('"{}/corruption/plot_result.png" is not found.'.format(test_path))
            return None

        return mean_acc

    def _parse_fourier(test_path: str):
        """
        """
        fourier_path = os.path.join(test_path, "fourier", "fhmap.png")

        if os.path.exists(fourier_path):
            parsed_fhmap = pd.DataFrame(
                [pd.Series([os.path.abspath(fourier_path)], index=["fhmap"])]
            )
        else:
            print('"{}/fourier/fhmap.png" is not found.'.format(test_path))
            return None

        return parsed_fhmap

    def _parse_spacial(test_path: str):
        """
        """
        spacial_path = os.path.join(test_path, "spacial", "plot.png")

        if os.path.exists(spacial_path):
            parsed_spacial = pd.DataFrame(
                [pd.Series([os.path.abspath(spacial_path)], index=["spacial"])]
            )
        else:
            print('"{}/spacial/plot.png" is not found.'.format(test_path))
            return None

        return parsed_spacial

    def _parse_layer(test_path: str):
        """
        """
        layer_path = os.path.join(test_path, "layer", "first_layer_weight.png")

        if os.path.exists(layer_path):
            parsed_layer = pd.DataFrame(
                [pd.Series([os.path.abspath(layer_path)], index=["layer"])]
            )
        else:
            print('"{}/layer/first_layer_weight.png" is not found.'.format(test_path))
            return None

        return parsed_layer

    def _parse_sensitivity(test_path: str):
        """
        """
        map_paths = glob.glob(os.path.join(test_path, 'sensitivity', '*000.png'))

        if len(map_paths) >= 1:
            parsed_sensitivity = pd.DataFrame(
                [pd.Series([os.path.abspath(map_paths[0])], index=["sensitivity"])]
            )
        else:
            print('"{}/sensitivity/*000.png" is not found.'.format(test_path))
            return None

        return parsed_sensitivity

    # main function from HERE
    if not os.path.exists(target_dir):
        raise FileNotFoundError('target_dir: "{}" is not found'.format(target_dir))

    test_paths = glob.glob(os.path.join(target_dir, "**", "test"), recursive=True)
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

        # parse corruption
        parsed_corruption = _parse_corruption(test_path)
        if parsed_corruption is not None:
            df_test = pd.concat([df_test, parsed_corruption], axis=1)

        # parse fourier
        parsed_fhmap = _parse_fourier(test_path)
        if parsed_fhmap is not None:
            df_test = pd.concat([df_test, parsed_fhmap], axis=1)

        # parse spacial
        parsed_spacial = _parse_spacial(test_path)
        if parsed_spacial is not None:
            df_test = pd.concat([df_test, parsed_spacial], axis=1)

        # parse fist layer
        parsed_layer = _parse_layer(test_path)
        if parsed_layer is not None:
            df_test = pd.concat([df_test, parsed_layer], axis=1)

        # parse sensitivity
        parsed_sensitivity = _parse_sensitivity(test_path)
        if parsed_sensitivity is not None:
            df_test = pd.concat([df_test, parsed_sensitivity], axis=1)

        # append to global df
        df = df.append(df_test, ignore_index=True, sort=False)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--target_dir", type=str, required=True)
    parser.add_argument("-p", "--save_path", type=str, default="../logs/viwer_output")
    opt = parser.parse_args()
    opt.save_path = os.path.splitext(opt.save_path)[0]

    df = convert_to_dataframe(opt.target_dir)
    with pd.ExcelWriter(opt.save_path + '.xlsx') as writer:
        df.to_excel(writer)

    if "corruption" in df.columns:
        df["corruption"] = df["corruption"].map(lambda s: "<img src='{}' height='100' />".format(s))
    if "fhmap" in df.columns:
        df["fhmap"] = df["fhmap"].map(lambda s: "<img src='{}' height='100' />".format(s))
    if "spacial" in df.columns:
        df["spacial"] = df["spacial"].map(lambda s: "<img src='{}' height='100' />".format(s))
    if "layer" in df.columns:
        df["layer"] = df["layer"].map(lambda s: "<img src='{}' height='100' />".format(s))
    if "sensitivity" in df.columns:
        df["sensitivity"] = df["sensitivity"].map(lambda s: "<img src='{}' height='100' />".format(s))

    table = df.to_html(classes=["table", "table-bordered", "table-hover"], escape=False)
    html_str = HTML_TEMPLATE.format(table=table)
    with open(opt.save_path + '.html', "w") as f:
        f.write(html_str)
