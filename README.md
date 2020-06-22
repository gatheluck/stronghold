# Stronghold

## Train

Example code:
```
cd apps
python train.py dataset=cifar10
```

if you want to check logs, please run the following code.

```
cd logs/train/[OUTPUT_DIR]
mlflow ui 
```

## Test

Example code:
```
cd apps
python test.py ckpt_path=[PATH_TO_CHEKPOINT]
```
By default, .ckpt file is saved under `logs/train/yyyy-mm-dd_tt-mm-ss/checkpoint/epoch\=X-val_loss_avg\=X.XX.ckpt`.