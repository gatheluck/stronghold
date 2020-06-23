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
By default, .ckpt file is saved under `logs/train/yyyy-mm-dd_tt-mm-ss/checkpoint/epoch=XX-val_loss_avg=X.XX.ckpt`.


## Note

| dataset | model | loss | train acc | val acc | batch size | optim | lr | mom | decay | schduler | step | gamma
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| fbdb_l2_basis-0031_cls-0022 | resnet50 | 0.1136  | 97.31 | 96.39 | 256 | SGD | 0.01 | 0.9 | 0.0001 | multi step | 30,60,80 | 0.1
