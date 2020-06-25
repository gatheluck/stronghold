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

## Transfer
If you want to do transfer learning 

| source | target | batch size | ep  | optim | lr | mom | decay | schduler | step | gamma | ref
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| ImagNet | CIFAR | 128  | 50 | SGD | 0.01 | 0.9 | 0.0001 | step | 30 | 0.1 | https://openreview.net/pdf?id=ryebG04YvB

Example code:
```
cd apps
python transfer.py weight=[PATH_TO_WEIGHT_OR_CHEKPOINT] original_num_classes=22 
```

## Test

Example code:
```
cd apps
python test.py ckpt_path=[PATH_TO_CHEKPOINT]
```
By default, .ckpt file is saved under `logs/train/yyyy-mm-dd_tt-mm-ss/checkpoint/epoch=XX-val_loss_avg=X.XX.ckpt`.


## Note
### Train
| dataset | model | loss | train acc | val acc | batch size | optim | lr | mom | decay | schduler | step | gamma
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
| fbdb_l2_basis-0031_cls-0022 | resnet50 | 0.1136  | 97.31 | 96.39 | 256 | SGD | 0.01 | 0.9 | 0.0001 | multi step | 30,60,80 | 0.1

### Transfer
| source | target  | model | batch size | ep  | loss | train acc | val acc | batch size | optim | lr | mom | decay | schduler | step | gamma | unfreeze
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| fbdb_l2_basis-0031_cls-0022 | cifar10  | resnet50 | 256 | 50  | 2.180 | 18.74 | 19.35 | 256 | SGD | 0.01 | 0.9 | 0.0001 | step | 30 | 0.1 | layer4.2.bn3.weight, layer4.2.bn3.bias, fc.weight, fc.bias