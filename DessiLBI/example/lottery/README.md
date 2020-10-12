# DessiLBI WITH EARLY STOPPING AND RETRAIN FINDS EFFECTIVE SUB-NETWORKS

This fold contains codes for using DessiLBI find winning tickets. Thanks for [Rethinking the Value of Network Pruning (Pytorch) (ICLR 2019)](https://github.com/Eric-mingjie/rethinking-network-pruning) release their code. This implementation relies heavily on this repo.

## Dependencies
torch                v1.0+,
torchvision          v0.1.8,
tensorboardX         v1.8.

## How to run?

"cifar.py" is used to train with SLBI to find mask, then "lottery_ticket.py" is used to retrain the model. 

To reproduce our results, you can run as follows

```shell
# resnet56_group
python cifar.py --epochs 200 --schedule 300 --arch resnet --depth 56  --optimizer slbi --mu 100 --kappa 1  --conv_lambda 0.1 --fc_lambda 1 --save_dir resnet56_group 

# vgg16_group
python cifar.py --epochs 200 --schedule 300 --arch vgg16_bn --depth 16  --optimizer slbi --mu 100 --kappa 1 --fc_lambda 1 --conv_lambda 0.05 --save_dir vgg16_group &

# resnet50_lasso
python cifar.py --epochs 200 --schedule 300 --arch resnet --depth 50  --optimizer slbi --mu 500 --kappa 1 --fc_lambda 0.05 --save_dir resnet50_lasso &

# vgg16_lasso
python cifar.py --epochs 200 --schedule 300 --arch vgg16_bn --depth 16  --optimizer slbi --mu 200 --kappa 1 --fc_lambda 0.03 --conv_lambda 0.0 --save_dir vgg16_lasso 
```

Then, the followed codes can be used to retrain the model
```shell
python lottery_ticket.py --arch [network arch] --depth [depth]  --save_dir [save_dir for previous run] --gamma_supp True --gamma_epoch [epoch you want to use to load mask] 

#For example
python lottery_ticket.py --arch resnet --depth 56  --save_dir resnet56_group --gamma_supp True --gamma_epoch 35 
```
