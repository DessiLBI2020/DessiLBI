mkdir data
CUDA_VISIBLE_DEVICES=0 python pretrain_lbi_weight_norm.py --model vgg16_bn --epoch 200 --kappa 1 --mu 500 --post_mode finetune  --logname ./tblog/vgg16bn_lbi_weight_mu500_norm_0.0003 --fc_lambda 0.0003 --lbi_momentum 0 
