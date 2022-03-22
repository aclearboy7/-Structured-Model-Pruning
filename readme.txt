vgg16_finetunewithsparsity_checkpoint.pth ，vgg16经过60%通道剪枝后再次稀疏化训练 epoch 160，best 93.95% ,final 93.82%
./pruned model/vgg16pruned2.pth.tar , vgg16_finetunewithsparsity_checkpoint再剪枝20%通道，剪枝后92.31%
vgg16_finetune2_checkpoint.pth.tar , vgg16第二次剪枝后的finetune, best 93.95% ,final 93.79%