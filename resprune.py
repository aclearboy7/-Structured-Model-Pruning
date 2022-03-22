# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16, help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.7, help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./weights/res164sparsity_model_best.pth.tar', type=str, metavar='PATH', help='path to the model (default: none)')
parser.add_argument('--save', default='./save_CIFAR100', type=str, metavar='PATH', help='path to save pruned model (default: none)')
# args = parser.parse_args()
args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        # checkpoint = torch.load(args.model, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], False)
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

print(model)

def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # 加载测试数据
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # 对R, G，B通道应该减的均值
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # 记录类别预测正确的个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct.numpy() / len(test_loader.dataset)))
    
    pred = correct.numpy() / len(test_loader.dataset)
    print('pred',pred)
    return pred

# 原模型测试结果
acc = test(model)

#********************************预剪枝*********************************#
# 统计网络中BN的weights总共有多少个，即总共被剪枝的channel scaling factors总数
total = 0 # vgg16: 4224
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

# 确定剪枝的全局阈值
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
# 按照权值大小排序
y, i = torch.sort(bn) # y -value, i -index
thre_index = int(total * args.percent)
# 确定要剪枝的阈值，小于这个值的剪掉
thre = y[thre_index]


pruned = 0
cfg = [] # 剪枝后，每层的通道数
cfg_mask = [] # 每一BN层的mask
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # 要保留的通道标记Mask图,1保留,0剪枝
        mask = weight_copy.gt(thre.cuda()).float()
        # 剪枝掉的通道数总数 = 之前层剪掉的通道总数 + 本层原通道数 - 保留的通道数
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask) # mask掉bn的weights和bias，1保留
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask))) # 该BN层保留的通道数
        cfg_mask.append(mask.clone())   # 记录该BN层的mask
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
#********************************预剪枝后model测试*********************************
def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # 加载测试数据
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # 对R, G，B通道应该减的均值
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # 记录类别预测正确的个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct.numpy() / len(test_loader.dataset)))
    
    pred = correct.numpy() / len(test_loader.dataset)
    print('pred',pred)
    return pred

acc = test(model)

#------------------------------正式剪枝----------------------------------#
# Make real prune
print(cfg)
newmodel = vgg(dataset=args.dataset, depth=16, init_weights=True, cfg=cfg)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()]) # 统计剪枝后参数量
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
# 定义原始模型和新模型的每一层保留通道索引的mask
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    # 对BN层剪枝
    if isinstance(m0, nn.BatchNorm2d):
        # np.squeeze(array)：把array降维
        # np.argwhere(array)：返回array中非0元素的索引
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # 如果维度是1，那么就新增一维，这是为了和BN层的weight的维度匹配
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        m1.weight.data = m0.weight.data[idx1.tolist()].clone() # BN的weights
        m1.bias.data = m0.bias.data[idx1.tolist()].clone() # BN的bias
        m1.running_mean = m0.running_mean[idx1.tolist()].clone() # BN的running_mean
        m1.running_var = m0.running_var[idx1.tolist()].clone() # BN的running_var

        # BN剪枝完成后，就可以把end_mask作为start_mask对下一层卷积进行剪枝，新的end_mask = cfg_mask[layer_id_in_cfg + 1 ]
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    
    # 对卷积层剪枝
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        # 对一个卷积层的参数剪枝（两步），注意卷积核Tensor维度为[c_out, c_in, w, h]，c_out为输出通道数（即filter个数），c_in为输入通道数
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() # 对c_in剪枝，对每个filter裁剪掉不必要的通道
        w1 = w1[idx1.tolist(), :, :, :].clone() # 再对c_out剪枝，裁剪掉不必要的filter
        m1.weight.data = w1.clone() # 该卷积层剪枝完成，复制新的权重到新模型
    
    # 对全连接层剪枝
    elif isinstance(m0, nn.Linear):
        # 注意卷积核Tensor维度为[n, c, w, h]，两个卷积层连接，下一层的输入维度n'就等于当前层的c
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()
        

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'vgg16_'+ str(args.percent) +'_pruned.pth.tar'))

print(newmodel)
test(newmodel)
print("Number of parameters: \n"+str(num_parameters)+"\n")

