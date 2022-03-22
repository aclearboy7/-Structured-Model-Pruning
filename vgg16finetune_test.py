# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import models
from torchsummary import summary
from ptflops import get_model_complexity_info


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16, help='depth of the vgg')
parser.add_argument('--refine', default='./pruned model/vgg16pruned2.pth.tar', type=str, metavar='PATH', help='path to the pruned model to be fine tuned. e.g. ./pruned model/xxx.pth.tar')
parser.add_argument('--param', default='./weights/vgg16_finetune2_checkpoint.pth.tar', type=str, metavar='PATH', help='path to the model (default: none)')
parser.add_argument('--arch', default='vgg', type=str, help='architecture to use')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['cfg'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

# if args.param:
#     if os.path.isfile(args.param):
#         print("=> loading checkpoint '{}'".format(args.param))
#         checkpoint = torch.load(args.param)
#         args.start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
#         model.load_state_dict(checkpoint['state_dict'])
#         print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
#               .format(args.param, checkpoint['epoch'], best_prec1))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.model))

print(model)

'''原模型测试结果'''
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
    # print('pred',pred)
    return pred

acc = test(model)
summary(model,(3,32,32))

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3,32,32), as_strings=False, print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))