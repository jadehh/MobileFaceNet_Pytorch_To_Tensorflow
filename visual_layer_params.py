#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/30 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/30  上午10:05 modify by jade
import torch
import numpy as np
import cv2
import tensorflow as tf

params = np.load("npy/all_params.npy")
image1 = cv2.imread("examples/2019-08-27-18-29-13.jpg")
image1 = cv2.resize(image1, (112, 112))
image2 = cv2.imread("examples/test.jpg")
image2 = cv2.resize(image2, (112, 112))
image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

image = np.array([image1, image2])
np.save("npy/image.npy", image)
image = np.transpose(image, (0, 3, 1, 2))
image = torch.from_numpy(image).type(torch.FloatTensor)
batch = 2
conv1_weight = np.load("npy/conv1.conv.weight.npy")
conv1_bn_weight = np.load("npy/conv1.bn.weight.npy")
conv1_bn_biases = np.load("npy/conv1.bn.bias.npy")

conv1_bn_biases = np.zeros_like(conv1_bn_biases)
conv1_bn_weight = np.ones_like(conv1_bn_weight)
kernel = (3, 3)
stride = (2, 2)
conv1 = torch.nn.Conv2d(3, 64, kernel_size=kernel, stride=stride, padding=(1, 1), bias=False)
batch_norm = torch.nn.BatchNorm2d(64, affine=False)
batch_norm.momentum = 0
conv1.weight.data = torch.from_numpy(conv1_weight)  # 给卷积的 kernel 赋值
# batch_norm.weight.data = torch.from_numpy(conv1_bn_weight)
# batch_norm.bias.data = torch.from_numpy(conv1_bn_biases)
conv_result = conv1(image)
conv_np = conv_result.detach().numpy()
result1 = batch_norm(conv_result).detach().numpy()




junzhi = np.mean(conv_np, axis=0,keepdims=True)
fangcha = np.var(conv_np, axis=0,keepdims=True)
result2 = (conv_np - junzhi) / (np.sqrt(fangcha))

np.save("result.npy", result2)

# print(result)
