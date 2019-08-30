#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/30 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/30  下午6:28 modify by jade
import torch
import numpy as np

image1 = np.zeros(shape=[4,4,3])
image2 = np.ones(shape=[4,4,3])
image3 = np.ones(shape=[4,4,3]) * 2
input_np = np.array([image1,image2,image3],dtype=np.float32)
input_np = np.load("npy/input.npy")
# input_np = np.transpose(input_np,(0,3,1,2))
input = torch.from_numpy(input_np)
batch_norm = torch.nn.BatchNorm2d(3,affine=False)
output  = batch_norm(input)
output_np = output.detach().numpy()


input_np_reshape = np.sum(input_np,axis=0)
junzhi = np.mean(input_np, axis=0,keepdims=True)
fangcha = np.var(input_np, axis=0,keepdims=True)
# junzhi = np.reshape(junzhi, [1, 3, 1, 1])
# fangcha = np.reshape(fangcha, [1, 3, 1, 1])
result = (input_np-junzhi) / (np.sqrt(fangcha)+batch_norm.eps)



print("Done")