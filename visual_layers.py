#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/30 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/30  上午10:13 modify by jade
import torch
import cv2
from nets.mobileface_net_th import MobileFaceNet
import numpy as np
model = MobileFaceNet(embedding_size=512)
model.load_state_dict(torch.load("models/model_mobilefacenet.pth"))
model.eval()
print(model.conv1)
image = cv2.imread("examples/2019-08-27-18-29-13.jpg")
image = cv2.resize(image, (112, 112))
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = np.transpose(image, (2, 0, 1))
image = np.array([image])
image = torch.from_numpy(image).type(torch.FloatTensor)
#print(model.conv1(image))