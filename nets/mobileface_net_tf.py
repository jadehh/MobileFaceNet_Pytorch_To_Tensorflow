#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/30 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/30  上午10:16 modify by jade
import tensorflow as tf
from tensorflow.python.keras.layers import *
from custom_Layers.customLayers import *
import cv2
import numpy as np
from tensorflow.python.keras import Model
class Conv_block(Layer):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1,name=""):
        super(Conv_block, self).__init__()
        self.padding = padding
        if self.padding:
            self.pad = ZeroPadding2D(padding=(self.padding,self.padding),name=name)
            self.pad_name = "valid"
        else:
            self.pad_name = "same"
        self.conv = Conv2D(out_c, kernel_size=kernel, strides=stride, padding='valid', use_bias=False,name=name)
        self.bn = BatchNorm2D(out_c,name=name)
        # self.prelu = PReLU(name=name)


    def call(self, inputs):
        if self.padding:
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        x = self.bn(x)
        # x = self.prelu(x)
        return x

class MobileFaceNet(Model):
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1,1),name="conv1")
    def call(self, inputs):
        x = self.conv1(inputs)
        return x

def bias_traonspose(path):
    return np.load("../npy/"+path)

def params_transpose(path):
    params = np.load("../npy/"+path)
    if params.ndim == 4:
        return np.transpose(params,[2,3,1,0])
    else:
        return params

def set_layer_params(layer,layer_layer_name):
    conv = layer.conv
    weights_param = params_transpose(layer_layer_name+".conv.weight"+".npy")
    conv.set_weights([tf.convert_to_tensor(weights_param)])

    bn = layer.bn
    bias_param = bias_traonspose(layer_layer_name + ".bn.bias.npy")
    bn.set_biases(bias_param)
    bn_weights_param = params_transpose(layer_layer_name+".bn.weight.npy")
    bn.set_weights(bn_weights_param)





if __name__ == '__main__':
    image = cv2.imread("../examples/2019-08-27-18-29-13.jpg")
    image = cv2.resize(image,(112,112))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = np.array([image],dtype=np.float32)
    image = tf.convert_to_tensor(image)
    model = MobileFaceNet()
    model.build(input_shape=(None,112,112,3))
    layer1 = model.conv1
    set_layer_params(layer1,'conv1')
    conv1_1 = np.transpose(layer1(image).numpy(),[0,3,1,2])

    conv1_1_result = np.load("../result.npy")
    success = conv1_1 == conv1_1_result

    # set_layer_params(layer1_3,weights_path='conv1.prelu')
    print("Done")


