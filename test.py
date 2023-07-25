#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:36 2019

@author: xingyu
"""

from numpy import pad
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys



if __name__ == "__main__":
    sys.path.append(".") 
    from utils import summary
    from net   import snoring_net

    device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = snoring_net(numclasses_frame=2, mchannel=1).to(device)    
    summary(model, input_size=(1, 64, 64), batch_size=1, device=device)

    model.eval() 
    x = torch.randn((1,1,64,64)) # 输入张量
    torch.onnx.export(model, # 搭建的网络
        x,
        'model.onnx', # 输出模型名称
        input_names=["input"], # 输入命名
        output_names=["output"], # 输出命名
        dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
    )