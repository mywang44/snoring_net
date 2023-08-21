#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    dwwang@listenai.com  2023.05.17
"""

from numpy import pad
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys


# @staticmethod
class group_unit(nn.Module):            
    def __init__(self, in_channels, group_out_channels, pointwise_out_channels, kernel_size, stride, padding, group, casual_dim = 2, right_contex=1):
        super(group_unit, self).__init__()
        self.group_conv = nn.Conv2d(in_channels=in_channels, out_channels= group_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=group)
        self.activate   = nn.ReLU(inplace=True)
        self.pointwise  = nn.Conv2d(in_channels=group_out_channels, out_channels=pointwise_out_channels, kernel_size=(1,1), bias=False)

    def forward(self, src, mase=None):
        out = self.group_conv(src)
        out = self.activate(out)
        out = self.pointwise(out)
        return out


# @staticmethod
class conv_unit(nn.Module):            
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, group, casual_dim = 2, right_contex=1, is_active=True, is_Transpose=False):
        super(conv_unit, self).__init__()
        self.is_active = is_active
        self.is_transpose = is_Transpose

        if self.is_transpose:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=kernel_size)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=group)
        
        if self.is_active:
                self.activate   = nn.ReLU(inplace=True)

    def forward(self, src, mase=None):
        out = self.conv(src)
        if self.is_active:
            out = self.activate(out)
        return out



class snoring_net(nn.Module):
    def __init__(self, numclasses_frame, mchannel):
        super(snoring_net, self).__init__()
        self.conv0  = conv_unit(in_channels=mchannel, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=1)
        self.conv1  = conv_unit(in_channels=16,       out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=1)

        self.conv2  = conv_unit(in_channels=16,       out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=1)
        self.conv3  = conv_unit(in_channels=32,       out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=1)

        self.block1_0  = group_unit(in_channels=32,  group_out_channels=64, pointwise_out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4)
        self.block1_1  = group_unit(in_channels=32,  group_out_channels=64, pointwise_out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block1_2  = group_unit(in_channels=32,  group_out_channels=64, pointwise_out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block1_3  = group_unit(in_channels=32,  group_out_channels=64, pointwise_out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 

        self.block2_0  = group_unit(in_channels=32,  group_out_channels=128, pointwise_out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block2_1  = group_unit(in_channels=64,  group_out_channels=128, pointwise_out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block2_2  = group_unit(in_channels=64,  group_out_channels=128, pointwise_out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block2_3  = group_unit(in_channels=64,  group_out_channels=128, pointwise_out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        
        self.block3_0  = group_unit(in_channels=64,   group_out_channels=256, pointwise_out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block3_1  = group_unit(in_channels=128,  group_out_channels=256, pointwise_out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block3_2  = group_unit(in_channels=128,  group_out_channels=256, pointwise_out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        self.block3_3  = group_unit(in_channels=128,  group_out_channels=256, pointwise_out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), group=4) 
        
        self.fc1 = nn.Linear(128*2*4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, numclasses_frame)

        
    def forward(self, x, mask=None):        
        out = self.conv0(x)
        out = out + self.conv1(out)
        out = F.max_pool2d(out,  kernel_size=(2,1), ceil_mode=True)
        out = self.conv2(out)
        out = out + self.conv3(out)
        out = F.max_pool2d(out,  kernel_size=(2,2), ceil_mode=True)

        out = self.block1_0(out)
        out = out + self.block1_1(out)
        out = out + self.block1_2(out)
        out = out + self.block1_3(out)
        out = F.max_pool2d(out,  kernel_size=(2,2), ceil_mode=True)

        out = self.block2_0(out)
        out = out + self.block2_1(out)
        out = out + self.block2_2(out)
        out = out + self.block2_3(out)
        out = F.max_pool2d(out,  kernel_size=(2,2), ceil_mode=True)

        out = self.block3_0(out)
        out = out + self.block3_1(out)
        out = out + self.block3_2(out)
        out = out + self.block3_3(out)
        out = F.max_pool2d(out,  kernel_size=(2,2), ceil_mode=True)
        
        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = F.relu(self.fc3(out))
        out = F.softmax(out, dim=-1)
        # out = torch.log_softmax(out, dim=-1)
        # out = torch.argmax(out, dim=1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    sys.path.append(".") 
    from utils import summary

    device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = snoring_net(numclasses_frame=2, mchannel=1).to(device)    
    summary(model, input_size=(1, 64, 64), batch_size=32, device=device)

    model.eval() 
    x = torch.randn((32,1,64,64)) # 输入张量
    torch.onnx.export(model, # 搭建的网络
        x,
        'model.onnx', # 输出模型名称
        input_names=["input"], # 输入命名
        output_names=["output"], # 输出命名
        dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
    )
