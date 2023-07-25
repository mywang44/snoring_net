# -*- coding: utf-8 -*-

import linger as linger

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
import torchaudio
from torch.utils.data import Dataset, DataLoader

# from fastprogress import master_bar, progress_bar
import numpy as np
import time
from torchvision.models import *
from torchvision.transforms import transforms
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings(action="ignore")

from net import snoring_net
from tqdm import tqdm
import sys

import pdb

from sklearn.metrics import precision_recall_fscore_support
import random

import pickle



num_classes = 2
lr = 0.0002
eta_min = 1e-5
t_max = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import numpy as np


# 定义一个自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.label_list[idx]

        mel_spec_db = []
        with open(file_path, 'rb') as fp:
            mel_spec_db = pickle.load(fp)
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)

        # 返回梅尔频谱图和标签
        return mel_spec_db, label

#-----------------------------------------augment--------------------------------------------
def prepare_data(files, labels):
    files = np.array(files)
    labels = np.array(labels)

    group_num = 4  # 原始样本 + 三个增强样本
    data_len = len(files)

    # 计算满足group_num倍数的最大样本数
    max_len = (data_len // group_num) * group_num

    # 切割出满足group_num倍数的样本
    files_grouped = files[:max_len].reshape(-1, group_num)
    labels_grouped = labels[:max_len].reshape(-1, group_num)

    # 处理不足一组的样本
    remain_files = files[max_len:]
    remain_labels = labels[max_len:]

    bounds = int(len(files_grouped)*0.7)

    # 按组划分训练集和测试集
    train_files  = files_grouped[:bounds].flatten().tolist()
    train_labels = labels_grouped[:bounds].flatten().tolist()
    test_files   = files_grouped[bounds:].flatten().tolist()
    test_labels  = labels_grouped[bounds:].flatten().tolist()

    # 将剩余的不足一组的样本全部放入训练集
    train_files += remain_files.tolist()
    train_labels += remain_labels.tolist()

    train_dataset = AudioDataset(train_files, train_labels)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset  = AudioDataset(test_files, test_labels)
    test_loader   = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(data_len, len(train_loader), len(test_loader))

    return train_loader, test_loader, test_dataset


def train(train_loader, test_loader, test_dataset, mode = "float", load_model_path = None, num_epochs = 50):
    net = snoring_net(numclasses_frame=2, mchannel=1).to(device) 
    # print(net)

    dummy_input = torch.randn((1,1,64,64)) # 输入张量
    # 使用linger进行浮点约束设置
    if mode == "float":
        print("original float train...")
    elif mode == "clamp" or mode == "quant":
        print("clamp train...")              
        linger.trace_layers(net, net, dummy_input.to(device), fuse_bn=True)
        # linger.disable_normalize(net.last_layer)
        type_modules  = (nn.Conv2d)
        normalize_modules = (nn.Conv2d, nn.Linear)
        linger.normalize_module(net, type_modules = type_modules, normalize_weight_value=16, normalize_bias_value=16, normalize_output_value=16)
        net = linger.normalize_layers(net, normalize_modules = normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)        
        if mode == "quant":   # 添加linger量化训练设置
            # linger.disable_quant(net.last_fc)
            quant_modules = (nn.Conv2d, nn.Linear)
            net = linger.init(net, quant_modules = quant_modules)
    else:
        assert("wrong mode, stop!")

    if load_model_path is not None:
        net.load_state_dict(torch.load(load_model_path), strict=True)
    # print(net)
    
    loss_function = nn.CrossEntropyLoss().cuda()
    # loss_function = nn.MSELoss().cuda()
    optimizer = Adam(params=net.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    # train
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            fea, labels = data
            optimizer.zero_grad()
            outputs = net(fea.to(device))  
            loss = loss_function(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        test_steps = len(test_loader)
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_fea, val_labels = val_data
                outputs = net(val_fea.to(device))
                predict_y = torch.argmax(outputs,  dim=1)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(test_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            save_path = "tmp.ignore/snoring_net." + mode + ".best.pt"

            # 检查并创建保存路径
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # 保存模型
            torch.save(net.state_dict(), save_path)



    from utils import summary
    net.eval()     
    # summary(net, input_size=(1, 64, 64), batch_size=1, device=device)
    with torch.no_grad():
        save_path = "tmp.ignore/snoring_net." + mode + ".onnx" 
        torch.onnx.export(net, 
                          dummy_input.to(device), 
                          save_path,
                          input_names=["input"], # 输入命名
                          output_names=["output"], # 输出命名
                          #dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}},  # 动态轴
                          export_params=True,
                          opset_version=12, 
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                          )

    return net

def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
   
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
           
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
       
    acc = sum([1 if true == pred else 0 for true, pred in zip(y_true, y_pred)]) / len(y_true)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f'Accuracy: {acc}, F1: {f1}, Recall: {recall}')

    return acc, f1, recall

# export PATH=/usr/bin:$PATH

def list_directory_contents(directory_path):
    # Create a list to store the subdirectories and files
    dirs_  = []
    files_ = []

    # Loop through all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory_path):
        # Add the subdirectories to the contents list
        for dir in dirs:
            dirs_.append(os.path.join(root, dir))
        # Add the files to the contents list
        for file in files:
            files_.append(os.path.join(root, file))

    # Return the contents list
    return dirs_, files_


data_path = '/home/nizai8a/snoring_net/Snoring_Dataset/fea'

if __name__ == "__main__":
    # train_files = [false_data + filename for filename in os.listdir(false_data)]
    _, files  = list_directory_contents(data_path)

    random.shuffle(files)
    labels = [int(filename.split('.')[0].split('/')[-2]) for filename in files]
    
    train_loader, test_loader, test_dataset = prepare_data(files, labels)

    # trained_net = train(train_loader, test_loader, test_dataset, mode = "float", load_model_path = None, num_epochs = 5)#浮点训练
    # trained_net = train(train_loader, test_loader, test_dataset, mode = "clamp", load_model_path = "./tmp.ignore/snoring_net.float.best.pt", num_epochs =3)#约束训练
    trained_net = train(train_loader, test_loader, test_dataset, mode = "quant", load_model_path = "./tmp.ignore/snoring_net.clamp.best.pt", num_epochs = 3)#量化训练


    acc, f1, recall = evaluate(trained_net, test_loader, device)

    print("done")
