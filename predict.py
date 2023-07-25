import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from net import snoring_net
import librosa
import pickle
import numpy as np
from pathlib import Path
import os
import sys
import numpy as np

import sounddevice as sd
import tempfile

import linger
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_fea(file_path):
    with open(file_path, 'rb') as f:
        fea = pickle.load(f)
    return fea

def predict_from_fea(model, fea_dir, threshold=3):
    fea_files = [os.path.join(fea_dir, f) for f in os.listdir(fea_dir) if f.endswith('.fea')]
    fea_files.sort()  # ensure the order

    prediction_list = []
    for fea_file in fea_files:
        fea = load_fea(fea_file)
        batch = torch.from_numpy(fea[np.newaxis, np.newaxis, ...]).float().to(device)
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)
        prediction_list.append(predicted.item())

    smooth_predictions = []
    for i in range(len(prediction_list)):
        start = max(0, i - threshold)
        end = min(len(prediction_list) - 1, i + threshold)
        smooth_predictions.append(round(sum(prediction_list[start:end+1]) / (end - start + 1)))
   
    snoring_count = sum(smooth_predictions)
    snoring_percentage = snoring_count / len(smooth_predictions)
    snoring_duration = snoring_count

    return smooth_predictions, snoring_duration, snoring_percentage

def main():
    model_path = "/data/user/mywang44/snoring_net/tmp.ignore/snoring_net.quant.best.pt"
    model = snoring_net(numclasses_frame=2, mchannel=1).to(device)

    dummy_input = torch.randn((1,1,64,64)) # 输入张量
    linger.trace_layers(model, model, dummy_input.to(device), fuse_bn=True)
    # linger.disable_normalize(net.last_layer)
    type_modules  = (nn.Conv2d)
    normalize_modules = (nn.Conv2d, nn.Linear)
    linger.normalize_module(model, type_modules = type_modules, normalize_weight_value=16, normalize_bias_value=16, normalize_output_value=16)
    nmodelet = linger.normalize_layers(model, normalize_modules = normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)        
    # linger.disable_quant(net.last_fc)
    quant_modules = (nn.Conv2d, nn.Linear)
    model = linger.init(model, quant_modules = quant_modules)

    model.load_state_dict(torch.load(model_path))
   
    fea_dir = "/data/user/mywang44/snoring_net/Snoring_Dataset/fea/1"
    prediction_list, snoring_duration, snoring_percentage = predict_from_fea(model, fea_dir)
   
    print("The prediction result is:", prediction_list)
    if snoring_duration > 0:
        print("Snoring event detected.")
        print(f"Snoring duration: {snoring_duration} seconds.")
        print(f"Snoring accounts for {snoring_percentage * 100}% of the total duration.")
    else:
        print("No snoring event detected.")

if __name__ == '__main__':
    main()
