# snoring_net
snoring detection project using LNN(linger&amp;thinker)

## 介绍
本仓库利用LNN工具链实现鼾声检测模型的落地。主要包括浮点训练、量化训练、模型打包、模拟引擎执行、固件烧录并芯片运行。其中固件烧录并芯片运行需要在聆思的开发板上来完成。

## 环境配置
### linger环境配置及安装
https://github.com/LISTENAI/linger/blob/main/doc/tutorial/install.md

### thinker环境配置及安装
https://github.com/LISTENAI/thinker/blob/main/thinker/docs/tutorial/install.md

## requirement
pandas-1.1.5  
scikit-learn-1.0.2  
torchaudio-0.9.0  
tqdm-4.65.0  
librosa-0.7.2  
resampy-0.2.2  
numba-0.48  

## 数据集
仓库采用了三个数据集，分别是Kaggle-Snoring、Kaggle-Female and Male Snoring和 ESC-50 数据库。  

Kaggle - Snoring   
数据来源：https://www.kaggle.com/datasets/tareqkhanemu/snoring  
数据介绍：该数据集包含两个文件夹，一个为鼾声，另一个为非鼾声。鼾声文件夹中含有总共500段鼾声音频，每段音频的持续时间为1秒。  
男性和女性鼾声数据集 (Female and Male Snoring)  

Kaggle - Female and Male Snoring  
数据来源：https://www.kaggle.com/datasets/orannahum/female-and-male-snoring  
数据介绍：男性和女性鼾声数据集中，每个wav文件的持续时间为1秒，采样率为44100Hz，共1000段鼾声音频。  

ESC-50  
数据来源：https://github.com/karolpiczak/ESC-50  
数据介绍：ESC-50是一个多用途、标注准确的环境声音数据集，包括50个类别和2000个音频样本（每个样本持续时间为5秒）。在本项目中，选取了数据集中非鼾声的音频，每条音频分割为1秒的片段，从而得到2500条非鼾声数据。  

为了保证数据平衡，本仓库使用数据增强来处理鼾声样本。最终数据集包含6000段鼾声样本和6000段非鼾声样本。  
处理好的数据集可以从该处下载：  

链接: https://pan.baidu.com/s/1QJzifB5Mde2iSc7GM9V8fg 提取码: 1234   

并以如下方式存放于项目中：  
____Snoring_Dataset  
｜__fea  
｜__orgin  
｜__resample_16k  


## 音频文件特征提取
### 训练集、测试集特征提取
运行脚本tools/fea_extra copy.py。
### 预测数据集特征提取
运行脚本tools/long_fea_extra_int8.py

## 开始训练
### 浮点训练
在main中设置训练模式为float，并运行脚本train.py
```
trained_net = train(train_loader, test_loader, test_dataset, mode = "float", load_model_path = None, num_epochs = 5)
```

### 约束训练
替换使用约束训练代码clamp，并运行脚本train.py
```
trained_net = train(train_loader, test_loader, test_dataset, mode = "clamp", load_model_path = "./tmp.ignore/snoring_net.float.best.pt", num_epochs =3)
```
### 量化训练
替换使用量化训练训练代码，并运行脚本train.py
```
trained_net = train(train_loader, test_loader, test_dataset, mode = "quant", load_model_path = "./tmp.ignore/snoring_net.clamp.best.pt", num_epochs = 3)
```
最终该脚本会在./tmp.ignore/文件夹下生成一个snoring_net.quant.onnx

### 模型打包
切换到thinker-env环境，使用thinker离线工具tpacker将刚才生成的onnx计算图打包
```
tpacker -g tmp.ignore/snoring_net.quant.onnx -d True -o snor_model_origin.bin
```
### 推理执行
使用调用示例工程test_thinker，指定输入数据、资源文件和输出文件名称即可运行模拟代码。
```
chmod +x ./bin/test_thinker  
./bin/test_thinker /data/user/mywang44/snoring_net/Snoring_Dataset_c/fea_int8/1 /data/user/mywang44/thinker/demo/test_thinker/ snor_model_origin.bin /data/user/mywang44/thinker/demo/test_thinker/output 1 64 64 6
```

## 模型评估
Accuracy: 0.986, F1: 0.986, Recall: 0.980
