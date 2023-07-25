import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005*noise
    return data_noise

def change_pitch(data, sample_rate):
    return librosa.effects.pitch_shift(data, sample_rate, np.random.randint(-5, 5))

def change_speed(data):
    speed_change = np.random.uniform(low=0.9,high=1.1)
    return librosa.effects.time_stretch(data, speed_change)

def augment_data(data, sr):
    # 对数据进行增强
    data_noise = add_noise(data)
    data_pitch = change_pitch(data, sr)
    data_speed = change_speed(data)

    return data_noise, data_pitch, data_speed

def main():
    # 待处理音频的文件夹
    audioDirectory  = '/home/nizai8a/snoring_net/Snoring_Dataset/orgin/1'
    # 输出文件夹
    outputDirectory = '/home/nizai8a/snoring_net/Snoring_Dataset/orgin/1_augment'

    # 寻找"audioDirectory"文件夹中的音频文件，返回值为绝对路径的列表类型
    clean_speech_paths = librosa.util.find_files(
        directory=audioDirectory,
        ext=['wav'],
        recurse=True, # 如果选择True，则对输入文件夹的子文件夹也进行搜索，否则只搜索输入文件夹
    )

    # 遍历所有音频文件
    for file in tqdm(clean_speech_paths, desc='Augmenting dataset'):
        # 获取音频文件的文件名，用作输出文件名使用
        fileName = os.path.basename(file)
        baseName, _ = os.path.splitext(fileName)

        # 使用librosa读取待处理音频
        y, sr = librosa.load(file, sr=None)

        # 对音频文件进行增强
        y_noise, y_pitch, y_speed = augment_data(y, sr)

        # 将增强后的数据保存到硬盘
        sf.write(os.path.join(outputDirectory, baseName + '_noise.wav'), y_noise, sr)
        sf.write(os.path.join(outputDirectory, baseName + '_pitch.wav'), y_pitch, sr)
        sf.write(os.path.join(outputDirectory, baseName + '_speed.wav'), y_speed, sr)

if __name__ == '__main__':
    main()
