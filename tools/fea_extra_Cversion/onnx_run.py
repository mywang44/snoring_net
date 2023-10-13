# -*- coding: utf-8 -*-

import warnings

import numpy as np

warnings.filterwarnings(action="ignore")

import librosa
import onnxruntime as ort

def write_mel_weights():
    mel_matrix = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64, htk=False, fmin=20, fmax=8000)
    mel_list = []

    with open('mel.txt', 'w') as f:
        for i in range(mel_matrix.shape[0]):
            for j in range(mel_matrix.shape[1]):
                if mel_matrix[i][j] > 0:
                    mel_list.append([int(i), int(j), mel_matrix[i][j]])
                    f.write('{')
                    f.write(str(i))
                    f.write(',')
                    f.write(str(j))
                    f.write(',')
                    f.write(str(mel_matrix[i][j]))
                    f.write('}')
                    f.write(',')


def write_window():
    import scipy.signal
    weight_matrix = scipy.signal.get_window('hann', 1024, fftbins=True)

    with open('window.txt', 'w') as f:
        f.write('{')
        for i in range(weight_matrix.shape[0]):
            f.write(str(weight_matrix[i]))
            f.write(',')
        f.write('}')


def fea_extra(wav=None, sr=None, n_fft=1024, hop_length=512, n_mels=64, fmin=20, fmax=8000, top_db=80, eps=1e-6):
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin,
                                          fmax=fmax)

    # stft_result = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=True)
    # stft_result = np.abs(stft_result) ** 2
    # mel_data = librosa.filters.mel(sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # spec = np.dot(mel_data, stft_result)

    spec_db = librosa.power_to_db(spec, top_db=top_db)
    mean = spec_db.mean()
    std = spec_db.std()
    spec_norm = (spec_db - mean) / (std + eps)
    print(mean,std)
    return spec_norm

if __name__ == '__main__':
    # # 初始化结果变量
    # result = []
    #
    # # 遍历数组，每隔两个元素相加
    # for i in range(0, len(samples), 2):
    #     sum_of_three = sum(samples[i:i + 2])  # 对每隔两个元素进行相加
    #     result.append(sum_of_three)

    # 加载原始音频
    target_sr = 16000
    wav_path = 'resources/0_198.wav'
    y, origin_sr = librosa.load(wav_path, sr=None)
    y_16k = librosa.resample(y, orig_sr=origin_sr, target_sr=target_sr)
    if y_16k.shape[0] < 2 * target_sr:
        y_16k = np.pad(y_16k, int(np.ceil((2.04 * target_sr - y_16k.shape[0]) / 2)), mode='reflect')
    else:
        y_16k = y_16k[:2.04 * target_sr]
    spec_norm = fea_extra(y_16k, target_sr)

    # 加载C数据
    eps = 1e-6
    mel_result = []
    with open('output.txt', 'r') as f:
        for line in f.readlines():
            mel_result.append([float(l) for l in line.split()])
    c_spec_db = np.array(mel_result,dtype=np.float32)
    # mean = c_spec_db.mean()
    # std = c_spec_db.std()
    # c_spec_norm = (c_spec_db - mean) / (std + eps)
    c_spec_norm = c_spec_db.T

    # 加载ONNX模型
    snor_model = ort.InferenceSession("resources/snoring_net.float.onnx")

    # 执行推理
    model_output1 = snor_model.run(None, {"input": spec_norm[:, :64][np.newaxis, np.newaxis, :, :]})
    model_output2 = snor_model.run(None, {"input": c_spec_norm[:, :64][np.newaxis, np.newaxis, :, :]})

    model_output2 = model_output2[0][0]
    print(model_output1[0][0])
    print(model_output2)
    #
    # audioExt = 'WAV'
    # input_sample = None
    # output_sample = 16000
    # audioDirectory = ['/data/user/mywang44/snoring_net/Snoring_Dataset/orgin/1',
    #                   '/data/user/mywang44/snoring_net/Snoring_Dataset/orgin/0']
    # audio_outputDirectory = ['/data/user/mywang44/snoring_net/Snoring_Dataset/resample_16k/1',
    #                          '/data/user/mywang44/snoring_net/Snoring_Dataset/resample_16k/0']
    # fea_outputDirectory = ['/data/user/mywang44/snoring_net/Snoring_Dataset/fea/1',
    #                        '/data/user/mywang44/snoring_net/Snoring_Dataset/fea/0']
    #
    # for i, dire in enumerate(audioDirectory):
    #     clean_speech_paths = librosa.util.find_files(
    #         directory=dire,
    #         ext=audioExt,
    #         recurse=True,
    #     )
    #
    #     for file in tqdm.tqdm(clean_speech_paths, desc='No.{} dataset resampling'.format(i)):
    #         fileName = os.path.basename(file)
    #         y, sr = librosa.load(file, sr=input_sample)
    #
    #         y_16k = librosa.resample(y, orig_sr=sr, target_sr=output_sample)
    #         if y_16k.shape[0] < 2 * output_sample:
    #             y_16k = np.pad(y_16k, int(np.ceil((2.04 * output_sample - y_16k.shape[0]) / 2)), mode='reflect')
    #         else:
    #             y_16k = y_16k[:2.04 * output_sample]
    #
    #         audio_outputFileName = os.path.join(audio_outputDirectory[i], fileName)
    #         sf.write(audio_outputFileName, y_16k, output_sample)
    #
    #         fea = fea_extra(wav=y_16k, sr=output_sample)
    #         fea_fileName = fileName.split('.')[0] + ".fea"
    #         fea_outputFileName = os.path.join(fea_outputDirectory[i], fea_fileName)
    #         with open(fea_outputFileName, 'wb') as f:
    #             pickle.dump(fea, f)
    #             # f.write(str(fea))