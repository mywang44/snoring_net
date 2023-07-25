import os
import numpy as np
import librosa
import torch
import soundfile as sf
import numpy as np
import librosa
import numpy as np
import librosa
import warnings
warnings.filterwarnings(action="ignore")

def fea_extra(wav=None, sr=None, n_fft=1024, hop_length=512, n_mels=64, fmin=20, fmax=8000, top_db=80, eps=1e-6):
    if wav.shape[0] < 2*sr:
        wav = np.pad(wav, int(np.ceil((2.04*sr-wav.shape[0])/2)), mode='reflect')
    else:
        wav = wav[:2.04*sr]

    spec    = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec,top_db=top_db)
    mean = spec_db.mean()
    std  = spec_db.std()
    spec_norm = (spec_db - mean) / (std + eps)

    return spec_norm

if __name__ == '__main__':
    input_sample = None
    output_sample = 16000
    audioFile = '/data/user/mywang44/snoring_net/predicate_data/Snoring.wav'
    audio_outputDirectory = '/data/user/mywang44/snoring_net/Snoring_Dataset/resample_16k'
    fea_int8_outputDirectory = '/data/user/mywang44/snoring_net/Snoring_Dataset/fea_int8'

    y, sr = librosa.load(audioFile, sr=input_sample)
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=output_sample)

    for idx in range(0, len(y_16k), output_sample):
        y_16k_segment = y_16k[idx:idx+output_sample]
        if len(y_16k_segment) < output_sample:
            break
        audio_outputFileName = os.path.join(audio_outputDirectory, f'audio_segment_{idx//output_sample}.wav')
        sf.write(audio_outputFileName, y_16k_segment, output_sample)

        fea = fea_extra(wav=y_16k_segment, sr=output_sample)

        # Convert the float feature to int8
        fea_int8 = np.int8(fea / np.max(np.abs(fea)) * 127)
        fea_int8_fileName = f'fea_int8_segment_{idx//output_sample}.bin'
        fea_int8_outputFileName = os.path.join(fea_int8_outputDirectory, fea_int8_fileName)
        fea_int8.tofile(fea_int8_outputFileName)




#用于推理执行部分的前处理，将一段长音频分割 并做和训练前处理相同的操作