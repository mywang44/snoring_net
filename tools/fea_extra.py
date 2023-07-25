import numpy as np
import librosa
import warnings
warnings.filterwarnings(action="ignore")

import os
import librosa
import tqdm
import soundfile as sf
import pickle

def fea_extra(file_path=None, wav=None, sr=None, n_fft=1024, hop_length=512, n_mels=64, fmin=20, fmax=8000, top_db=80, eps=1e-6):
    #wav, sr = librosa.load(file_path,sr=sr)
    if wav.shape[0] < 2*sr:
        wav = np.pad(wav, int(np.ceil((2.04*sr-wav.shape[0])/2)), mode='reflect')
    else:
        wav = wav[:2.04*sr]

    spec    = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec,top_db=top_db)
    mean = spec_db.mean()
    std  = spec_db.std()
    spec_norm = (spec_db - mean) / (std + eps)
    # spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
    # spec_scaled = spec_scaled.astype(np.uint8)

    return spec_norm


if __name__ == '__main__':

    audioExt = 'WAV'
    input_sample = None
    output_sample = 16000
    audioDirectory  = ['/home/nizai8a/snoring_net/Snoring_Dataset/orgin/1', '/home/nizai8a/snoring_net/Snoring_Dataset/orgin/0']
    audio_outputDirectory = ['/home/nizai8a/snoring_net/Snoring_Dataset/resample_16k/1', '/home/nizai8a/snoring_net/Snoring_Dataset/resample_16k/0']
    fea_outputDirectory = ['/home/nizai8a/snoring_net/Snoring_Dataset/fea/1', '/home/nizai8a/snoring_net/Snoring_Dataset/fea/0']

	
    for i, dire in enumerate(audioDirectory):
        clean_speech_paths = librosa.util.find_files(
                directory=dire,
                ext=audioExt,
                recurse=True, 
            )

        for file in tqdm.tqdm(clean_speech_paths, desc='No.{} dataset resampling'.format(i)):
            fileName = os.path.basename(file)
            y, sr = librosa.load(file, sr=input_sample)

            y_16k = librosa.resample(y, orig_sr=sr, target_sr=output_sample)
            audio_outputFileName = os.path.join(audio_outputDirectory[i], fileName)
            sf.write(audio_outputFileName, y_16k, output_sample)

            fea = fea_extra(wav=y_16k, sr=output_sample)
            fea_fileName = fileName.split('.')[0]+".fea"
            fea_outputFileName = os.path.join(fea_outputDirectory[i], fea_fileName)
            with open(fea_outputFileName,'wb') as f:
                pickle.dump(fea, f)
                # f.write(str(fea))



#项目原本的前处理脚本，可以生成fea文件用于训练。
#最后选择此脚本进行训练前处理