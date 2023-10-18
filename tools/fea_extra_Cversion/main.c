
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION

#include "dr_mp3.h"

#include "heads/Params.h"
#include "./kiss_fft130/kiss_fft.h"

float *wavRead_f32(const char *filename, uint32_t *sampleRate, uint64_t *sampleCount, uint32_t *channels) {
    drwav_uint64 totalSampleCount = 0;
    float *input = drwav_open_file_and_read_pcm_frames_f32(filename, channels, sampleRate, &totalSampleCount);
    if (input == NULL) {
        drmp3_config pConfig;
        input = drmp3_open_file_and_read_f32(filename, &pConfig, &totalSampleCount);
        if (input != NULL) {
            *channels = pConfig.outputChannels;
            *sampleRate = pConfig.outputSampleRate;
        }
    }
    if (input == NULL) {
        fprintf(stderr, "read file [%s] error.\n", filename);
        exit(1);
    }
    *sampleCount = totalSampleCount * (*channels);
    return input;
}

uint64_t Resample_f32(const float *input, float *output, int inSampleRate, int outSampleRate, uint64_t inputSize,
                      uint32_t channels
) {
    if (input == NULL)
        return 0;
    uint64_t outputSize = (uint64_t) (inputSize * (double) outSampleRate / (double) inSampleRate);
    outputSize -= outputSize % channels;
    if (output == NULL)
        return outputSize;
    double stepDist = ((double) inSampleRate / (double) outSampleRate);
    const uint64_t fixedFraction = (1LL << 32);
    const double normFixed = (1.0 / (1LL << 32));
    uint64_t step = ((uint64_t) (stepDist * fixedFraction + 0.5));
    uint64_t curOffset = 0;
    for (uint32_t i = 0; i < outputSize; i += 1) {
        for (uint32_t c = 0; c < channels; c += 1) {
            *output++ = (float) (input[c] + (input[c + channels] - input[c]) * (
                    (double) (curOffset >> 32) + ((curOffset & (fixedFraction - 1)) * normFixed)
            )
            );
        }
        curOffset += step;
        input += (curOffset >> 32) * channels;
        curOffset &= (fixedFraction - 1);
    }
    return outputSize;
}


// 函数用于执行对称补齐操作
void reflect_pad(float *input, int input_size, float *output, int output_size, int pad_width) {
    int i, j;

    // 复制原始数据到输出数组的中间部分
    for (i = 0; i < input_size; i++) {
        output[pad_width + i] = input[i];
    }

    // 补齐左边
    for (i = 0; i < pad_width; i++) {
        output[i] = input[pad_width - i];
    }

    // 补齐右边
    for (i = 0; i < pad_width; i++) {
        output[output_size - 1 - i] = input[input_size - 1 - (pad_width - i)];
    }
}

// 函数用于按照窗口大小w和步长s从数组data中读取数据
void readDataInWindow(float *data, int n, int w, int s) {
    for (int i = 0; i <= n - w; i += s) {
        float tmpData[2048];
        // 从下标i开始读取窗口大小w的数据
        printf("Window %d-%d: ", i, i + w - 1);
        int tmpI = 0;
        for (int j = i; j < i + w; j++) {
            tmpData[tmpI] = data[j];
            printf("%.10f", data[j]);
            tmpI++;
        }
        printf("\n");
    }
}

void fft(kiss_fft_cfg fft_tool, float *audio, float *fft_value) {
    kiss_fft_cpx fftIn[window_len];
    kiss_fft_cpx fftOut[window_len];

    int nFreq = 1 + window_len / 2;

    for (int i = 0; i < window_len; i++) {
        fftIn[i].r = WindowData[i] * audio[i];
        fftIn[i].i = 0.0;
    }

    kiss_fft(fft_tool, fftIn, fftOut);

    for (int i = 0; i < nFreq; i++) {
        double real = fftOut[i].r;
        double imag = fftOut[i].i;
        double complex_result = sqrt(real * real + imag * imag);
        complex_result = complex_result * complex_result;
        fft_value[i] = complex_result;
    }
}

void calc_feature(float *m_fft_vec, float *feature_out) {
    int r = 0;
    double sum = 0;

    int numElements = sizeof(dataTable) / sizeof(dataTable[0]);

    // 遍历结构体数组并打印数据
    for (int i = 0; i < numElements; i++) {
        if (r != dataTable[i].i) {
            feature_out[r] = sum;

            r = dataTable[i].i;
            sum = 0;
        }
        sum += dataTable[i].value * m_fft_vec[dataTable[i].j];
    }
    feature_out[r] = sum;

}

// 函数用于将一维浮点数数组保存到文本文件
void saveArrayToFile(float array[], int length, const char *filename) {
    FILE *file = fopen(filename, "a"); // "a" 表示以追加模式打开文件

    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
        return;
    }

    for (int i = 0; i < length; i++) {
        fprintf(file, "%.2f ", array[i]); // 以两位小数的格式写入文件
    }

    fprintf(file, "\n"); // 写入换行符表示一维数组结束

    fclose(file); // 关闭文件
}

// 函数用于将一维浮点数数组保存到文本文件
void saveArrayToFileInt(int8_t array[], int length, const char *filename) {
    FILE *file = fopen(filename, "a"); // "a" 表示以追加模式打开文件

    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
        return;
    }

    for (int i = 0; i < length; i++) {
        fprintf(file, "%d ", array[i]); // 以两位小数的格式写入文件
    }

    fprintf(file, "\n"); // 写入换行符表示一维数组结束

    fclose(file); // 关闭文件
}

int main() {
    char *in_file = "0_111.wav";
    uint32_t targetSampleRate = 16000;

    uint32_t sampleRate = 0;
    uint64_t sampleCount = 0;
    uint32_t channels = 0;
    float *input = wavRead_f32(in_file, &sampleRate, &sampleCount, &channels);
    uint64_t targetSampleCount = Resample_f32(input, NULL, sampleRate, targetSampleRate, sampleCount, channels);
    if (input) {
        float *output = (float *) malloc(targetSampleCount * sizeof(float));
        if (output) {
            Resample_f32(input, output, sampleRate, targetSampleRate, sampleCount / channels, channels);

            float *data = (float *) malloc(targetSampleRate / channels * sizeof(float)); // 分配数组内存

            for (int i = 0; i < targetSampleCount; i = i + 2) {
                float curr = *(output + i);
                float next = *(output + i + 1);
                data[i / channels] = (curr + next) / 2;
            }
            free(output);

            // pad补齐
            int pad_width = ceil((2.04 * targetSampleRate - (float) targetSampleCount / (float) channels) / 2);
            int input_size = targetSampleCount / channels;
            int output_size = input_size + 2 * pad_width;
            float *padOut = (float *) malloc(output_size * sizeof(float));
            reflect_pad(data, input_size, padOut, output_size, pad_width);
            free(data);

            // 特征提取
            int window_size = 1024; // 窗口大小
            int hop_size = 512; // 步长
            int n_fft = 1024;
            pad_width = n_fft / 2;
            int output_size_2 = output_size + 2 * pad_width;
            float *padOut2 = (float *) malloc(output_size_2 * sizeof(float));
            reflect_pad(padOut, output_size, padOut2, output_size_2, pad_width);
            free(padOut);

            kiss_fft_cfg fft_tool = kiss_fft_alloc(window_len, 0, 0, 0);
            KISS_FFT_FREE(fft_tool);

            const char *out_filename = "c_output_float.txt";
            const char *out_filename2 = "c_output_int.txt";
            int frameSize = output_size_2 / hop_size - 1;
            int frameIndex = 0;
            float melResult[frameSize][mel_bins];

            for (int sample_index = 0; sample_index <= output_size_2 - window_size; sample_index += hop_size) {
                float tmpData[window_size];
                // 从下标i开始读取窗口大小w的数据
                int tmpI = 0;
                for (int j = sample_index; j < sample_index + window_size; j++) {
                    tmpData[tmpI] = padOut2[j];
                    tmpI++;
                }

                int fft_size = 1 + window_len / 2;
                float fft_value[fft_size];
//                printf("开始 Window %d-%d: \n", sample_index, sample_index + window_size - 1);
                fft(fft_tool, tmpData, fft_value);

                float feature_out[mel_bins];
                float spec[mel_bins];

                calc_feature(fft_value, feature_out);

                float log_max = 1e-10f;
                for (int mel_index = 0; mel_index < mel_bins; mel_index++) {
                    float v = feature_out[mel_index];
                    if (log_max > v) {
                        v = log_max;
                    }
                    spec[mel_index] = 10 * log10f(v);
                    melResult[frameIndex][mel_index] = 10 * log10f(v);
                }

                frameIndex++;
            }
            // 计算均值方差
            float mean = 0;
            for (int ii = 0; ii < frameSize; ii++) {
                for (int jj = 0; jj < mel_bins; jj++) {
                    mean += melResult[ii][jj];
                }
            }
            mean = mean / (float) (frameSize * mel_bins);

            float std = 0;
            for (int ii = 0; ii < frameSize; ii++) {
                for (int jj = 0; jj < mel_bins; jj++) {
                    std += powf(melResult[ii][jj] - mean, 2);
                }
            }
            std = sqrtf(std / (float) (frameSize * mel_bins));

            float melResultNorm[frameSize][mel_bins];
            int8_t melResultNormInt[frameSize][mel_bins];

            float maxAbsValue = 0.0f;
            for (int ii = 0; ii < frameSize; ii++) {
                for (int jj = 0; jj < mel_bins; jj++) {
                    float melNorm = (melResult[ii][jj] - mean) / (std + 1e-6f);
                    melResultNorm[ii][jj] = melNorm;

                    float absValue = fabsf(melNorm);
                    if (absValue > maxAbsValue) {
                        maxAbsValue = absValue;
                    }
//                    // 转为int8
//                    melNorm = floorf(melNorm * 64 + 0.5);
//                    int8_t int8Value = (int8_t) (int) melNorm;
//                    melResultNormInt[ii][jj] = int8Value;

                }
            }
            printf("%f %f", mean, std);

            for (int ii = 0; ii < frameSize; ii++) {
                for (int jj = 0; jj < mel_bins; jj++) {
                    float melNorm = melResultNorm[ii][jj];
                    melNorm = melNorm / maxAbsValue;
                    melNorm = melNorm * 127;
                    int8_t int8Value = (int8_t) (int) melNorm;
                    melResultNormInt[ii][jj] = int8Value;

                }
            }
            for (int ii = 0; ii < frameSize; ii++) {
                saveArrayToFile(melResultNorm[ii], sizeof(melResultNorm[ii]) / sizeof(melResultNorm[ii][0]),
                                out_filename);
                saveArrayToFileInt(melResultNormInt[ii], sizeof(melResultNormInt[ii]) / sizeof(melResultNormInt[ii][0]),
                                   out_filename2);
            }
        }
    }
    free(input);
    return 0;


}

#ifdef __cplusplus
}
#endif
