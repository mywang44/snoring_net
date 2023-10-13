# 动态cmake
cmake_minimum_required(VERSION 3.17) # CMake所需的最低版本
project(PianoInfer) # 项目的名称和支持的编程语言

set(CMAKE_CXX_STANDARD 14)

# 可执行文件，编译和链接源文件
add_executable(PianoInfer main.c kiss_fft130/kiss_fft.c dr_mp3.h dr_wav.h)