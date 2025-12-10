from numba import cuda
import numpy as np
import tiffile as tif
import matplotlib.pyplot as plt
import math
import napari
import time
from TO_HSV import flim_intensity_to_rgb

@cuda.jit
def flim_fit_2D(output, image, pos_taus, x_data):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x
    N, H, W = image.shape
    data = image[:,y,x]
    A = data[0]
    C = data[-1]
    best_tau = pos_taus[0]
    min_dif = np.inf
    
    for t in range(pos_taus.size):
        tau = pos_taus[t]
        sum_dif = 0.0
        for n in range(N):
            y_fit = A * math.exp(-x_data[n] / tau) + C
            sum_dif += abs(data[n] - y_fit)
        if sum_dif < min_dif:
            best_tau = tau
            min_dif = sum_dif
    if best_tau in [pos_taus[0], pos_taus[-1]]:
        best_tau = 0
    output[y, x] = best_tau
    
@cuda.jit
def dilate_cuda_2D(output, input, size):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x
    y_dim = cuda.gridDim.x
    x_dim = cuda.blockDim.x
    Image, Sample, H, W = input.shape
    for i in range(Image):
        for z in range(Sample):
            sum_value = float(0.0)
            for n in range(max(y-size, 0), min(y+size+1,y_dim),1):
                for m in range(max(x-size, 0), min(x+size+1,x_dim),1):
                    sum_value += float(input[i, z, n, m])
    output[z, y, x] = sum_value

@cuda.jit
def mean_cuda_2D(output, input):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x
    y_dim = cuda.gridDim.x
    x_dim = cuda.blockDim.x
    _, N, H, W = input.shape
    sum_value = 0.0
    for z in range(N):
        sum_value += input[0, z, y, x]
    output[y, x] = sum_value / N
    
@cuda.jit
def dilation_cuda_3D(output, input, size, threshold):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x
    y_dim = cuda.gridDim.x
    x_dim = cuda.blockDim.x
    frames, N, H, W = input.shape
    for z in range(N):
        sum_value = 0.0
        for n in range(max(y-size, 0), min(y+size+1,y_dim),1):
            for m in range(max(x-size, 0), min(x+size+1,x_dim),1):
                for f in range(frames):
                    sum_value += float(input[f, z, n, m])
        if sum_value > threshold:
            output[z, y, x] = sum_value
        else:
            output[z, y, x] = 0

@cuda.jit
def mean_cuda_3D(output, input):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x
    y_dim = cuda.gridDim.x
    x_dim = cuda.blockDim.x
    frames, N, H, W = input.shape
    sum_value = 0.0
    for f in range(frames):
        for z in range(N):
            sum_value += input[f, z, y, x]
    output[y, x] = sum_value / (N*frames)

