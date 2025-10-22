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


if __name__ == "__main__":
    stack = tif.imread("E:/532_Paper/Imaging/Zoomed_FLIM/Euglena/FLIM_STACK_NO_1.tif")
    start = 9
    stack = stack[start:,:,:].astype(np.float32)
    x_data = np.linspace(0, 6.82, 23, dtype=np.float32)
    tau_image = np.zeros((512,512), dtype=np.float32)
    res_image = np.zeros((23, 512,512), dtype=np.float32)
    min_tau = 0.5
    max_tau = 2
    pos_taus = np.linspace(min_tau, max_tau, 1000, dtype=np.float32)

    image_gpu = cuda.to_device(stack)
    diluted_image_gpu = cuda.to_device(stack)
    tau_gpu = cuda.to_device(tau_image)
    pos_taus_gpu = cuda.to_device(pos_taus)
    x_data_gpu = cuda.to_device(x_data)


    block_dilution = (512)
    threads_dilution = (512)
    block_fit = (512)
    threads_fit = (512)

    mean_cuda_2D[block_dilution, threads_dilution](diluted_image_gpu, image_gpu, 2, 0)
    flim_fit_2D[block_fit, threads_fit](tau_gpu, diluted_image_gpu, pos_taus_gpu, x_data_gpu, 10)
    t = time.time()
    
    print((time.time()-t)/1000)
    tau_gpu.copy_to_host(tau_image)
    diluted_image_gpu.copy_to_host(res_image)

    FLIM = flim_intensity_to_rgb(tau_image, res_image.mean(axis=0), flim_min=0.3, flim_max=1, gamma=1)

    viewer = napari.Viewer()
    viewer.add_image(tau_image, name = "Tau")
    viewer.add_image(res_image, name="Image")
    viewer.add_image(FLIM, name="FLIM")
    napari.run()

