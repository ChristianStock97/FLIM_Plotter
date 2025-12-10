import numpy as np
import tiffile as tif
import matplotlib.pyplot as plt
import napari
from numba import cuda
import time

@cuda.jit
def _convert_to_image_kernel(data, time_stack, Interleaving,
                             width, hight, SamplePerPixel, peak_offset):
    """
    CUDA kernel version of convert_to_image.

    data: 1D device array
    time_stack: 3D device array (SamplePerPixel, hight, width)
    """

    tid = cuda.grid(1)
    total_pixels = hight * width
    if tid >= total_pixels:
        return

    # Decode 1D thread index -> (n, m)
    n = tid // width
    m = tid % width

    # Determine interleaving group for this column
    l = m % Interleaving
    k = (m - l) // Interleaving  # position index inside this l-loop

    # Compute size_per_line (how many data elements per line n)
    size_per_line = 0
    for l2 in range(Interleaving):
        # how many m's for this l2?
        # ceil((width - l2)/Interleaving) but clamped at ≥0
        remaining = width - l2
        if remaining > 0:
            count_l2 = (remaining + Interleaving - 1) // Interleaving
        else:
            count_l2 = 0
        size_per_line += count_l2 * SamplePerPixel + 4  # +4 padding per l2

    # Start of this line's data in the flat input
    pos_line_start = n * size_per_line

    # Offset within the line from all previous l2 < l
    offset_l = 0
    for l2 in range(l):
        remaining = width - l2
        if remaining > 0:
            count_l2 = (remaining + Interleaving - 1) // Interleaving
        else:
            count_l2 = 0
        offset_l += count_l2 * SamplePerPixel + 4

    pos_l_start = pos_line_start + offset_l

    # Offset from the start of this l-block to this particular (n, m)
    pos_pixel = pos_l_start + k * SamplePerPixel

    # Now copy SamplePerPixel values into time_stack[:, n, m]
    p = 0
    for s in range(peak_offset,SamplePerPixel):
        time_stack[p, n, m] = data[pos_pixel + s]
        p += 1

def convert_to_image_cuda(path, Interleaving,num_frames=1, sample_offset=0, peak_offset=0):
    """
    Host-side wrapper that behaves like your original convert_to_image,
    but executes the core filling loop on the GPU.
    """
    SamplePerPixel = 4 * Interleaving
    SamplePerImage =  SamplePerPixel * 512 * 512
    #data = load_data(path, length, sample_offset)
    #num_frames = int(len(data) / (4 * Interleaving * 512 * 512))
    
    hight=512 
    width=511
    threads_per_block=512
    # Allocate output on host
    time_stack_host = np.zeros((num_frames, SamplePerPixel-peak_offset, hight, width), dtype=np.uint8)

    # Move data to GPU
    data_dev = cuda.to_device(np.zeros(SamplePerImage, dtype=np.uint8))
    time_stack_dev = cuda.to_device(np.zeros((SamplePerPixel-peak_offset, hight, width), dtype=np.uint8))

    # Launch kernel
    total_pixels = hight * width
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    with open(path, 'rb') as file: 
        file.seek(sample_offset+3070)
        for k in range(num_frames):
            data = file.read(SamplePerImage)
            ret_data = np.frombuffer(data, dtype=np.uint8)
            if (ret_data.size == SamplePerImage):
                cuda.to_device(ret_data, to=data_dev)
                _convert_to_image_kernel[blocks_per_grid, threads_per_block](
                    data_dev, time_stack_dev,
                    Interleaving, width, hight, SamplePerPixel, peak_offset
                )
                # Copy result back
                time_stack_dev.copy_to_host(time_stack_host[k])
                
    
    return time_stack_host

def load_data(file_path, length, offset):
    ret_data = np.zeros(length, dtype=np.uint8)
    with open(file_path, 'rb') as file:
        file.seek(offset+3070)
        data = file.read(length)
        ret_data = np.frombuffer(data, dtype=np.uint8)
    return ret_data


if __name__ == "__main__":
    
    Interleaving = 8
    num_frames = 100
    sample_offset = 6668

    path = "E:/FLIM_Messungen/1/x8"
    name = "2025.11.10_15.37.27.atb"
    file_path = path + "/" + name
    peak_offset = 7
    t = time.time()
    time_stack = convert_to_image_cuda(file_path, Interleaving, num_frames, sample_offset, peak_offset)
    t = time.time() - t
    print(f"CUDA conversion time for {num_frames} frames: {t:.2f} seconds")

    v = napari.Viewer()
    v.add_image(time_stack, name='FLIM Image', colormap='gray', contrast_limits=[0, 255])
    napari.run()