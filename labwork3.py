import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import time


def cpu_grayscale(image):
    gray = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
    return np.stack((gray, gray, gray), axis=2).astype(np.uint8)

def gpu_grayscale_kernel(image, result):
    height, width = image.shape
    row = cuda.grid(1)
    col = cuda.grid(2)
    if row < height and col < width:
        value = 0.2989 * image[row, col, 0] + 0.5870 * image[row, col, 1] + 0.1140 * image[row, col, 2]
        result[row, col] = value

def gpu_grayscale(image):
    height, width, _ = image.shape
    result = np.empty((height, width), dtype=np.float32)
    threads_per_block = (16, 16)
    blocks = (height // threads_per_block[0] + 1, width // threads_per_block[1] + 1)
    gpu_grayscale_kernel[blocks, threads_per_block](image, result)
    return result


if __name__=='__main__':
    image=plt.imread('img_forest.jpeg')
    start_time_cpu = time.time()
    cpu_output=cpu_grayscale(image)
    print(cpu_output)
    print("--- %s seconds ---" % (time.time() - start_time_cpu))
    start_time_gpu = time.time()
    gpu_output=gpu_grayscale(image)
    print(gpu_output)
    print("--- %s seconds ---" % (time.time() - start_time_gpu))