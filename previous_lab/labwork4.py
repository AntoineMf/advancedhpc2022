import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import time


def cpu_grayscale(image):
    # Check if image is already grayscale
    if len(image.shape) == 2:
        return image

    # Convert image to grayscale using luminosity formula
    return 0.3 * image[:,:,0] + 0.59 * image[:,:,1] + 0.11 * image[:,:,2]

@cuda.jit
def gpu_grayscale_kernel(image, result):
    height = image.shape[0]
    width = image.shape[1]
    i, j = cuda.grid(2)
    if i >= height or j >= width:
        return

    result[i, j] = (image[i, j, 0] * 0.3 + image[i, j, 1] * 0.59 + image[i, j, 2] * 0.11)

def gpu_grayscale(image):
    height, width, _ = image.shape
    result = np.empty((height, width), dtype=np.float32)
    threads_per_block = (16, 16)
    blocks = (height // threads_per_block[0] + 1, width // threads_per_block[1] + 1)
    start_time_gpu = time.time()
    gpu_grayscale_kernel[blocks, threads_per_block](image.copy(), result)
    print("--- GPU %s seconds ---" % (time.time() - start_time_gpu))
    return result


image=plt.imread('big_image.jpg')
start_time_cpu = time.time()
cpu_output=cpu_grayscale(image)
print(cpu_output)
print("--- CPU %s seconds ---" % (time.time() - start_time_cpu))

gpu_output=gpu_grayscale(image)
print(gpu_output)