import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import time
import math

def gaussian(x, y, sigma):
    return (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))


@cuda.jit
def gaussian_blur_kernel(input_image, output_image, kernel_size,sigma=2):
    # Get the 2D indices of the current pixel
    i, j = cuda.grid(2)

    # Allocate shared memory for the kernel weights
    kernel_weights = cuda.shared.array(shape=(kernel_size, kernel_size), dtype=float)

    # Initialize the kernel weights using a Gaussian function
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel_weights[x, y] = gaussian(x, y, sigma)

    # Perform the convolution
    for x in range(-kernel_size//2, kernel_size//2+1):
        for y in range(-kernel_size//2, kernel_size//2+1):
            output_image[i, j] += input_image[i+x, j+y] * kernel_weights[x, y]

def gaussian_blur(input_image, kernel_size=7):
    # Allocate memory on the GPU for the input and output images
    input_image_gpu = cuda.device_array(input_image.shape, dtype=input_image.dtype)
    output_image_gpu = cuda.device_array(input_image.shape, dtype=input_image.dtype)
    print(input_image)
    print(input_image_gpu)
    # Copy the input image to the GPU
    cuda.to_device(input_image, input_image_gpu)

    # Set the number of threads per block and the number of blocks
    threads_per_block = (16, 16)
    blocks_per_grid = (input_image.shape[0] // threads_per_block[0] + 1, input_image.shape[1] // threads_per_block[1] + 1)

    # Call the kernel function with the input image, output image, and kernel size as arguments
    gaussian_blur_kernel[blocks_per_grid, threads_per_block](input_image_gpu, output_image_gpu, kernel_size)

    # Copy the output image back to the CPU
    output_image = cuda.from_device(output_image_gpu, output_image.shape, dtype=output_image.dtype)

    return output_image


image=plt.imread('img_forest.jpeg')
start_time = time.time()
output=gaussian_blur(image)
print("--- GPU %s seconds ---" % (time.time() - start_time))
print(output)
