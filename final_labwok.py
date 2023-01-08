import numba
from numba import cuda
import numpy as np
import cv2
import math
import time 


@cuda.jit
def rgb_v_cuda(rgb_image, v_image):
    '''
    Kernel to get the V from an image by converting RGB to HSV GPU optimized. 
    H and S are not used so they are not calculated to avoid unnecessary calculation
    :rgb_image: RGB image input to convert to HSV
    :v_image: V image (from HSV) output 
    '''
    i, j = numba.cuda.grid(2)
    cmax = max(rgb_image[i][j][0], rgb_image[i][j][1], rgb_image[i][j][2]) /255
    v_image[i][j] = cmax * 100

def rgb_to_v(rgb_image):
    '''
    Function to call the kernel that will convert the input image from RGB to V (HSV)
    :rgb_image: RGB image input to convert to HSV
    :return: return the v matrix of the image
    '''
    v_image = np.empty((rgb_image.shape[0],rgb_image.shape[1]), dtype=rgb_image.dtype)
    block_dim = (32,32)
    grid_dim = (rgb_image.shape[0] // block_dim[0] + 1, rgb_image.shape[1] // block_dim[1] + 1)
    rgb_v_cuda[grid_dim, block_dim](rgb_image, v_image)
    return v_image

@cuda.jit
def kuwahara_kernel(input_rgb,output_rgb,image_v,window_size):
  '''
  Kernel to compute the kuwahara filter using RGB image input, V matrix from the HSV conversion and the window_size
  :rgb_image: RGB image input to filter
  :output_rgb: RGB image filtered
  :image_v: V matrix of the image (HSV conversion)
  :window_size: window size of the filter
  '''
  i, j = cuda.grid(2)
  #If the filter cannot be applied (out of the matrix +/- window size), the pixel is returned without modification and remains the same
  if i < window_size or i + window_size >= input_rgb.shape[0] or j < window_size or j + window_size >= input_rgb.shape[1]:
      return
  #4 possible windows : top left, bot left, top right, bot right
  current_windows = (((i-window_size, i+1), (j-window_size, j+1)),((i, i+window_size+1), (j-window_size, j+1)),
                ((i-window_size, i+1), (j, j+window_size+1)),((i, i+window_size+1), (j, j+window_size+1)))
  
  #window top left
  mean_window1=np.float32(0)
  len_window1=np.float32(0)
  for x in range(current_windows[0][0][0],current_windows[0][0][1]):
    for y in range(current_windows[0][1][0],current_windows[0][1][1]):
      mean_window1+=np.float32(image_v[x,y])
      len_window1+=1
  
  mean_window1/=len_window1
  std_window1=np.float32(0)

  for x in range(current_windows[0][0][0],current_windows[0][0][1]):
    for y in range(current_windows[0][1][0],current_windows[0][1][1]):
      std_window1+=(np.float32(image_v[x,y])-mean_window1)**2
  #All std replaced by the square substract of each v of the window by the v mean of the window
  #Every length will be the window size square so we can dispose of the calcul of the division by the lenght and the square root at the end
  #You can find the version with the real Std in the other script > old_script directory

  #window bot left
  std_window2=np.float32(0)

  mean_window2=np.float32(0)
  
  len_window2=np.float32(0)
  for x in range(current_windows[1][0][0],current_windows[1][0][1]):
    for y in range(current_windows[1][1][0],current_windows[1][1][1]):
      mean_window2+=np.float32(image_v[x,y])
      
      len_window2+=1
  
  mean_window2/=len_window2
  

  for x in range(current_windows[1][0][0],current_windows[1][0][1]):
    for y in range(current_windows[1][1][0],current_windows[1][1][1]):
      std_window2+=(np.float32(image_v[x,y])-mean_window2)**2
  
  
  #window top right
  std_window3=np.float32(0)

  mean_window3=np.float32(0)
  
  len_window3=np.float32(0)
  for x in range(current_windows[2][0][0],current_windows[2][0][1]):
    for y in range(current_windows[2][1][0],current_windows[2][1][1]):
      mean_window3+=np.float32(image_v[x,y])
      len_window3+=1
  
  mean_window3/=len_window3
  

  for x in range(current_windows[2][0][0],current_windows[2][0][1]):
    for y in range(current_windows[2][1][0],current_windows[2][1][1]):
      std_window3+=(np.float32(image_v[x,y])-mean_window3)**2
  
  #window bot right
  std_window4=np.float32(0)
  mean_window4=np.float32(0)
  len_window4=np.float32(0)
  for x in range(current_windows[3][0][0],current_windows[3][0][1]):
    for y in range(current_windows[3][1][0],current_windows[3][1][1]):
      mean_window4+=np.float32(image_v[x,y])
      len_window4+=1
  
  mean_window4/=len_window4

  for x in range(current_windows[3][0][0],current_windows[3][0][1]):
    for y in range(current_windows[3][1][0],current_windows[3][1][1]):
      std_window4+=(np.float32(image_v[x,y])-mean_window4)**2
  
  #Get the smallest std to define the new rgb pixel value of the whole window of pixel tested 
  #by taking the average of the rgb of the window that has the smallest std
  smallest_std=min(std_window1,std_window2,std_window3,std_window4)

  new_r=0
  new_g=0
  new_b=0

  if smallest_std==std_window1:
    for x in range(current_windows[0][0][0],current_windows[0][0][1]):
      for y in range(current_windows[0][1][0],current_windows[0][1][1]):
        
        new_r+=input_rgb[x,y,0]
        new_g+=input_rgb[x,y,1]      
        new_b+=input_rgb[x,y,2]

    new_r/=len_window1
    new_g/=len_window1
    new_b/=len_window1
    

  elif smallest_std==std_window2:
    for x in range(current_windows[1][0][0],current_windows[1][0][1]):
      for y in range(current_windows[1][1][0],current_windows[1][1][1]):
        
        new_r+=input_rgb[x,y,0]
        new_g+=input_rgb[x,y,1]      
        new_b+=input_rgb[x,y,2]
        
    new_r/=len_window2
    new_g/=len_window2
    new_b/=len_window2
  
  elif smallest_std==std_window3:
    
    for x in range(current_windows[2][0][0],current_windows[2][0][1]):
      for y in range(current_windows[2][1][0],current_windows[2][1][1]):
        
        new_r+=input_rgb[x,y,0]
        new_g+=input_rgb[x,y,1]      
        new_b+=input_rgb[x,y,2]
        
    new_r/=len_window3
    new_g/=len_window3
    new_b/=len_window3
  
  else:
    for x in range(current_windows[3][0][0],current_windows[3][0][1]):
      for y in range(current_windows[3][1][0],current_windows[3][1][1]):
        
        new_r+=input_rgb[x,y,0]
        new_g+=input_rgb[x,y,1]      
        new_b+=input_rgb[x,y,2]
        
    new_r/=len_window4
    new_g/=len_window4
    new_b/=len_window4

  #Apply the new value to the matrix in the memory
  output_rgb[i][j][0]=new_r
  output_rgb[i][j][1]=new_g
  output_rgb[i][j][2]=new_b



def kuwahara_filter(rgb_image,output_rgb,v_image):
    '''
    Function to call the kuwahara filter kernel
    :rgb_image: RGB image input to filter
    :output_rgb: RGB image output kuwahara filtered
    :v_image: V matrix of the image (from the HSV conversion)
    '''
    window_size=3
    block_dim = (8,8)
    grid_dim = (rgb_image.shape[0] // block_dim[0] + 1, rgb_image.shape[1] // block_dim[1] + 1)
    kuwahara_kernel[grid_dim, block_dim](rgb_image, output_rgb,v_image,window_size)
    return output_rgb

start_time = time.time()
#Load the testing image
rgb_image = cv2.imread("img_forest.jpeg")
#Get the V matrix of the RGB image
v_image=rgb_to_v(rgb_image)
#Create the output image
output_rgb = np.copy(rgb_image)
#Apply the Kuwahara filter to the output image
output_rgb=kuwahara_filter(rgb_image,output_rgb,v_image)
#write the filtered image
cv2.imwrite("output.png", output_rgb)
print("--- %s seconds to filter the image using the Kuwahara filter---" % (time.time() - start_time))