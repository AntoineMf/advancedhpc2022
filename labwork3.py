import matplotlib.pyplot as plt
import numpy as np

image=plt.imread('img_forest.jpeg')
gray_image = np.mean(image.reshape(-1, 3), axis=1).reshape(image.shape[0], image.shape[1])
print(gray_image)