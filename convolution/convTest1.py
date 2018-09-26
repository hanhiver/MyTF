import cv2 as cv 
import numpy as np 
from matplotlib import pyplot

image = cv.imread('./RES/lena.jpg')

# 锐化
kernel_sharp = np.array([
	[0, -2, 0],
	[-2, 9, -2],
	[0, -2, 0]
	])

# 边界检测
kernel_edges = np.array([
	[-1, -1, -1],
	[-1, 8, -1],
	[-1, -1, -1]
	])

# 模糊
kernel_blur = np.array([
	[2, 2, 2],
	[2, 2, 2],
	[2, 2, 2]
	]) / 18.0

# 浮雕
kernel_emboss = np.array([
	[-2, -1, 0],
	[-1, 1, 1],
	[0, 1, 2]
	])

sharpen = cv.filter2D(image, -1, kernel_sharp)
edges = cv.filter2D(image, -1, kernel_edges)
blur = cv.filter2D(image, -1, kernel_blur)
emboss = cv.filter2D(image, -1, kernel_emboss)

titles = ['Sharp', 'Edges', 'Blur', 'Emboss']
images = [sharpen, edges, blur, emboss]

for i in range(4):
    pyplot.subplot(2, 2, i+1), pyplot.imshow(images[i], 'gray')
    pyplot.title(titles[i])
    pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
