#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/17 上午10:22
# @Author : WanYao Zhang
# import cv2
#
# def adjust_threshold(val):
#     _, binary = cv2.threshold(gray_image, val, 255, cv2.THRESH_BINARY)
#     cv2.imshow('Thresholded Image', binary)
#
# # 读取图像并转换为灰度
# image = cv2.imread('img1.png')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 创建窗口并调整窗口大小
# cv2.namedWindow('Thresholded Image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Thresholded Image', 600, 400)  # 将窗口大小调整为 600x400 像素
#
# # 创建 Trackbar
# cv2.createTrackbar('Threshold', 'Thresholded Image', 0, 255, adjust_threshold)
#
# # 初始阈值
# adjust_threshold(127)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('img1.png', 0)
# img = cv2.medianBlur(img, 5)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

import numpy as np
import cv2

# 读取灰度图像
image = cv2.imread('../src/img2.png', cv2.IMREAD_GRAYSCALE)

def auto_canny(image, sigma=0.33):
    # 计算图像的中值
    median = np.median(image)

    # 设置 Canny 边缘检测的低阈值和高阈值
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # 使用 Canny 边缘检测
    edges = cv2.Canny(image, lower, upper)
    return edges

# 使用自适应阈值处理
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
edges = auto_canny(adaptive_thresh)

# 查找轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 将灰度图像转换为彩色图像，以便更明显地显示轮廓
image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

max_contour = max(contours, key=cv2.contourArea)
# 显示轮廓，调整颜色和线宽
cv2.drawContours(image_contours, max_contour, -1, (0, 255, 255), 4)

# 设置窗口名称和大小
window_name = 'Adaptive Threshold Contours'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # 调整窗口大小，例如800x600

# 显示图像
cv2.imshow(window_name, image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
