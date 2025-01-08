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

# def auto_canny(image, sigma=0.33):
#     # 计算图像的中值
#     median = np.median(image)
#
#     # 设置 Canny 边缘检测的低阈值和高阈值
#     lower = int(max(0, (1.0 - sigma) * median))
#     upper = int(min(255, (1.0 + sigma) * median))
#
#     # 使用 Canny 边缘检测
#     edges = cv2.Canny(image, lower, upper)
#     return edges
#
# # 使用自适应阈值处理
# adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# edges = auto_canny(adaptive_thresh)
#
# # 查找轮廓
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#

import numpy as np
import cv2

# 读取图像
image_path = r"D:\Project\mlsd\contours4.png"
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print(f"Error: Unable to load image at {image_path}. Please check the file path.")
    exit()

def find_max_contour(image, threshold1=50, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        if len(max_contour) <= 100:
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea)

    else:
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
        else:
            max_contour = contours[0]
    return max_contour



image_show = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
max_contour = find_max_contour(image)
print(len(max_contour))
# 显示轮廓，调整颜色和线宽
cv2.drawContours(image_show, [max_contour], -1, (255), 2)  # BGR格式颜色
# 获取所有轮廓点的坐标
contour_points = np.column_stack(np.where(image_show == 255))
print(len(contour_points))
# 设置窗口名称和大小
window_name = 'Adaptive Threshold Contours'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # 调整窗口大小，例如800x600

# 显示图像
cv2.imshow(window_name, image_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

