#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/24 下午1:36
# @Author : WanYao Zhang
from scipy.spatial import KDTree
import open3d as o3d
import cv2
import random
import numpy as np

def read_point_cloud(filename):
    '''
    作用：点云读取
    下一步加入对读取点云格式的判断
    '''
    point_cloud = o3d.io.read_point_cloud(filename)
    return point_cloud

def height_min_max(points,unit,step=100):
    '''
    计算点云数据的最高点和最低点，迭代逼近，直到高度符合要求。

    参数：
    - points: 点云数据，格式为 Nx3 的 numpy 数组。
    - unit: 单位， 'm'或 'mm'。
    - step: 每次迭代逼近的步长，默认为 100。

    返回：
    - 满足高度要求时的最小点和最大点，如果找不到则返回 None。
    '''
    sorted_z_min = np.sort(points[:, 2])
    sorted_z_max = np.sort(points[:, 2])[::-1]

    i = 0
    j = 0
    while i < len(sorted_z_min) and j < len(sorted_z_max):
        wall_min = sorted_z_min[i]
        wall_max = sorted_z_max[j]

        height = wall_max - wall_min

        # 检查单位并判断高度是否满足要求
        if unit == 'mm':
            if 2000 <= height <= 5000:
                return wall_min, wall_max
        elif unit == 'm':
            if 2.0 <= height <= 5.5:
                return wall_min, wall_max

        # 如果高度不符合，继续迭代
        i += step
        j += step

    # 如果循环结束仍未找到合适的高度
    return None

def unit_vector(vector):
    '''
    计算输入点云的单位大小
    '''
    data_random = np.random.choice(len(vector), size=100, replace=False)
    data_random_points = vector[data_random, :]  # 随机抽取100个三维点

    data_random_min = np.min(data_random_points, axis=0)
    data_random_max = np.max(data_random_points, axis=0)

    distance = np.sqrt((data_random_max[0] - data_random_min[0]) ** 2 \
                       + (data_random_max[1] - data_random_min[1]) ** 2 \
                       + (data_random_max[2] - data_random_min[2]) ** 2)
    if distance < 1000:
        unit = 'm'
    else:
        unit = 'mm'
    return unit
def sliced_point_cloud(wall_min,wall_max,height,section_thickness,points):
    '''
    对点云进行切片化处理
    '''
    random_slice = [random.uniform(wall_min,wall_max) for _ in range(10)]
    for slice in random_slice:
        mask = (points[:,2] >= slice - section_thickness/20) & (points[:,2] <= slice + section_thickness/20)
        multi_sliced_points = points[mask]
    lower_bound = wall_min + height
    upper_bound = wall_min + height + section_thickness * 2
    # lower_bound = wall_min
    # upper_bound = wall_max
    mask = (lower_bound < points[:, 2]) & (points[:, 2] < upper_bound)
    sliced_points = points[mask]
    return sliced_points,multi_sliced_points
def remove_noise(points,step,percentile):
    '''
    去除点云数据噪声
    '''
    points_2d = points[:,:2]
    xmin, ymin = np.amin(points_2d, axis=0)
    xmax, ymax = np.amax(points_2d, axis=0)
    bounds_x = xmax - xmin
    bounds_y = ymax - ymin
    # 按照密度阈值进行筛选
    grid_size_x = np.floor(bounds_x / (2*step)).astype(int)
    grid_size_y = np.floor(bounds_y / (2*step)).astype(int)

    # 计算点云的密度值
    x_edges = np.linspace(xmin,xmax,grid_size_x + 1)
    y_edges = np.linspace(ymin,ymax,grid_size_y + 1)
    histogram,_,_ = np.histogram2d(points_2d[:,0],points_2d[:,1],bins=[x_edges,y_edges])
    non_zero_histogram = histogram[histogram > 0]

    # 筛选密度高的区域
    threshold = np.percentile(non_zero_histogram, percentile)  # 选择密度的30百分位
    print(f"threshold: {threshold}")
    high_density_indices = np.argwhere(histogram > threshold)


    # 创建KDTree来加速点的提取
    kdtree = KDTree(points_2d)

    # 根据高密度区域的边界值查找落在这些区域内的点
    high_density_points = []
    for i, j in high_density_indices:
        x_min_edge, x_max_edge = x_edges[i], x_edges[i + 1]
        y_min_edge, y_max_edge = y_edges[j], y_edges[j + 1]

        # 查询所有在高密度区域内的点
        points_in_box = kdtree.query_ball_point([(x_min_edge + x_max_edge) / 2, (y_min_edge + y_max_edge) / 2],
                                                r=max(x_max_edge - x_min_edge, y_max_edge - y_min_edge) / 2)

        if points_in_box:
            high_density_points.append(points_2d[points_in_box])

    # 将所有的高密度点转换为NumPy数组
    high_density_points = np.vstack(high_density_points)

    return high_density_points
def binary_graph_conversion(points,step):
    '''
    二值化，将点云数据转换成三通道 RGB 图片
    '''

    points_2d = points
    xmin, ymin = np.amin(points_2d, axis=0)
    xmax, ymax = np.amax(points_2d, axis=0)
    width = np.ceil((xmax - xmin) / step)
    height = np.ceil((ymax - ymin) / step)

    # 创建一个三通道的白色图像
    img = np.full((int(height), int(width), 3), fill_value=255, dtype=np.uint8)
    for i in range(len(points_2d)):
        col = np.floor((points_2d[i, 0] - xmin) / step)
        row = np.floor((points_2d[i, 1] - ymin) / step)
        img[int(row), int(col)] = [0, 0, 0]  # 黑色像素点

    return img
    # reversed_img = np.flipud(img)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # filename = "../../mlsd/clap.png"
    # cv2.imwrite(filename, img)



def point_cloud_to_png(points):
    unit = unit_vector(points)
    print(unit)
    wall_min, wall_max = height_min_max(points,unit)
    print("墙的最低最高点：",wall_min,wall_max)
    if unit == 'm':
        height = 1.5
        section_thickness = 0.05
        step = 0.01 #像素大小
    else:
        height = 1500
        section_thickness = 50
        step = 10
    sliced_points,multi_sliced_points = sliced_point_cloud(wall_min,wall_max,height,section_thickness,points)
    r_noise_points = remove_noise(sliced_points,step,percentile = 30)
    img = binary_graph_conversion(r_noise_points,step)
    return sliced_points,multi_sliced_points,img