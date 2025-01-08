#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/17 下午5:10
# @Author : WanYao Zhang
import cv2
import math
import numpy as np
from functional_domain import line_door_walls
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from config import Config

def convert_to_point_cloud(x, y, off_size, Image_rows,minx,miny):
    """
    根据公式将像素坐标 (x, y) 转换为点云坐标。
    """
    point_cloud_x = (x - off_size) * 10 + minx
    point_cloud_y = (Image_rows - y - off_size) * 10 + miny
    return [point_cloud_x, point_cloud_y]

def find_intersection(p1, p2, p3, p4):
    # 根据两条线段的点，计算线段方程的参数
    # 线段1: (x1, y1) -> (x2, y2)
    # 线段2: (x3, y3) -> (x4, y4)

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 计算分母
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 如果分母为零，则线段平行或共线，没有交点
    if denominator == 0:
        return None

    # 计算交点的坐标
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # 检查交点是否在两条线段的范围内
    if min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and \
            min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4):
        return np.array([intersect_x, intersect_y])

    return None


def update_line_segments(line_vertices):
    i = 0  # 初始化迭代索引
    print("卡住了")
    while i < len(line_vertices):
        if i == len(line_vertices) - 1:
            line1 = line_vertices[i]
            line2 = line_vertices[0]
        else:
            line1 = line_vertices[i]
            line2 = line_vertices[i + 1]

        # 获取第一条线段的终点和第二条线段的起点
        end_of_line1 = line1[0][1]
        start_of_line2 = line2[0][0]

        # 如果终点和起点不一致，计算它们的交点
        if tuple(end_of_line1) != tuple(start_of_line2):
            intersection = find_intersection(line1[0][0], line1[0][1], line2[0][0], line2[0][1])
            if intersection is None:
                continue
            elif len(intersection) > 0:
                # 更新第一条线段的终点和第二条线段的起点为交点
                line_vertices[i] = [([line1[0][0], intersection])]
                if i == len(line_vertices) - 1:
                    line_vertices[0] = [([intersection, line2[0][1]])]  # 如果是最后一条线段，更新第一条线段
                else:
                    line_vertices[i + 1] = [([intersection, line2[0][1]])]
        i += 1

    return line_vertices


def slope(point1,point2):
    '''
    计算直线的斜率
    '''
    x1,y1 = point1
    x2,y2 = point2
    if x2 - x1 == 0:
        if y2 - y1 ==0:
            return None
        else:
            return float('inf')
    else:
        return (y2-y1)/(x2-x1)

def distance_points(point1, point2):
    '''
    计算点之间的距离
    '''
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate(line1,line2,threshold = 5):
    '''
    判断两条直线的关系
    '''
    lines = []
    # 计算第一条线的全部信息
    slope1 = slope(line1[0],line1[1])
    if slope1 is None:
        angle1 = None
    elif slope1 == float('inf'):
        angle1 = 0
    else:
        angle1 = math.atan(slope1) * 180 / math.pi  # 弧度转角度
    if angle1 < 0:
        angle1 += 180
    distance1 = distance_points(line1[0],line1[1])

    # 计算第二条线的全部信息
    slope2 = slope(line2[0],line2[1])
    if slope2 is None:
        angle2 = None
    elif slope2 == float('inf'):
        angle2 = 0
    else:
        angle2 = math.atan(slope2) * 180 / math.pi  # 弧度转角度
    if angle2 < 0:
        angle2 += 180
    distance2 = distance_points(line2[0],line2[1])

    # 判断两条线段是否合并
    if distance1 <= threshold or distance2 <= threshold:
        print("距离1：", distance1, "距离2", distance2)
        line = [line1[0],line2[1]]
        lines.append(line)
    elif angle1 is None or angle2 is None:
        print("角度1：",angle1,"角度2：",angle2)
        line = [line1[0], line2[1]]
        lines.append(line)
    elif abs(angle1 - angle2) < 10:
        print("角度合并","角度1：", angle1, "角度2：", angle2)
        line = [line1[0],line2[1]]
        lines.append(line)
    else:
        lines.append(line1)
    return lines

def sorted_vertices(vertices):
    '''
    对多边形的线段重新排序和合并
    '''
    sorted_lines = []
    for i in range(len(vertices)):
        if i == len(vertices) - 1:
            line1 = vertices[i]
            line2 = vertices[0]
            lines = calculate(line1,line2)
        else:
            line1 = vertices[i]
            line2 = vertices[i + 1]
            lines = calculate(line1, line2)

        if len(lines) == 2:
            sorted_lines.append(lines[0])
            sorted_lines.append(lines[1])
        else:
            sorted_lines.append(lines)
    return sorted_lines

def distance_point_to_line(px, py, x1, y1, x2, y2):
    """
    计算点 (px, py) 到线段 (x1, y1), (x2, y2) 的距离
    """
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def points_extract_points_near_line(sclice_points,start,end,threshold = 5):
    points = []
    if abs(start[0] - end[0]) <= 0.1:
        for point in sclice_points:
            x,y = point
            miny, maxy = min(start[1], end[1]), max(start[1], end[1])
            minx, maxx = start[0] - threshold, start[0] + threshold
            if (miny <= y <= maxy) and minx <= x <= maxx:
                points.append([x, y])
    elif abs(start[1] - end[1]) <= 0.1:
        for point in sclice_points:
            x, y = point
            minx, maxx = min(start[0], end[0]), max(start[0], end[0])
            miny, maxy = start[1] - threshold, start[1] + threshold
            if (miny <= y <= maxy) and minx <= x <= maxx:
                points.append([x, y])
    return np.array(points)

def png_extract_points_near_line(img, start, end, threshold=5, gray_threshold=240):
    """
        提取原始图像上靠近线段 (start, end) 的点，并且灰度值要大于指定的阈值
        :param img: 输入的原始图像
        :param start: 线段起点 (x1, y1)
        :param end: 线段终点 (x2, y2)
        :param threshold: 点到线段的距离阈值，控制提取的邻域大小
        :param gray_threshold: 灰度值的阈值，只有灰度值大于该值的点才会被提取
        :return: 返回线段附近的点坐标列表
        """
    points = []
    ys, xs = np.where(img >= gray_threshold)
    if abs(start[0] - end[0]) <= 0.1:
        for x,y in zip(xs, ys):
            miny, maxy = min(start[1],end[1]), max(start[1],end[1])
            minx,maxx = start[0] - threshold, start[0] + threshold
            if (miny <= y <= maxy) and minx <= x <= maxx:
                points.append([x, y])
    elif abs(start[1] - end[1]) <= 0.1:
        for x,y in zip(xs, ys):
            minx, maxx = min(start[0],end[0]), max(start[0],end[0])
            miny,maxy = start[1] - threshold, start[1] + threshold
            if (miny <= y <= maxy) and minx <= x <= maxx:
                points.append([x, y])
    # for x, y in zip(xs, ys):
    #     dist = distance_point_to_line(x, y, start[0], start[1], end[0], end[1])
    #     if dist < threshold:
    #         points.append([x, y])
    return np.array(points)


def match_and_visualize(vertices, img,rt_slice_points,threshold = 10):
    '''
    将线和点进行匹配后更新墙和洞口位置
    '''
    xy_points = rt_slice_points[:,:2]*1000
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_walls = []
    all_doors = []
    off_size = Config.off_size
    Image_rows = Config.Image_rows
    minx = Config.minx
    miny = Config.miny

    # for line in vertices:
    #     # 将原始起点和终点转换为点云坐标
    #     start = convert_to_point_cloud(line[0][0], line[0][1], off_size, Image_rows, minx, miny)
    #     end = convert_to_point_cloud(line[1][0], line[1][1], off_size, Image_rows, minx, miny)
    #
    #     # 提取接近直线的点
    #     line_points = points_extract_points_near_line(xy_points, start, end, 100)
    #     x = line_points[:, 0].reshape(-1, 1)
    #     y = line_points[:, 1]
    #
    #     # 使用 RANSAC 进行拟合
    #     ransac = RANSACRegressor(random_state=42)
    #     ransac.fit(x, y)
    #
    #     # 获取拟合直线的系数
    #     slope = ransac.estimator_.coef_[0]  # 斜率
    #     intercept = ransac.estimator_.intercept_  # 截距
    #
    #     # 计算新的直线的起点和终点
    #     new_start_x = x.min()  # 使用点云中的最小 x 值
    #     new_end_x = x.max()  # 使用点云中的最大 x 值
    #     new_start_y = slope * new_start_x + intercept
    #     new_end_y = slope * new_end_x + intercept
    #
    #     # 更新直线的起点和终点
    #     line[0] = (new_start_x, new_start_y)
    #     line[1] = (new_end_x, new_end_y)
    #
    #     # 可视化 RANSAC 拟合结果
    #     inlier_mask = ransac.inlier_mask_
    #     outlier_mask = np.logical_not(inlier_mask)
    #
    #     plt.scatter(x[inlier_mask], y[inlier_mask], color='blue', marker='o', label='Inliers')
    #     plt.scatter(x[outlier_mask], y[outlier_mask], color='red', marker='x', label='Outliers')
    #     plt.plot([new_start_x, new_end_x], [new_start_y, new_end_y], color='green', linewidth=2, label='RANSAC Fit')
    #     plt.plot([start[0], end[0]], [start[1], end[1]], color='orange', linewidth=2, linestyle='--',
    #              label='Original Line')
    #     # plt.plot([start[0],end[0]],[start[1],end[1]],color = "black",linewidth=3,lable="initial")
    #     plt.legend(loc='upper left')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('RANSAC Regression Fit')
    #     plt.show()

    for line in vertices:
        # start, end = line[0][0],line[0][1]
        start,end = line[0],line[1]
        line_points = png_extract_points_near_line(gray, start, end, threshold)

        # if len(line_points) > 100:
        ptolines = line_door_walls.points_to_line(line_points, start[0], start[1], end[0], end[1])
        sort_ptolines = line_door_walls.sort_stend(ptolines, start[0], start[1])
        # sort_ptolines.extend(end)
        walls,doors = line_door_walls.cluster_walls_and_doors(sort_ptolines, 50)
        all_walls.extend(walls)
        all_doors.extend(doors)
        # # 可视化提取出的点
        # plt.figure(figsize=(8, 6))
        # plt.imshow(gray, cmap='gray')
        # plt.scatter(line_points[:, 0], line_points[:, 1], color='red', s=1, label='Points near line')
        #
        # plt.plot([start[0], end[0]], [start[1], end[1]], color='blue', lw=1, label='Line Segment')
        #
        # plt.title(f'Points Near Line from {start} to {end}')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend(loc='best')
        # plt.show()
    return all_walls,all_doors