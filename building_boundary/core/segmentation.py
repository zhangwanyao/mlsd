# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np  # 导入NumPy库，用于数组操作
from skimage.measure import LineModelND, ransac  # 从skimage.measure模块中导入LineModelND和ransac函数

from .segment import BoundarySegment  # 从当前目录下的segment模块中导入BoundarySegment类

'''

这段代码是一个线段提取算法，使用了RANSAC算法来拟合直线并提取线段。主要功能包括：

ransac_line_segmentation: 使用RANSAC算法对一系列点进行直线拟合，并返回属于直线内的点的索引列表。

extend_segment: 根据给定的内点序列和距离阈值，对RANSAC找到的线段进行扩展，直到无法再添加更多的内点为止。

extract_segment: 从一系列点中提取线段。根据RANSAC找到的内点索引列表，以及给定的距离阈值，将内点连续地分割成不同的线段。

get_insert_loc: 使用二分查找算法确定应该插入新线段的位置，以保持线段列表的有序性。

get_remaining_sequences: 根据给定的内点索引和标记数组，获取剩余的序列，即未被分配到任何线段的点的序列。

extract_segments: 递归地从一圈点中提取线段。根据RANSAC找到的内点索引列表，以及给定的距离阈值，将所有内点提取为线段，并将剩余的点序列分配给新的线段。

boundary_segmentation: 使用RANSAC算法从一系列点中提取线性段。这是算法的主要入口点，负责初始化数据、调用线段提取函数，并将最终提取到的线段恢复到原始坐标系中。
'''

def ransac_line_segmentation(points, distance):
    """
    使用RANSAC算法对直线进行分割。

    参数
    ----------
    points : (Mx2) 数组
        点的坐标
    distance : 浮点数
        点到直线的最大距离，用于确定是否属于直线。

    返回
    -------
    inliers : 布尔值列表
        表示点是否是内点（即属于直线）。
    """
    _, inliers = ransac(points, LineModelND,  # 使用RANSAC算法拟合直线
                        min_samples=2,  # 最小样本数
                        residual_threshold=distance,  # 残差阈值，用于判断点是否属于直线
                        max_trials=1000)  # 最大迭代次数
    return inliers  # 返回内点的布尔值列表

def extend_segment(segment, points, indices, distance):
    """
    根据序列和距离扩展由RANSAC找到的线段。

    参数
    ----------
    segment : int列表
        属于线段/直线的点的索引。
    points : (Mx2) 数组
        所有点的坐标。
    indices : int列表
        序列中点的索引。
    distance : 浮点数
        点到直线的最大距离，用于确定点是否属于该直线。

    返回
    -------
    segment : int列表
        属于线段/直线的点的索引。
    """
    n_points = len(points)  # 点的数量
    line_segment = BoundarySegment(points[segment])  # 创建由给定点索引构成的边界线段对象

    edge_case = indices[0] == 0 and indices[-1] == n_points - 1  # 判断是否为边缘情况

    i = segment[0] - 1  # 从线段的起始点向前遍历
    while True:
        if edge_case and i < indices[0]:  # 处理边缘情况
            i = i % n_points
        elif i < indices[0] - 1 or i < 0:  # 判断是否到达序列的起始位置
            break

        if line_segment.dist_point_line(points[i]) < distance:  # 如果点到直线的距离小于阈值
            segment.insert(0, i)  # 将点插入到线段的起点
        else:
            if i - 2 >= indices[0]:  # 如果距离起始位置的距离大于等于2个点
                if not (line_segment.dist_point_line(
                            points[i - 1]) < distance and
                        line_segment.dist_point_line(
                            points[i - 2]) < distance):
                    break  # 如果前两个点不满足条件，则退出循环
            elif edge_case:
                if not (line_segment.dist_point_line(
                            points[(i - 1) % n_points]) < distance and
                        line_segment.dist_point_line(
                            points[(i - 2) % n_points]) < distance):
                    break  # 如果前两个点不满足条件，则退出循环
        i -= 1  # 继续向前遍历

    i = segment[-1] + 1  # 从线段的终点向后遍历
    while True:
        if edge_case and i > indices[-1]:  # 处理边缘情况
            i = i % n_points
        elif i > indices[-1] + 1 or i >= n_points:  # 判断是否到达序列的末尾
            break

        if line_segment.dist_point_line(points[i]) < distance:  # 如果点到直线的距离小于阈值
            segment.append(i)  # 将点添加到线段的末尾
        else:
            if i + 2 <= indices[-1]:  # 如果距离末尾位置的距离大于等于2个点
                if not (line_segment.dist_point_line(
                            points[i + 1]) < distance and
                        line_segment.dist_point_line(
                            points[i + 2]) < distance):
                    break  # 如果后两个点不满足条件，则退出循环
            elif edge_case:
                if not (line_segment.dist_point_line(
                            points[(i + 1) % n_points]) < distance and
                        line_segment.dist_point_line(
                            points[(i + 2) % n_points]) < distance):
                    break  # 如果后两个点不满足条件，则退出循环
        i += 1  # 继续向后遍历

    return segment  # 返回线段的索引列表

def extract_segment(points, indices, distance):
    """
    从一系列点中提取线段。

    参数
    ----------
    points : (Mx2) 数组
        所有点的坐标。
    indices : int列表
        序列中点的索引。
    distance : 浮点数
        点到直线的最大距离，用于确定点是否属于该直线。

    返回
    -------
    segment : int列表
        属于线段/直线的点的索引。
    """
    inliers = ransac_line_segmentation(points[indices], distance)  # 使用RANSAC算法找到线段的内点
    inliers = indices[inliers]  # 将内点索引转换为全局索引

    # 根据内点的连续性将内点分割成不同的片段
    sequences = np.split(inliers, np.where(np.diff(inliers) != 1)[0] + 1)
    # 选取最长的片段作为线段的内点
    segment = list(max(sequences, key=len))

    # 如果线段内点数量大于1，则对线段进行扩展
    if len(segment) > 1:
        segment = extend_segment(segment, points, indices, distance)
    # 如果线段内只有一个点，根据序列情况选择邻近点进行扩展
    elif len(segment) == 1:
        if segment[0] + 1 in indices:
            segment.append(segment[0] + 1)
            segment = extend_segment(segment, points, indices, distance)
        elif segment[0] - 1 in indices:
            segment.insert(0, segment[0] - 1)
            segment = extend_segment(segment, points, indices, distance)

    return segment  # 返回线段的索引列表

def get_insert_loc(segments, segment):
    """
    使用二分查找来找到应该插入新线段的位置。

    参数
    ----------
    segments : int列表的列表
        属于线段/直线的点的索引。
    segment : int列表
        属于线段/直线的点的索引。

    返回
    -------
    loc : int
        应该插入线段的索引位置。
    """
    if len(segments) == 0:  # 如果线段列表为空，直接返回0
        return 0
    if segment[0] > segments[-1][0]:  # 如果新线段的起点比已有线段的最后一个起点大，直接插入到末尾
        return len(segments)

    lo = 0  # 初始化搜索范围的下界
    hi = len(segments)  # 初始化搜索范围的上界
    while lo < hi:  # 使用二分查找
        mid = (lo + hi) // 2  # 计算中间索引
        if segment[0] < segments[mid][0]:  # 如果新线段的起点小于中间线段的起点
            hi = mid  # 缩小搜索范围的上界
        else:
            lo = mid + 1  # 缩小搜索范围的下界
    return lo  # 返回应该插入线段的位置

def get_remaining_sequences(indices, mask):
    """
    获取剩余的序列，给定已经属于某个线段的点。

    参数
    ----------
    indices : int列表
        序列中点的索引。
    mask : bool列表
        标记属于某个线段的点。

    返回
    -------
    sequences : int列表的列表
        每个剩余序列的索引。
    """
    sequences = np.split(indices, np.where(np.diff(mask) == 1)[0] + 1)  # 根据mask标记的线段边界分割indices数组

    if mask[0]:  # 如果第一个点属于线段
        sequences = [s for i, s in enumerate(sequences) if i % 2 == 0]  # 提取奇数位置的序列（剩余序列）
    else:
        sequences = [s for i, s in enumerate(sequences) if i % 2 != 0]  # 提取偶数位置的序列（剩余序列）

    sequences = [s for s in sequences if len(s) > 1]  # 去除只有一个点的序列

    return sequences  # 返回剩余序列的索引

def extract_segments(segments, points, indices, mask, distance):
    """
    从一圈点中提取线段。

    注意：这是一个递归函数。用一个空的segments列表开始。

    参数
    ----------
    segments : int列表的列表
        属于线段/直线的点的索引。
    points : (Mx2)数组
        所有点的坐标。
    indices : int列表
        序列中点的索引。
    mask : bool列表
        标记属于某个线段的点。
    distance : float
        一个点被认为属于该线的最大距离。

    """
    if len(indices) == 2:  # 如果只有两个点
        segment = list(indices)  # 创建一个线段
        segment = extend_segment(segment, points, indices, distance)  # 扩展线段
    else:
        segment = extract_segment(points, indices, distance)  # 提取线段

    if len(segment) > 2:  # 如果线段中包含超过两个点
        insert_loc = get_insert_loc(segments, segment)  # 获取插入位置
        segments.insert(insert_loc, segment)  # 将线段插入segments列表中

        mask[segment] = False  # 将线段中的点标记为False，表示它们已经被处理过

        sequences = get_remaining_sequences(indices, mask[indices])  # 获取剩余序列

        for s in sequences:
            extract_segments(segments, points, s, mask, distance)  # 递归调用提取线段函数处理剩余序列

def extract_segments_face_points(segments, points, indices, mask, distance):
    """
    从一圈点中提取线段。

    注意：这是一个递归函数。用一个空的segments列表开始。

    参数
    ----------
    segments : int列表的列表
        属于线段/直线的点的索引。
    points : (Mx2)数组
        所有点的坐标。
    indices : int列表
        序列中点的索引。
    mask : bool列表
        标记属于某个线段的点。
    distance : float
        一个点被认为属于该线的最大距离。

    """
    for line in segments:
        segment = extend_segment(segment, points, indices, distance)  # 扩展线段



def boundary_segmentation(points, distance):
    """
    使用RANSAC提取线性段。

    参数
    ----------
    points : (Mx2)数组
        点的坐标。
    distance : float
        一个点被认为属于该线的最大距离。

    返回
    -------
    segments : int数组的列表
        线性段。
    """
    points_shifted = points.copy()  # 复制点坐标
    shift = np.min(points_shifted, axis=0)  # 计算坐标偏移量
    points_shifted -= shift  # 对点进行偏移

    # mask = np.ones(len(points_shifted), dtype=np.bool)
    mask = np.ones(len(points_shifted), dtype=bool)  # 创建一个标记数组，标记每个点是否已经被处理过
    indices = np.arange(len(points_shifted))  # 创建一个索引数组，包含了点的索引

    segments = []  # 用于存储线段的列表
    extract_segments(segments, points_shifted, indices, mask, distance)  # 提取线段

    segments = [points_shifted[i] + shift for i in segments]  # 将线段恢复到原始坐标系中

    return segments

