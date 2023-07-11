# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

'''
# 这段代码主要是用于计算建筑物主要方向以及规范化建筑物的墙壁线段方向。函数功能包括：
#
# get_primary_segments(segments, num_points): 检查并返回至少由给定数量的点支持的线段。
# find_main_orientation(segments): 返回由最多点支持的线段的方向。
# sort_orientations(orientations): 根据具有该方向的线段的长度对方向进行排序。
# compute_primary_orientations(primary_segments, angle_epsilon): 基于给定的主要线段计算主要方向。
# check_perpendicular(primary_orientations, angle_epsilon): 检查是否存在与主要方向垂直的方向。
# add_perpendicular(primary_orientations, angle_epsilon): 如果主要方向中不存在近似的垂直方向，则添加一个与主要方向垂直的方向。
# get_primary_orientations(segments, num_points=None, angle_epsilon=0.05): 通过检查支持的点数来计算建筑物的主要方向。
# regularize_segments(segments, primary_orientations, max_error=None): 将线段的方向设置为给定方向中最接近的方向。
'''

# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math  # 导入数学库
import numpy as np  # 导入NumPy库
import utils  # 导入自定义的工具模块
from building_boundary.utils.error import ThresholdError
from building_boundary.utils.angle import min_angle_difference,perpendicular



def get_primary_segments(segments, num_points):
    """
    检查线段并返回至少由给定数量的点支持的线段。

    Parameters
    ----------
    segments : list of BoundarySegment
        建筑（部分）的边界（墙）线段。
    num_points : int, optional
        被认为是主要线段的线段需要至少被多少个点支持。

    Returns
    -------
    primary_segments : list of segments
        至少被给定数量的点支持的线段。
    """
    primary_segments = [s for s in segments if len(s.points) >= num_points]
    return primary_segments


def find_main_orientation(segments):
    """
    检查由最多点支持的线段，并返回该线段的方向。

    Parameters
    ----------
    segments : list of BoundarySegment
        建筑（部分）的边界（墙）线段。

    Returns
    -------
    main_orientation : float
        由最多点支持的线段的方向，以弧度表示。
    """
    longest_segment = np.argmax([len(s.points) for s in segments])  # 找到支持点最多的线段索引
    main_orientation = segments[longest_segment].orientation  # 获取该线段的方向
    return main_orientation


def sort_orientations(orientations):
    """
    根据具有该方向的线段的长度对方向进行排序。

    Parameters
    ----------
    orientations : dict
        方向和对应长度

    Returns
    -------
    sorted_orientations : list of float

    """
    unsorted_orientations = [o['orientation'] for o in orientations]  # 获取所有方向
    lengths = [o['size'] for o in orientations]  # 获取每个方向对应的长度
    sort = np.argsort(lengths)[::-1]  # 对长度进行降序排列，并获取索引
    sorted_orientations = np.array(unsorted_orientations)[sort].tolist()  # 根据索引排序方向
    return sorted_orientations


def compute_primary_orientations(primary_segments, angle_epsilon=0.05):
    """
    基于给定的主要线段计算主要方向。

    Parameters
    ----------
    primary_segments : list of BoundarySegment
        主要线段。
    angle_epsilon : float, optional
        如果两个角度的差小于此值（以弧度表示），则这两个角度将被认为是相同的。

    Returns
    -------
    primary_orientations : list of float
        以弧度表示的计算出的主要方向，按具有该方向的线段的长度排序。
    """
    orientations = []  # 存储方向及其长度的列表

    for s in primary_segments:  # 遍历主要线段
        a1 = s.orientation  # 获取当前线段的方向
        for o in orientations:  # 遍历已知方向
            a2 = o['orientation']  # 获取已知方向
            angle_diff = min_angle_difference(a1, a2)  # 计算两个角度的最小差异
            if angle_diff < angle_epsilon:  # 如果差异小于阈值
                if len(s.points) > o['size']:  # 如果当前线段的长度大于已知方向的长度
                    o['size'] = len(s.points)  # 更新已知方向的长度
                    o['orientation'] = a1  # 更新已知方向的方向
                break
        else:  # 如果当前方向不在已知方向列表中
            orientations.append({'orientation': a1,
                                 'size': len(s.points)})  # 添加当前方向到已知方向列表

    primary_orientations = sort_orientations(orientations)  # 对方向进行排序

    return primary_orientations


def check_perpendicular(primary_orientations, angle_epsilon=0.05):
    """
    检查是否存在与主要方向垂直的方向。

    Parameters
    ----------
    primary_orientations : list of floats
        主要方向的列表，其中列表中的第一个方向是主要方向（以弧度表示）。
    angle_epsilon : float, optional
        如果两个角度的差小于此值（以弧度表示），则这两个角度将被认为是相同的。

    Returns
    -------
     : int
        主要方向的垂直方向的索引。如果找不到近似的垂直方向，则返回-1。
    """
    main_orientation = primary_orientations[0]  # 获取主要方向
    diffs = [min_angle_difference(main_orientation, a)
             for a in primary_orientations[1:]]  # 计算主要方向与其他方向的角度差异
    diffs_perp = np.array(diffs) - math.pi/2  # 计算与主要方向垂直的角度差异
    closest_to_perp = np.argmin(np.abs(diffs_perp))  # 找到最接近垂直方向的索引
    if diffs_perp[closest_to_perp] < angle_epsilon:  # 如果最接近的垂直方向的角度差异小于阈值
        return closest_to_perp + 1  # 返回垂直方向的索引
    else:
        return -1  # 否则返回-1，表示未找到垂直方向


def add_perpendicular(primary_orientations, angle_epsilon=0.05):
    """
    如果主要方向中不存在近似的垂直方向，则添加一个与主要方向垂直的方向。

    Parameters
    ----------
    primary_orientations : list of floats
        主要方向的列表，其中列表中的第一个方向是主要方向（以弧度表示）。
    angle_epsilon : float, optional
        如果两个角度的差小于此值（以弧度表示），则这两个角度将被认为是相同的。

    Returns
    -------
    primary_orientations : list of floats
        经过改进的主要方向列表
    """
    main_orientation = primary_orientations[0]  # 获取主要方向
    # 如果只有一个主要方向，则添加一个与之垂直的方向
    if len(primary_orientations) == 1:
        primary_orientations.append(
            perpendicular(main_orientation)
        )
    else:
        # 检查是否存在近似垂直方向
        perp_idx = check_perpendicular(primary_orientations,
                                       angle_epsilon=angle_epsilon)
        perp_orientation = perpendicular(main_orientation)  # 计算与主要方向垂直的方向
        # 如果不存在近似的垂直方向
        if perp_idx == -1:
            primary_orientations.append(perp_orientation)  # 添加垂直方向到主要方向列表
        else:
            # 如果存在近似的垂直方向，则将该方向调整为确切的垂直方向
            primary_orientations[perp_idx] = perp_orientation

    return primary_orientations  # 返回经过改进的主要方向列表


def get_primary_orientations(segments, num_points=None,
                             angle_epsilon=0.05):
    """
    通过检查支持的点数来计算建筑物的主要方向。
    如果找到非常接近的多个方向，则将采用平均方向。
    如果找不到主要方向，则将采用由大多数点支持的线段的方向。

    Parameters
    ----------
    segments : list of BoundarySegment
        建筑（部分）的边界（墙）线段。
    num_points : int, optional
        被认为是主要线段的线段需要至少被多少个点支持。
    angle_epsilon : float, optional
        如果两个角度的差小于此值（以弧度表示），则这两个角度将被认为是相同的。

    Returns
    -------
    primary_orientations : list of float
        The computed primary orientations in radians.
    """
    if num_points is not None:
        primary_segments = get_primary_segments(segments, num_points)  # 获取至少被num_points个点支持的线段
    else:
        primary_segments = []

    if len(primary_segments) > 0:  # 如果存在主要线段
        primary_orientations = compute_primary_orientations(primary_segments,
                                                            angle_epsilon)  # 计算主要方向
    else:
        primary_orientations = [find_main_orientation(segments)]  # 否则采用支持点最多的线段的方向作为主要方向

    primary_orientations = add_perpendicular(primary_orientations,
                                             angle_epsilon=angle_epsilon)  # 添加垂直方向到主要方向列表

    return primary_orientations  # 返回计算出的主要方向列表


def regularize_segments(segments, primary_orientations, max_error=None):
    """
    将线段的方向设置为给定方向中最接近的方向。

    Parameters
    ----------
    segments : list of BoundarySegment
        要规范化的墙线段。
    primary_orientations : list of floats
        所有其他方向将设置为的方向，以弧度表示。
    max_error : float or int, optional
        规范化后线段可以具有的最大误差。如果超过此值，则保留原始方向。

    Returns
    -------
    segments : list of BoundarySegment
        规范化后的墙线段。
    """
    for s in segments:
        target_orientation = s.target_orientation(primary_orientations)  # 获取线段目标方向
        try:
            s.regularize(target_orientation, max_error=max_error)  # 规范化线段方向
        except ThresholdError:
            pass

    return segments  # 返回规范化后的线段

