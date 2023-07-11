import bisect  # 导入bisect模块，用于对有序列表进行插入操作

import numpy as np  # 导入NumPy库，用于数值计算

import utils  # 导入自定义的工具函数
from building_boundary.utils import create_pairs
from building_boundary.utils.angle import angle_difference
from .intersect import perpedicular_line_intersect  # 导入perpedicular_line_intersect函数
from .segment import BoundarySegment  # 导入BoundarySegment类
from building_boundary.utils.geometry import distance
from building_boundary.utils.error import ThresholdError
'''

这段代码包含了一系列函数，用于合并具有相似方向的线段。具体作用如下：

find_pivots(orientations, angle): 找到方向变化超过给定角度的位置索引，作为枢轴点。
get_points_between_pivots(segments, pivots): 返回两个枢轴点之间的点。
get_segments_between_pivots(segments, pivots): 获取两个枢轴之间的线段。
parallel_distance(segment1, segment2): 计算两个平行线段之间的距离。
check_distance(segments, pivots, max_distance): 检查所有相邻平行线段之间的距离是否大于给定的最大距离，如果是，则插入一个枢轴。
merge_between_pivots(segments, start, end, max_error=None): 合并两个枢轴之间的线段。
merge_segments(segments, angle_epsilon=0.05, max_distance=None, max_error=None): 合并具有相似方向的线段，根据角度差和距离差将相邻线段合并成更长的线段。
'''

def find_pivots(orientations, angle):
    """
    找到方向变化超过给定角度的位置索引。

    Parameters
    ----------
    orientations : list of float
        方向序列。
    angle : float or int
        被认为是枢轴的角度差。

    Returns
    -------
    pivot_indices : list of int
        枢轴点的索引。
    """
    # 计算相邻方向之间的差异
    ori_diff = np.fromiter((angle_difference(a1, a2) for
                            a1, a2 in create_pairs(orientations)),
                           orientations.dtype)
    # 根据角度差异是否超过给定的角度阈值，生成布尔值数组
    pivots_bool = ori_diff > angle
    # 找到布尔数组中为True的索引位置
    pivots_idx = list(np.where(pivots_bool)[0] + 1)

    # 边缘情况处理：首尾方向差异超过阈值
    if pivots_idx[-1] > (len(orientations)-1):
        del pivots_idx[-1]  # 删除最后一个枢轴索引
        pivots_idx[0:0] = [0]  # 将0作为第一个枢轴索引

    return pivots_idx


def get_points_between_pivots(segments, pivots):
    """
    返回两个枢轴点之间的点。

    Parameters
    ----------
    segments : list of BoundarySegment
        线段。
    pivots : list of int
        枢轴点的索引。

    Returns
    -------
    points : (Mx2) array
    """
    k, n = pivots
    points = [segments[k].points[0]]  # 添加第一个枢轴点之前的点
    if k < n:
        for s in segments[k:n]:
            points.extend(s.points[1:])  # 添加两个枢轴点之间每个线段的端点
    else:  # 边缘情况：跨越首尾的情况
        for s in segments[k:]:
            points.extend(s.points[1:])  # 添加第一个枢轴点到最后一个线段的端点
        for s in segments[:n]:
            points.extend(s.points[1:])  # 添加第一个线段到最后一个枢轴点的端点

    return np.array(points)


def get_segments_between_pivots(segments, pivots):
    """
    获取两个枢轴之间的线段。

    Parameters
    ----------
    segments : list of BoundarySegment
        线段。
    pivots : list of int
        枢轴点的索引。

    Returns
    -------
    segments : list of int
        位于枢轴之间的线段的索引。
    """
    k, n = pivots
    if k < n:
        return list(range(k, n))
    else:  # 边缘情况：跨越首尾的情况
        segments_pivots = list(range(k, len(segments)))
        segments_pivots.extend(range(0, n))
        return segments_pivots


def parallel_distance(segment1, segment2):
    """
    计算两个平行线段之间的距离。

    Parameters
    ----------
    segment1 : BoundarySegment
        第一个线段。
    segment2 : BoundarySegment
        第二个线段。

    Returns
    -------
    distance : float
        从线段1的端点到线段2的直线的垂直距离。
    """
    intersect = perpedicular_line_intersect(segment1, segment2)
    if len(intersect) > 0:  # 如果存在交点
        # 计算线段1的端点到交点的距离
        return distance(segment1.end_points[1], intersect)
    else:
        return float('inf')  # 如果没有交点，则距离无穷大


def check_distance(segments, pivots, max_distance):
    """
    检查所有相邻平行线段之间的距离是否大于给定的最大距离，
    如果是，则插入一个枢轴。

    Parameters
    ----------
    segments : list of BoundarySegment
        线段。
    pivots : list of int
        枢轴点的索引。
    max_distance : float
        两个相邻平行线段之间可以合并的最大距离。

    Returns
    -------
    pivots : list of int
        枢轴点的索引。
    """
    distances = []
    # 计算相邻线段之间的距离
    for i, pair in enumerate(create_pairs(segments)):
        if (i + 1) % len(segments) not in pivots:
            distances.append(parallel_distance(pair[0], pair[1]))
        else:
            distances.append(float('nan'))  # 将非相邻线段的距离设为nan
    too_far = np.where(np.array(distances) > max_distance)[0] + 1
    if len(too_far) > 0:
        too_far[-1] = too_far[-1] % len(segments)
        for x in too_far:
            bisect.insort_left(pivots, x)  # 将距离超过阈值的位置作为新的枢轴点插入
    return pivots, distances


def merge_between_pivots(segments, start, end, max_error=None):
    """
    合并两个枢轴之间的线段。

    Parameters
    ----------
    segments : list of BoundarySegment
        线段。
    start : int
        第一个线段的索引。
    end : int
        要合并的线段的索引。
    max_error : float
        点到计算线的最大误差（距离）。

    Returns
    -------
    merged_segment : BoundarySegment
        合并的线段。
    """
    if end == start + 1:
        return segments[start]  # 如果只有一个线段，则直接返回该线段
    else:
        points = get_points_between_pivots(
            segments,
            [start, end]
        )

        merged_segment = BoundarySegment(points)  # 创建一个新的线段
        merge_segments = np.array(segments)[get_segments_between_pivots(
            segments, [start, end]
        )]
        longest_segment = max(merge_segments, key=lambda s: s.length)  # 找到最长的线段
        orientation = longest_segment.orientation
        # 根据最长线段的方向对新线段进行规范化
        merged_segment.regularize(orientation, max_error=max_error)
        return merged_segment


def merge_segments(segments, angle_epsilon=0.05,
                   max_distance=None, max_error=None):
    """
    合并具有相似方向的线段。

    Parameters
    ----------
    segments : list of BoundarySegment
        线段。
    angle_epsilon : float
        两个角度被认为是相同的角度（以弧度表示）。
    max_distance : float
        如果两个相邻的线段（根据角度差）之间的距离小于此值，则合并线段。
    max_error : float
        点到计算线的最大误差（距离）。

    Returns
    -------
    segments : list of BoundarySegment
        合并后的线段列表。
    """
    orientations = np.array([s.orientation for s in segments])  # 提取线段的方向
    pivots = find_pivots(orientations, angle_epsilon)  # 找到枢轴点

    if max_distance is not None:
        pivots, distances = check_distance(segments, pivots, max_distance)

    while True:
        new_segments = []
        try:
            for pivot_segment in create_pairs(pivots):
                new_segment = merge_between_pivots(
                    segments, pivot_segment[0], pivot_segment[1], max_error
                )
                new_segments.append(new_segment)  # 将合并后的线段添加到列表中
            break
        except ThresholdError:
            # 如果合并时超过了阈值，则需要重新调整枢轴点
            segments_idx = get_segments_between_pivots(segments, pivot_segment)
            new_pivot_1 = segments_idx[
                np.nanargmax(np.array(distances)[segments_idx])
            ]
            new_pivot_2 = (new_pivot_1 + 1) % len(segments)
            if new_pivot_1 not in pivots:
                bisect.insort_left(pivots, new_pivot_1)  # 插入新的枢轴点
            if new_pivot_2 not in pivots:
                bisect.insort_left(pivots, new_pivot_2)  # 插入新的枢轴点

    return new_segments

