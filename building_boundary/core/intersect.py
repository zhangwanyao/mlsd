# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from building_boundary import utils

'''

这段代码包含了一系列函数，用于计算线段之间的交点和添加垂直线段，具体作用如下：

perpedicular_line_intersect(segment1, segment2): 找到与线段1末端的垂直线与线段2的直线的交点。
intersect_distance(intersect, segment1, segment2): 计算线段与交点之间的距离。
compute_intersections(segments, perp_dist_weight=3): 遍历给定的线段列表，对于每一对相邻的线段，计算它们的交点。根据一定的条件选择交点或者在线段末尾添加垂直线，并计算垂直线的交点。将所有计算得到的交点收集起来，并返回作为 NumPy 数组。
'''


def perpedicular_line_intersect(segment1, segment2):
    """
    找到与线段1末端的垂直线与线段2的直线的交点。

    参数
    ----------
    segment1 : BoundarySegment
        BoundarySegment 对象
    segment2 : BoundarySegment
        接下来的 BoundarySegment 对象

    返回
    -------
     : (1x2) 数组
        交点的坐标。
    """
    perp_line = utils.geometry.perpedicular_line(segment1.line,  # 计算线段1末端处的垂直线
                                                 segment1.end_points[1])
    return segment2.line_intersect(perp_line)  # 计算垂直线与线段2的交点


def intersect_distance_wall(intersect, segment1, segment2):
    """
    计算线段与交点之间的距离。

    参数
    ----------
    intersect : (1x2) 数组
        交点的坐标。
    segment1 : BoundarySegment
        BoundarySegment 对象
    segment2 : BoundarySegment
        接下来的 BoundarySegment 对象

    返回
    -------
     : 浮点数
        线段与交点之间的距离。
    """
    segment1_end_points = utils.geometry.distance(segment1.end_points[1], intersect)


    return min(utils.geometry.distance(segment1.end_points[1], intersect),  # 计算交点与第一个线段终点的距离
               utils.geometry.distance(segment2.end_points[0], intersect))  # 计算交点与第二个线段起点的距离


def intersect_distance(intersect, segment1, segment2):
    """
    计算线段与交点之间的距离。

    参数
    ----------
    intersect : (1x2) 数组
        交点的坐标。
    segment1 : BoundarySegment
        BoundarySegment 对象
    segment2 : BoundarySegment
        接下来的 BoundarySegment 对象

    返回
    -------
     : 浮点数
        线段与交点之间的距离。
    """

    return min(utils.geometry.distance(segment1.end_points[1], intersect),  # 计算交点与第一个线段终点的距离
               utils.geometry.distance(segment2.end_points[0], intersect))  # 计算交点与第二个线段起点的距离

def compute_intersections(segments, perp_dist_weight=3):
    """
遍历给定的线段列表。
对于每一对相邻的线段，计算它们的交点。
根据一定的条件选择交点或者在线段末尾添加垂直线，并计算垂直线的交点。
将所有计算得到的交点收集起来，并返回作为 NumPy 数组。

该函数计算顺序排列的线段之间的交点。如果找不到交点，或者垂直线导致的交点更接近于线段，则会在两个线段之间添加一条垂直线。

参数
segments : 包含 BoundarySegment 对象的列表
要计算交点的墙壁线段。

perp_dist_weight : 浮点数, 可选
垂直线交点需要比两个线段之间的交点更接近多少才被优先考虑。

返回
intersections : (Mx1) 数组
计算得到的交点。
    """
    intersections = []  # 存储交点的列表
    num_segments = len(segments)  # 线段的数量
    for i in range(num_segments):  # 遍历所有线段
        segment1 = segments[i]  # 当前线段
        # 2024年3月25日 这里首尾相连难以做到，需要全部N平方遍历查找
        segment2 = segments[(i + 1) % num_segments]  # 下一个线段，循环到首尾相连
        intersect = segment1.line_intersect(segment2.line)  # 计算两线段的交点

        if any(intersect):  # 如果找到了交点
            intersect_dist = intersect_distance(intersect, segment1, segment2)  # 计算交点与线段的距离，取到线段两端的最小值
            perp_intersect = perpedicular_line_intersect(segment1, segment2)  # 计算垂直线的交点，线1末端生垂直线与线2的交点
            if any(perp_intersect):  # 如果找到垂直线的交点
                perp_intersect_dist = intersect_distance(perp_intersect,
                                                         segment1,
                                                         segment2)  # 计算垂直线的交点与线段的距离
                if ((intersect_dist >
                     perp_intersect_dist * perp_dist_weight) or
                        (segment2.side_point_on_line(intersect) == -1 and
                         perp_intersect_dist <
                         intersect_dist * perp_dist_weight)):  # 判断是否选择垂直线的交点
                    intersections.append(segment1.end_points[1])  # 将当前线段的终点添加到交点列表
                    intersections.append(perp_intersect)  # 将垂直线的交点添加到交点列表
                else:  # 否则选择线段之间的交点
                    intersections.append(intersect)  # 将线段之间的交点添加到交点列表
            else:  # 如果找不到垂直线的交点
                intersections.append(intersect)  # 将线段之间的交点添加到交点列表
        else:  # 如果找不到线段之间的交点，完全平行，这种情况几乎不会出现，所以需要设置平行的阈值，或者使用垂直线的交点与线段的距离即可
            # 如果找不到交点，则在线段末尾添加一条垂直线，并使用新线计算交点
            intersect = perpedicular_line_intersect(segment1, segment2)  # 计算垂直线的交点
            intersections.append(segment1.end_points[1])  # 将当前线段的终点添加到交点列表
            intersections.append(intersect)  # 将垂直线的交点添加到交点列表

    return np.array(intersections)  # 返回计算得到的交点数组
def compute_intersections_wall(segments, perp_dist_weight=3):
    """
遍历给定的线段列表。
对于每一对相邻的线段，计算它们的交点。
根据一定的条件选择交点或者在线段末尾添加垂直线，并计算垂直线的交点。
将所有计算得到的交点收集起来，并返回作为 NumPy 数组。

该函数计算顺序排列的线段之间的交点。如果找不到交点，或者垂直线导致的交点更接近于线段，则会在两个线段之间添加一条垂直线。

参数
segments : 包含 BoundarySegment 对象的列表
要计算交点的墙壁线段。

perp_dist_weight : 浮点数, 可选
垂直线交点需要比两个线段之间的交点更接近多少才被优先考虑。

返回
intersections : (Mx1) 数组
计算得到的交点。
    """

    intersections = []  # 存储交点的列表
    num_segments = len(segments)  # 线段的数量
    for i in range(num_segments):  # 遍历所有线段
        segment1 = segments[i]  # 当前线段
        # 2024年3月25日 这里首尾相连难以做到，需要全部N平方遍历查找
        for j in range(num_segments):  # 遍历所有线段
            if  i != j:
                segment2 = segments[j]  # 下一个线段，循环到首尾相连
                intersect = segment1.line_intersect(segment2.line)  # 计算两线段的交点
                if any(intersect):  # 如果找到了交点
                    intersect_dist = intersect_distance(intersect, segment1, segment2)  # 计算交点与线段的距离，取到线段两端的最小值
                    perp_intersect = perpedicular_line_intersect(segment1, segment2)  # 计算垂直线的交点，线1末端生垂直线与线2的交点

                    # ----------------------------情况1：一段隔墙的两面：（平行）交点与线段的距离极远，（接近）且垂线距离小于128----------------------------
                    # ----------------------------情况2：一段厚墙的两面：（平行）交点与线段的距离极远，（接近）且垂线距离小于250----------------------------
                    # ----------------------------情况3：平行忽略：（平行）交点与线段的距离极远，（接近）且垂线距离大于250----------------------------
                    # ----------------------------情况4：合并同墙，共线且重合：（平行）平行，且交点与线段的距离近，有重合部分----------------------------
                    # ----------------------------情况5：合并有洞，共线不重合，但最小距离大于门洞值：（平行）平行，且交点与线段的距离近，有重合部分----------------------------
                    # ----------------------------情况6：有门洞，新建线，共线不重合，但最小距离在门洞值区间：（平行）平行，且交点与线段的距离近，有重合部分----------------------------
                    # ----------------------------情况7：转角：（垂直）交点与线段的距离近----------------------------
                    # ----------------------------情况8：转角忽略：（垂直）交点与线段的距离远----------------------------
                    # ---------最鲁棒的是 7垂直且距离极近 > 4合并同墙，共线且重合 > 1一段隔墙的两面 >其他
                    #----------所以首先判断平行垂直距离关系，先把741三情况给解决了，再可选下列操作，甚至加入一点人工参数也可以。



                    if any(perp_intersect):  # 如果找到垂直线的交点
                        perp_intersect_dist = intersect_distance(perp_intersect,
                                                                 segment1,
                                                                 segment2)  # 计算垂直线的交点与线段的距离
                        if ((intersect_dist >
                             perp_intersect_dist * perp_dist_weight) or
                                (segment2.side_point_on_line(intersect) == -1 and
                                 perp_intersect_dist <
                                 intersect_dist * perp_dist_weight)):  # 判断是否选择垂直线的交点
                            intersections.append(segment1.end_points[1])  # 将当前线段的终点添加到交点列表
                            intersections.append(perp_intersect)  # 将垂直线的交点添加到交点列表
                        else:  # 否则选择线段之间的交点
                            intersections.append(intersect)  # 将线段之间的交点添加到交点列表
                    else:  # 如果找不到垂直线的交点
                        intersections.append(intersect)  # 将线段之间的交点添加到交点列表


                else:  # 如果找不到线段之间的交点，完全平行，这种情况几乎不会出现，所以需要设置平行的阈值，或者使用垂直线的交点与线段的距离即可
                    # 如果找不到交点，则在线段末尾添加一条垂直线，并使用新线计算交点
                    intersect = perpedicular_line_intersect(segment1, segment2)  # 计算垂直线的交点
                    intersections.append(segment1.end_points[1])  # 将当前线段的终点添加到交点列表
                    intersections.append(intersect)  # 将垂直线的交点添加到交点列表

    return np.array(intersections)  # 返回计算得到的交点数组