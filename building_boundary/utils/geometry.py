# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

# distance(p1, p2)：计算两个二维空间点之间的欧氏距离。它接受两个参数 p1 和 p2，这些参数可以是列表或数组，表示二维空间中的点。函数返回两点之间的欧氏距离。
# perpendicular_line(line, p)：在给定直线上的一点处返回与该直线垂直的直线。它接受两个参数，line 表示直线的系数（ax + by + c = 0），
# 其中 line 是一个包含三个元素的数组或列表，分别表示直线的 a、b、c 系数；p 表示直线上的一点，是一个包含两个元素的数组或列表，表示点的坐标。函数返回通过给定点 p 并且垂直于输入直线的直线的系数。
def distance(p1, p2):
    """
    The euclidean distance between two points.

    Parameters
    ----------
    p1 : list or array
        A point in 2D space.
    p2 : list or array
        A point in 2D space.

    Returns
    -------
    distance : float
        The euclidean distance between the two points.
    """
    # math.hypot() 函数用于计算给定点集中每个点到原点的欧几里得范数。
    return math.hypot(*(p1-p2))


def perpedicular_line(line, p):
    """
    返回通过给定点垂直于一条直线的直线。

    参数
    ----------
    line : (1x3) 类数组
        直线的系数 (a, b, c)，标准形式为 ax + by + c = 0。
    p : (1x2) 类数组
        直线上的一个点的坐标。

    返回
    -------
    line : (1x3) 类数组
        垂直于输入直线且通过给定点的直线的系数 (a, b, c)。
        返回的直线也是用标准形式表示的 (ax + by + c = 0)。
    """
    a, b, c = line
    # 对于垂直于原直线的直线，我们交换 'a' 和 'b' 系数，并取一个的负值
    pa = b  # 新的 'a' 系数
    pb = -a  # 新的 'b' 系数
    pc = -(p[0] * b - p[1] * a)  # 使用原直线上的点计算 'c' 系数
    return [pa, pb, pc]
