# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np  # 导入numpy库，用于处理数组和数值计算
from scipy.spatial import ConvexHull  # 从scipy库中导入ConvexHull类，用于计算凸包
from shapely import unary_union  # 从shapely库中导入unary_union函数，用于对几何图形进行联合操作
from shapely.geometry import Polygon  # 从shapely库中导入Polygon类，用于创建多边形对象

import concavity  # 导入concavity模块，假设是自定义的模块，用于计算凹度
# 从当前目录下的子模块中导入compute_alpha_shape、compute_bounding_box和compute_bounding_triangle函数
from .alpha_shape import compute_alpha_shape
from .bounding_box import compute_bounding_box
from .bounding_triangle import compute_bounding_triangle

'''

这段代码的作用是对给定的点集进行形状匹配，并提供一些工具函数来处理形状匹配过程中的各种计算和比较。

具体来说，这段代码包含以下几个函数：

compute_shape(points, alpha=None, k=None): 根据给定的点集计算形状。可以选择使用 alpha 参数进行 alpha 形状计算，也可以选择使用 k 参数进行基于 knn 的凹壳算法计算。如果同时设置了 alpha 和 k，则两种方法都将被使用，并且合并结果以找到边界点。

determine_non_fit_area(shape, basic_shape, max_error=None): 计算基本形状与给定形状之间的未匹配部分的面积。这个函数通常在计算完形状匹配后被调用，用于评估匹配的质量。

fit_basic_shape(shape, max_error=None, given_angles=None): 将形状与矩形和三角形进行比较，并返回最适合的基本形状（矩形或三角形）的多边形。如果给定了最大误差，则在满足条件的情况下返回形状，并指示基本形状是否适合得足够好。
'''

def compute_shape(points, alpha=None, k=None):
    """
    计算基于凹壳的一组点的形状。

    参数
    ----------
    points : (Mx2) 数组
        点的坐标。
    alpha : 浮点数
        设置为使用指定的alpha值计算alpha形状。如果同时设置了alpha和k，则两种方法都将被使用，并且合并结果以找到边界点。
    k : 整数
        设置为使用基于knn的凹壳算法，并使用指定数量的最近邻点。如果同时设置了alpha和k，则两种方法都将被使用，并且合并结果以找到边界点。

    返回
    -------
    shape : 多边形
        点的计算形状。
    """
    if alpha is not None:  # 如果alpha不为None，则使用alpha shape方法
        shape = compute_alpha_shape(points, alpha)  # 调用compute_alpha_shape函数计算alpha shape

        if k is not None:  # 如果k不为None，则继续使用knn算法
            # 调用concavity模块中的concave_hull函数计算凹壳，boundary_points存储凹壳边界点
            boundary_points = concavity.concave_hull(points, k, 1)
            # 使用shapely库中的Polygon类创建凹壳的多边形，并进行缓冲处理
            shape_ch = Polygon(boundary_points).buffer(0)
            shape = unary_union([shape, shape_ch])  # 对两个形状进行联合操作

        if type(shape) != Polygon:  # 如果形状不是多边形对象，说明形状由多个多边形组成
            shape = max(shape.geoms, key=lambda s: s.area)  # 选择面积最大的多边形作为整体形状

    elif k is not None:  # 如果alpha为None，但k不为None，则使用knn算法
        boundary_points = concavity.concave_hull(points, k, 1)  # 计算凹壳边界点
        shape = Polygon(boundary_points).buffer(0)  # 创建凹壳的多边形，并进行缓冲处理
    else:
        raise ValueError('Either k or alpha needs to be set.')  # 抛出数值错误，要求设置k或alpha

    return shape  # 返回计算出的形状


def determine_non_fit_area(shape, basic_shape, max_error=None):
    """
    计算基于凹壳的一组点的形状。

    参数
    ----------
    points : (Mx2) 数组
        点的坐标。
    alpha : 浮点数
        设置为使用指定的alpha值计算alpha形状。如果同时设置了alpha和k，则两种方法都将被使用，并且合并结果以找到边界点。
    k : 整数
        设置为使用基于knn的凹壳算法，并使用指定数量的最近邻点。如果同时设置了alpha和k，则两种方法都将被使用，并且合并结果以找到边界点。

    返回
    -------
    shape : 多边形
        点的计算形状。
    """
    diff = basic_shape - shape  # 计算基本形状与给定形状的差异
    if max_error is not None:  # 如果设置了最大误差
        diff = diff.buffer(-max_error)  # 对差异进行负缓冲处理
    print('non fit area: {}'.format(diff.area))  # 打印未匹配部分的面积
    return diff.area  # 返回未匹配部分的面积


def fit_basic_shape(shape, max_error=None, given_angles=None):
    """
    将形状与矩形和三角形进行比较。如果给定了最大误差，则在满足条件的情况下返回形状，并指示基本形状是否适合得足够好。

    参数
    ----------
    shape : 多边形
        点的形状。
    max_error : 浮点数，可选
        点与形状之间的最大误差（距离）。
    given_angles : 浮点数列表，可选
        如果设置，则在计算最小面积边界框时，将检查这些角度的最小面积边界框（而不是凸包所有边的角度）。

    返回
    -------
    basic_shape : 多边形
        最适合的基本形状（矩形或三角形）的多边形。
    basic_shape_fits : 布尔值
        如果找到的基本形状在给定的最大误差
    """
    convex_hull = ConvexHull(shape.exterior.coords)
    # 计算给定形状外部坐标的凸包，convex_hull存储凸包信息

    bounding_box = compute_bounding_box(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull,
        given_angles=given_angles,
        max_error=max_error
    )
    # 使用给定形状的外部坐标计算边界框，传递凸包信息、给定角度信息和最大误差

    bbox_non_fit_area = determine_non_fit_area(
        shape, bounding_box, max_error=max_error
    )
    # 计算形状与边界框之间的未匹配部分的面积，传递形状、边界框和最大误差

    if max_error is not None and bbox_non_fit_area <= 0:
        return bounding_box, True
    # 如果设置了最大误差且未匹配部分的面积小于等于0，则返回边界框及其匹配状态为True

    bounding_triangle = compute_bounding_triangle(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull
    )
    # 使用给定形状的外部坐标计算边界三角形，传递凸包信息

    tri_non_fit_area = determine_non_fit_area(
        shape, bounding_triangle, max_error=max_error
    )
    # 计算形状与边界三角形之间的未匹配部分的面积，传递形状、边界三角形和最大误差

    if max_error is not None and tri_non_fit_area <= 0:
        return bounding_triangle, True
    # 如果设置了最大误差且未匹配部分的面积小于等于0，则返回边界三角形及其匹配状态为True

    if bbox_non_fit_area < tri_non_fit_area:
        return bounding_box, False
    else:
        return bounding_triangle, False
    # 如果边界框的未匹配部分面积小于边界三角形的未匹配部分面积，则返回边界框及其匹配状态为False，否则返回边界三角形及其匹配状态为False
