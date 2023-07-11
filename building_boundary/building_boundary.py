# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""
import sys
import os
sys.path.append(os.path.abspath('./building_boundary'))
# print(sys.path)
import numpy as np
from shapely.geometry import Polygon
from shapes.fit import compute_shape, fit_basic_shape
from core.segment import BoundarySegment
from core.segmentation import boundary_segmentation
from core.merge import merge_segments
from core.intersect import compute_intersections
from core.regularize import get_primary_orientations, regularize_segments
from core.inflate import inflate_polygon
from building_boundary.utils.angle import perpendicular


def trace_boundary(points, ransac_threshold, max_error=None, alpha=None,
                   k=None, num_points=None, angle_epsilon=0.05,
                   merge_distance=None, primary_orientations=None,
                   perp_dist_weight=3, inflate=False):
    """
    Trace the boundary of a set of 2D points.

    points：(Mx2) 数组，点的坐标。
    ransac_threshold：浮点数，RANSAC 线拟合过程中被分类为内点的数据点的最大距离。
    max_error：浮点数，点与计算线之间的最大误差（距离）。
    alpha：浮点数，设置使用 alpha 形状来确定边界点，使用指定的 alpha。如果同时设置了 alpha 和 k，则会同时使用两种方法，并合并生成的形状以找到边界点。
    k：整数，使用基于 k 近邻的凹多边形算法来确定边界点，使用指定数量的最近邻。如果同时设置了 alpha 和 k，则会同时使用两种方法，并合并生成的形状以找到边界点。
    num_points：整数，可选参数，支持一段线段所需的点数，以被视为主要方向。如果手动设置了主要方向，则会忽略此参数。
    angle_epsilon：浮点数，两个角度被视为相同的角度差异（以弧度表示）。用于合并线段。
    merge_distance：浮点数，如果两个平行的连续线段之间的距离（基于角度 epsilon）低于此值，则线段将被合并。
    primary_orientations：浮点数列表，可选参数，边界的期望主要方向（）以弧度表示。如果在此处手动设置了方向，则不会计算这些方向。
    perp_dist_weight：浮点数，用于计算线段之间的交点。如果两个线段的交点与线段的距离大于 perp_dist_weight 倍的垂直线段交点与线段之间的距离，则将使用垂直交点。
    inflate：布尔值，如果设置为 True，则拟合线将移动到最远处的外点。
    Returns
    -------
    vertices : (Mx2) array
        The vertices of the computed boundary line
    """
    shape = compute_shape(points, alpha=alpha, k=k)
    # 使用给定的点集计算形状，根据参数alpha和k来选择算法

    boundary_points = np.array(shape.exterior.coords)
    # 提取形状的外部边界点坐标，并转换为NumPy数组
    basic_shape, basic_shape_fits = fit_basic_shape(
        shape,
        max_error=max_error,
        given_angles=primary_orientations,
    )
    # 计算形状的基本形状，如矩形或三角形，并检查其是否适合于给定的最大误差

    if max_error is not None and basic_shape_fits:
        return np.array(basic_shape.exterior.coords)
    # 如果设置了最大误差并且基本形状适合，则返回基本形状的外部边界坐标

    segments = boundary_segmentation(boundary_points, ransac_threshold)
    # 对形状的边界点进行分割，以获得线段

    if len(segments) in [0, 1, 2]:
        return np.array(basic_shape.exterior.coords)
    # 如果边界线段的数量为0、1或2，则返回基本形状的外部边界坐标

    boundary_segments = [BoundarySegment(s) for s in segments]
    # 将分割得到的线段转换为BoundarySegment对象的列表

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(
            boundary_segments,
            num_points,
            angle_epsilon=angle_epsilon
        )
    # 如果没有指定主方向或主方向列表为空，则根据参数获取主方向

    if len(primary_orientations) == 1:
        primary_orientations.append(
            perpendicular(primary_orientations[0])
        )
    # 如果只有一个主方向，则添加一个垂直于该方向的主方向

    boundary_segments = regularize_segments(boundary_segments,
                                            primary_orientations,
                                            max_error=max_error)
    # 对边界线段进行规范化，使其与主方向对齐

    boundary_segments = merge_segments(boundary_segments,
                                       angle_epsilon=angle_epsilon,
                                       max_distance=merge_distance,
                                       max_error=max_error)
    # 合并接近的边界线段

    vertices = compute_intersections(boundary_segments,
                                     perp_dist_weight=perp_dist_weight)
    # 计算边界线段的交点，根据垂直距离权重

    if inflate:
        remaining_points = boundary_segments[0].points
        for s in boundary_segments[1:]:
            remaining_points = np.vstack((remaining_points, s.points))
        vertices = inflate_polygon(vertices, remaining_points)
    # 如果指定了inflate参数，则对计算得到的交点进行膨胀操作

    polygon = Polygon(vertices)
    # 创建多边形对象，使用交点坐标构造多边形

    if not polygon.is_valid:
        return np.array(basic_shape.exterior.coords)
    # 如果多边形不是有效的几何形状，则返回基本形状的外部边界坐标

    if (len(boundary_segments) == len(basic_shape.exterior.coords) - 1 and
            basic_shape.area < polygon.area):
        return np.array(basic_shape.exterior.coords)
    # 如果边界线段的数量等于基本形状的边界点数减1，并且基本形状的面积小于多边形的面积，则返回基本形状的外部边界坐标

    return vertices
    # 返回计算得到的多边形的顶点坐标


def trace_boundary_face_points(boundary_points,  max_error=None,num_points=None, angle_epsilon=0.05,
                   merge_distance=None, primary_orientations=None,
                   perp_dist_weight=3, inflate=False):
    """
    Trace the boundary of a set of 2D points.

    points：(M*n x 2) 数组，点的坐标。
    max_error：浮点数，点与计算线之间的最大误差（距离）。
    num_points：整数，可选参数，支持一段线段所需的点数，以被视为主要方向。如果手动设置了主要方向，则会忽略此参数。
    angle_epsilon：浮点数，两个角度被视为相同的角度差异（以弧度表示）。用于合并线段。
    merge_distance：浮点数，如果两个平行的连续线段之间的距离（基于角度 epsilon）低于此值，则线段将被合并。
    primary_orientations：浮点数列表，可选参数，边界的期望主要方向（）以弧度表示。如果在此处手动设置了方向，则不会计算这些方向。
    perp_dist_weight：浮点数，用于计算线段之间的交点。如果两个线段的交点与线段的距离大于 perp_dist_weight 倍的垂直线段交点与线段之间的距离，则将使用垂直交点。
    inflate：布尔值，如果设置为 True，则拟合线将移动到最远处的外点。
    Returns
    -------
    vertices : (Mx2) array
        The vertices of the computed boundary line
    """
    boundary_segments = [BoundarySegment(np.array(s)) for s in boundary_points]
    # 将分割得到的线段转换为BoundarySegment对象的列表

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(
            boundary_segments,
            num_points,
            angle_epsilon=angle_epsilon
        )
    # 如果没有指定主方向或主方向列表为空，则根据参数获取主方向

    if len(primary_orientations) == 1:
        primary_orientations.append(
            perpendicular(primary_orientations[0])
        )
    # 如果只有一个主方向，则添加一个垂直于该方向的主方向

    boundary_segments = regularize_segments(boundary_segments,
                                            primary_orientations,
                                            max_error=max_error)
    # # 对边界线段进行规范化，使其与主方向对齐
    #
    boundary_segments = merge_segments(boundary_segments,
                                       angle_epsilon=angle_epsilon,
                                       max_distance=merge_distance,
                                       max_error=max_error)
    # 合并接近的边界线段

    vertices = compute_intersections(boundary_segments,
                                     perp_dist_weight=perp_dist_weight)
    # 计算边界线段的交点，根据垂直距离权重

    if inflate:
        remaining_points = boundary_segments[0].points
        for s in boundary_segments[1:]:
            remaining_points = np.vstack((remaining_points, s.points))
        vertices = inflate_polygon(vertices, remaining_points)
    # 如果指定了inflate参数，则对计算得到的交点进行膨胀操作

    polygon = Polygon(vertices)
    # 创建多边形对象，使用交点坐标构造多边形

    return vertices
    # 返回计算得到的多边形的顶点坐标