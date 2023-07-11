# -*- coding: utf-8 -*-
"""
这段代码用于多边形的膨胀，以确保其包含所有给定点。
@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from building_boundary import utils
from .segment import BoundarySegment  # 导入 BoundarySegment 类

'''
这段代码用于多边形的膨胀，以确保其包含所有给定点。具体作用如下：

point_on_line_segment(line_segment, point): 判断点是否在由两个点定义的线段上。
point_polygon_edges(point_on_polygon, edges): 确定点（位于多边形外部）位于多边形的哪些边上。
inflate_polygon(vertices, points): 膨胀多边形以包含所有给定的点。
'''


def point_on_line_segment(line_segment, point):
    """
    判断点是否在由两个点定义的线段上。

    参数
    ----------
    line_segment : (2x2) 数组
        由两个点坐标定义的线段。
    point : (1x2) 数组
        要检查的点的坐标。

    返回
    -------
     : 布尔值
        点是否在由两个点定义的线段上。 c
    """
    a = line_segment[0]  # 第一个点
    b = line_segment[1]  # 第二个点
    p = point  # 要检查的点
    if not np.isclose(np.cross(b-a, p-a), 0):  # 判断点是否在直线上
        return False
    else:
        dist_ab = utils.geometry.distance(a, b)  # 计算线段的长度
        dist_ap = utils.geometry.distance(a, p)  # 计算点到第一个点的距离
        dist_bp = utils.geometry.distance(b, p)  # 计算点到第二个点的距离
        if np.isclose(dist_ap + dist_bp, dist_ab):  # 判断点是否在线段上
            return True
        else:
            return False


def point_polygon_edges(point_on_polygon, edges):
    """
    确定点（位于多边形外部）位于多边形的哪些边上。

    参数
    ----------
    point_on_polygon : (1x2) 数组
        点的坐标。
    edges : 包含元组的列表，元组的元素是 (1x2) 数组
        每个元组包含多边形边的起点和终点坐标。

    返回
    -------
    found_edges : 整数列表
        给定点位于的边的索引。通常为长度为1，但如果点位于角落上，则位于两条边上，将返回两个索引。
    """
    found_edges = []
    for i, e in enumerate(edges):
        if point_on_line_segment(e, point_on_polygon):  # 判断点是否在每条边上
            found_edges.append(i)
    return found_edges


def inflate_polygon(vertices, points):
    """
    膨胀多边形以包含所有给定的点。

    参数
    ----------
    vertices : (Mx2) 数组
        多边形顶点的坐标。
    points : (Mx2) 数组
        多边形应包含的点的坐标。

    返回
    -------
    vertices : (Mx2) 数组
        膨胀多边形的顶点坐标。
    """
    new_vertices = vertices.copy()  # 复制多边形顶点坐标
    points_geom = [Point(p) for p in points]  # 创建点的几何对象
    n_vertices = len(vertices)  # 多边形顶点数量
    polygon = Polygon(vertices)  # 创建多边形对象
    edges = utils.create_pairs(new_vertices)  # 创建多边形边的起点和终点坐标列表

    # 找到不在多边形内部的点
    distances = np.array([polygon.distance(p) for p in points_geom])  # 计算每个点到多边形的距离
    outliers_mask = np.invert(np.isclose(distances, 0))  # 找到不在多边形内的点的布尔掩码
    outliers = points[outliers_mask]  # 不在多边形内的点
    distances = distances[outliers_mask]  # 不在多边形内的点到多边形的距离
    n_outliers = len(outliers)  # 不在多边形内的点的数量

    while n_outliers > 0:
        p = outliers[np.argmax(distances)]  # 选择距离最远的不在多边形内的点

        # 找到距离点最近的多边形边
        point_on_polygon, _ = nearest_points(polygon, Point(p))  # 找到距离点最近的多边形边
        point_on_polygon = np.array(point_on_polygon)  # 多边形边上的点的坐标
        nearest_edges = point_polygon_edges(point_on_polygon, edges)  # 确定点位于的多边形边的索引

        # 将多边形边移出以确保点
        for i in nearest_edges:
            delta = p - point_on_polygon  # 计算移动的距离
            p1 = new_vertices[i] + delta  # 计算移动后的第一个顶点
            p2 = new_vertices[(i+1) % n_vertices] + delta  # 计算移动后的第二个顶点
            # 创建线段对象
            l1 = BoundarySegment(np.array([new_vertices[(i-1) % n_vertices],
                                           new_vertices[i]]))
            l2 = BoundarySegment(np.array([p1, p2]))
            l3 = BoundarySegment(np.array([new_vertices[(i+1) % n_vertices],
                                           new_vertices[(i+2) % n_vertices]]))
            # 计算交点
            new_vertices[i] = l2.line_intersect(l1.line)
            new_vertices[(i+1) % n_vertices] = l2.line_intersect(l3.line)

            # 更新多边形
            polygon = Polygon(new_vertices)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # 修复多边形的无效性
                new_vertices = np.array(polygon.exterior.coords)  # 更新顶点坐标
                n_vertices = len(new_vertices)  # 更新顶点数量
                n_outliers = float('inf')  # 设置为无穷大，以便继续循环

            edges = utils.create_pairs(new_vertices)  # 更新多边形边的起点和终点坐标列表
            point_on_polygon, _ = nearest_points(polygon, Point(p))  # 找到距离点最近的多边形边
            point_on_polygon = np.array(point_on_polygon)  # 多边形边上的点的坐标

        distances = np.array([polygon.distance(p) for p in points_geom])  # 重新计算每个点到多边形的距离
        outliers_mask = np.invert(np.isclose(distances, 0))  # 找到不在多边形内的点的布尔掩码
        outliers = points[outliers_mask]  # 不在多边形内的点
        distances = distances[outliers_mask]  # 不在多边形内的点到多边形的距离

        if len(outliers) >= n_outliers:
            break
        n_outliers = len(outliers)

    if not Polygon(new_vertices).is_valid and Polygon(vertices).is_valid:
        return vertices
    else:
        return new_vertices
