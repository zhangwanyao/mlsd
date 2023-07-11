# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

'''
用于计算一组点的最小面积定向边界框（Minimum Area Oriented Bounding Box）。以下是每个函数的作用：

compute_edge_angles: 计算边缘与 x 轴之间的角度。它通过计算每个边缘的斜率来确定每个边缘的角度。

rotate_points: 使用给定的角度旋转坐标系中的点。这个函数根据给定的角度使用旋转矩阵来旋转点。

rotating_calipers_bbox: 使用旋转卡尺算法计算定向边界框的顶点。它旋转点集合，找到使定向边界框最小的角度，并返回定向边界框的四个顶点。

check_error: 检查给定的边界框是否足够接近点集，基于给定的最大误差。它计算每个点到边界框的距离，并检查是否小于最大误差。

compute_bounding_box: 计算一组点的最小面积定向边界框。它首先计算点集的凸包，然后根据凸包的边缘和角度，使用旋转卡尺算法计算定向边界框的顶点。根据需要，还可以提供最大误差以进行容错处理。
'''
def compute_edge_angles(edges):
    """
    Compute the angles between the edges and the x-axis.

    Parameters
    ----------
    edges : (Mx2x2) array
        The coordinates of the sets of points that define the edges.

    Returns
    -------
    edge_angles : (Mx1) array
        The angles between the edges and the x-axis.
    """
    edges_count = len(edges)
    edge_angles = np.zeros(edges_count)
    for i in range(edges_count):
        edge_x = edges[i][1][0] - edges[i][0][0]
        edge_y = edges[i][1][1] - edges[i][0][1]
        edge_angles[i] = math.atan2(edge_y, edge_x)

    return np.unique(edge_angles)


def rotate_points(points, angle):
    """
    Rotate points in a coordinate system using a rotation matrix based on
    an angle.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    angle : float
        The angle by which the points will be rotated (in radians).

    Returns
    -------
    points_rotated : (Mx2) array
        The coordinates of the rotated points.
    """
    # Compute rotation matrix
    rot_matrix = np.array(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    # Apply rotation matrix to the points
    points_rotated = np.dot(points, rot_matrix)

    return np.array(points_rotated)


def rotating_calipers_bbox(points, angles):
    """
    Compute the oriented minimum bounding box using a rotating calipers
    algorithm.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points of the convex hull.
    angles : (Mx1) array-like
        The angles the edges of the convex hull and the x-axis.

    Returns
    -------
    corner_points : (4x2) array
        The coordinates of the corner points of the minimum oriented
        bounding box.
    """
    min_bbox = {'angle': 0,
                'minmax': (0, 0, 0, 0),
                'area': float('inf')}

    for a in angles:
        # Rotate the points and compute the new bounding box
        rotated_points = rotate_points(points, a)
        min_x = min(rotated_points[:, 0])
        max_x = max(rotated_points[:, 0])
        min_y = min(rotated_points[:, 1])
        max_y = max(rotated_points[:, 1])
        area = (max_x - min_x) * (max_y - min_y)

        # Save if the new bounding box is smaller than the current smallest
        if area < min_bbox['area']:
            min_bbox = {'angle': a,
                        'minmax': (min_x, max_x, min_y, max_y),
                        'area': area}

    # Extract the rotated corner points of the minimum bounding box
    c1 = (min_bbox['minmax'][0], min_bbox['minmax'][2])
    c2 = (min_bbox['minmax'][0], min_bbox['minmax'][3])
    c3 = (min_bbox['minmax'][1], min_bbox['minmax'][3])
    c4 = (min_bbox['minmax'][1], min_bbox['minmax'][2])
    rotated_corner_points = [c1, c2, c3, c4]

    # Rotate the corner points back to the original system
    corner_points = np.array(rotate_points(rotated_corner_points,
                                           2*math.pi-min_bbox['angle']))

    return corner_points


def check_error(points, bbox, max_error):
    """
    Checks if the given bounding box is close enough to the points based on
    the given max error.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    bbox : (4x2) array
        The coordinates of the vertices of the bounding box.
    max_error : float
        The maximum error (distance) a point can have to the bounding box.

    Returns
    -------
     : bool
        If all points are within the max error distance of the bounding box.
    """
    distances = [bbox.exterior.distance(Point(p)) for p in points]
    return all([d < max_error for d in distances])


def compute_bounding_box(points, convex_hull=None,
                         given_angles=None, max_error=None):
    """
    Computes the minimum area oriented bounding box of a set of points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    convex_hull : scipy.spatial.ConvexHull, optional
        The convex hull of the points, as computed by SciPy.
    given_angles : list of float, optional
        If set the minimum area bounding box of these angles will be checked
        (instead of the angles of all edges of the convex hull).
    max_error : float, optional
        The maximum error (distance) a point can have to the bounding box.

    Returns
    -------
    bbox : polygon
        The minimum area oriented bounding box as a shapely polygon.
    """
    if convex_hull is None:
        convex_hull = ConvexHull(points)
    hull_points = points[convex_hull.vertices]

    if given_angles is None:
        angles = compute_edge_angles(points[convex_hull.simplices])
    else:
        angles = given_angles

    bbox_corner_points = rotating_calipers_bbox(hull_points, angles)
    bbox = Polygon(bbox_corner_points)

    if max_error is not None and given_angles is not None:
        if not check_error(points, bbox, max_error):
            angles = compute_edge_angles(points[convex_hull.simplices])
            bbox_corner_points = rotating_calipers_bbox(hull_points, angles)
            bbox = Polygon(bbox_corner_points)

    return bbox
