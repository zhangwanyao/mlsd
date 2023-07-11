import math  # 导入数学库

import numpy as np  # 导入NumPy库，用于处理数组和数值计算
from shapely import unary_union  # 从Shapely库中导入unary_union函数，用于对几何图形进行联合操作
from shapely.geometry import Polygon, MultiPolygon  # 从Shapely库中导入Polygon和MultiPolygon类，用于创建多边形对象

try:
    from CGAL.CGAL_Kernel import Point_2  # 尝试导入CGAL库中的Point_2类
    from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2, REGULAR  # 尝试导入CGAL库中的Alpha_shape_2类和REGULAR常量
    CGAL_AVAILABLE = True  # 标记CGAL是否可用
    cascaded_union = None  # 如果CGAL可用，则将cascaded_union设置为None
    Delaunay = None  # 如果CGAL可用，则将Delaunay设置为None
except ImportError:
    from shapely.ops import cascaded_union  # 如果CGAL库未安装，则从Shapely库中导入cascaded_union函数
    from scipy.spatial import Delaunay  # 如果CGAL库未安装，则从SciPy库中导入Delaunay类
    CGAL_AVAILABLE = False  # 标记CGAL不可用
    Point_2 = None  # 如果CGAL库未安装，则将Point_2设置为None
    Alpha_shape_2 = None  # 如果CGAL库未安装，则将Alpha_shape_2设置为None
    REGULAR = None  # 如果CGAL库未安装，则将REGULAR设置为None


'''
这段代码实现了计算一组点的alpha形状（或凹壳）的功能，其中包括两种实现方式：一种是使用CGAL库，另一种是使用Python代码。

alpha_shape_cgal 函数：使用CGAL库计算一组点的alpha形状。该函数首先将输入点转换为CGAL库的 Point_2 对象，然后创建 Alpha_shape_2 对象，并设置alpha值。接下来，通过迭代 alpha 形状的边，将边的端点添加到一个列表中，然后根据这些端点构建多边形，并对多边形进行联合操作，最后返回计算得到的 alpha 形状。

alpha_shape_python 函数：使用Python代码计算一组点的alpha形状。该函数使用 Delaunay 三角剖分算法对输入点进行三角化，然后计算每个三角形的面积和外接圆半径，根据给定的 alpha 值筛选符合条件的三角形，并将这些三角形的外部边界联合起来，最后返回计算得到的 alpha 形状。

compute_alpha_shape 函数：根据是否安装了 CGAL 库选择使用哪种方法计算点集的 alpha 形状。如果安装了 CGAL 库，则使用 alpha_shape_cgal 函数计算 alpha 形状；否则，使用 alpha_shape_python 函数计算 alpha 形状。
'''


def alpha_shape_cgal(points, alpha):
    """
    使用CGAL计算一组点的alpha形状（凹壳）。
    Alpha形状不包含任何内部区域。

    参数
    ----------
    points : (Mx2) 数组
        点的x和y坐标。
    alpha : 浮点数
        影响Alpha形状的形状。较高的值会导致更多的三角形被删除。

    返回
    -------
    alpha_shape : 多边形
        计算得到的Alpha形状，作为Shapely多边形。
    """
    points_cgal = [Point_2(*p) for p in points]  # 将点转换为CGAL库的Point_2对象

    as2 = Alpha_shape_2(points_cgal, 0, REGULAR)  # 创建Alpha_shape_2对象
    as2.set_alpha(alpha)  # 设置alpha值

    edges = []  # 存储Alpha形状的边
    for e in as2.alpha_shape_edges():  # 迭代Alpha形状的边
        segment = as2.segment(e)  # 获取边的线段
        edges.append([[segment.vertex(0).x(), segment.vertex(0).y()],
                      [segment.vertex(1).x(), segment.vertex(1).y()]])  # 将边的端点添加到edges列表
    edges = np.array(edges)  # 将边列表转换为NumPy数组

    e1s = edges[:, 0].tolist()  # 提取所有边的第一个端点
    e2s = edges[:, 1].tolist()  # 提取所有边的第二个端点
    polygons = []  # 存储形成的多边形

    while len(e1s) > 0:  # 当还有边未处理时
        polygon = []  # 存储一个多边形的顶点
        current_point = e2s[0]  # 获取当前处理的点
        polygon.append(current_point)  # 将当前点添加到多边形中
        del e1s[0]  # 删除e1s中的第一个元素
        del e2s[0]  # 删除e2s中的第一个元素

        while True:  # 迭代查找多边形的其他顶点
            try:
                i = e1s.index(current_point)  # 查找下一个点在e1s中的索引
            except ValueError:  # 如果找不到下一个点
                break  # 跳出循环

            current_point = e2s[i]  # 获取下一个点
            polygon.append(current_point)  # 将下一个点添加到多边形中
            del e1s[i]  # 删除e1s中对应的点
            del e2s[i]  # 删除e2s中对应的点

        polygons.append(polygon)  # 将形成的多边形添加到polygons列表

    polygons = [Polygon(p) for p in polygons if len(p) > 2]  # 创建多边形对象，要求至少有3个点

    alpha_shape = MultiPolygon(polygons).buffer(0)  # 创建MultiPolygon对象，并进行缓冲处理

    return alpha_shape  # 返回计算得到的Alpha形状


def triangle_geometry(triangle):
    """
    计算三角形的面积和外接圆半径。

    参数
    ----------
    triangle : (3x3) 数组样式
        形成三角形的点的坐标。

    返回
    -------
    area : 浮点数
        三角形的面积
    circum_r : 浮点数
        三角形的外接圆半径
    """
    pa, pb, pc = triangle  # 提取三角形的三个顶点
    # 计算三角形的边长
    a = math.hypot((pa[0] - pb[0]), (pa[1] - pb[1]))
    b = math.hypot((pb[0] - pc[0]), (pb[1] - pc[1]))
    c = math.hypot((pc[0] - pa[0]), (pc[1] - pa[1]))
    # 计算三角形的半周长
    s = (a + b + c) / 2.0
    # 使用海伦公式计算三角形的面积
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    if area != 0:
        # 计算三角形的外接圆半径
        circum_r = (a * b * c) / (4.0 * area)
    else:
        circum_r = 0
    return area, circum_r # 返回三角形的面积和外接圆半径


def alpha_shape_python(points, alpha):
    """
    计算点的alpha形状（或凹壳）。

    参数
    ----------
    points : (Mx2) 数组
        点的坐标。
    alpha : 浮点数
        影响Alpha形状的形状。较高的值会导致更多的三角形被删除。

    返回
    -------
    alpha_shape : 多边形
        计算得到的Alpha形状，作为Shapely多边形。
    """
    triangles = []  # 存储三角形的列表
    tri = Delaunay(points)  # 使用Delaunay三角剖分算法进行三角化
    for t in tri.simplices:  # 迭代三角化的结果
        area, circum_r = triangle_geometry(points[t])  # 计算三角形的面积和外接圆半径
        if area != 0:
            if circum_r < (1.0 / alpha + 5000):  # 根据alpha值筛选三角形
                triangles.append(Polygon(points[t]))  # 将符合条件的三角形添加到列表中

    # 将所有三角形联合在一起，并确保结果为MultiPolygon
    alpha_shape = unary_union(triangles)
    if type(alpha_shape) == MultiPolygon:  # 如果结果是MultiPolygon
        alpha_shape = MultiPolygon([Polygon(s.exterior) for s in triangles])  # 提取每个多边形的外部边界
    elif type(alpha_shape) == Polygon:
        alpha_shape = Polygon(alpha_shape.exterior)  # 如果结果是单个Polygon，提取其外部边界
    else:
        polygons = [geom for geom in alpha_shape.geoms if isinstance(geom, Polygon)]
        if polygons:
            return polygons[0]

    return alpha_shape  # 返回计算得到的Alpha形状


def compute_alpha_shape(points, alpha):
    """
    计算点的alpha形状（或凹壳）。

    参数
    ----------
    points : (Mx2) 数组
        点的坐标。
    alpha : 浮点数
        影响Alpha形状的形状。较高的值会导致更多的三角形被删除。

    返回
    -------
    alpha_shape : 多边形
        计算得到的Alpha形状，作为Shapely多边形。
    """
    if len(points) < 4:  # 如果点的数量小于4，无法计算Alpha形状
        raise ValueError('Not enough points to compute an alpha shape.')  # 抛出数值错误
    if CGAL_AVAILABLE:  # 如果CGAL库可用
        alpha_shape = alpha_shape_cgal(points, alpha)  # 使用CGAL计算Alpha形状
    else:  # 如果CGAL库不可用
        alpha_shape = alpha_shape_python(points, alpha)  # 使用Python代码计算Alpha形状
    return alpha_shape  # 返回计算得到的Alpha形状

