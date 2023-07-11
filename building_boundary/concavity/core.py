import math

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from shapely.geometry import *

from concavity.utils import make_ccw, ConcaveHull

'''
concave_hull(coords, knn, increase_knn=1): 这个函数用于计算给定点集的凹多边形。它接受点的坐标和最近邻点的数量作为输入，并返回一个 Shapely 多边形对象，表示计算得到的凹多边形。

process_vertices(coords, angle_threshold, filter_type, convolve, get_convex, output_type): 这个函数用于处理顶点。它接受点集、角度阈值、过滤类型、是否平滑角度、是否获取凸点、输出类型等参数作为输入，并返回满足条件的顶点索引和对应的角度。

find_concave_vertices(geom, angle_threshold=0, filter_type='peak', convolve=False, get_convex=False, output_type='geopandas'): 这个函数用于寻找多边形外部和内部的凹顶点。它接受一个 Shapely 多边形或多多边形作为输入，并根据指定的条件查找凹顶点。根据输出类型的不同，它可以返回一个 GeoPandas DataFrame 或一个点和角度的元组列表。

find_convex_vertices(geom, angle_threshold, filter_type, convolve=False, get_concave=False, output_type='geopandas'): 这个函数用于寻找多边形外部和内部的凸顶点。它接受一个 Shapely 多边形或多多边形作为输入，并根据指定的条件查找凸顶点。根据输出类型的不同，它可以返回一个 GeoPandas DataFrame 或一个点和角度的元组列表。

plot_concave_hull(coords, concave_poly, figsize=(10, 10)): 这个函数用于绘制凹多边形外壳。它接受计算凹多边形外壳时使用的点的坐标、输出的凹多边形以及图形尺寸作为输入，并绘制包含凹多边形和原始点的图形。

plot_vertices(geom, concave_vertices_df=None, convex_vertices_df=None, figsize=(10, 10)): 这个函数用于绘制多边形的凹/凸顶点。它接受一个多边形、凹顶点数据和凸顶点数据以及图形尺寸作为输入，并绘制包含多边形、凹顶点和凸顶点的图形。
'''

# 定义一个函数，用于计算给定点集的凹多边形
def concave_hull(coords, knn, increase_knn=1):
    """
    :param coords: np.ndarray, shapely coordinate sequence
        用于计算凹多边形的点的坐标
    :param knn: int
        最近邻点的数量，确定凹边界的复杂程度
    :param increase_knn: int
        如果上一次迭代失败，则增加 k 的数量

    :return: shapely Polygon
    """
    # 根据 knn 的值自动调整增加的 knn 的数量
    increase_knn = math.ceil(knn / 20)
    # 调用 make_ccw 函数，确保点是逆时针方向的
    return make_ccw(ConcaveHull(coords, knn, increase_knn).concave_hull)

# 定义一个函数，用于处理顶点
def process_vertices(coords, angle_threshold, filter_type, convolve, get_convex, output_type):
    # 计算向量
    vecs = np.array([coords[x + 1] - coords[x] for x in range(coords.shape[0] - 1)])
    # 单位向量
    unit_vecs = np.array([i / np.linalg.norm(i) for i in vecs])

    # 计算角度
    signs = np.array( \
        [np.sign(np.cross(unit_vecs[-1], unit_vecs[0]))] + \
        [np.sign(np.cross(unit_vecs[x], unit_vecs[x + 1])) \
         for x in range(0, unit_vecs.shape[0] - 1)])

    degs = np.array([np.degrees(np.arccos(np.clip(np.dot(unit_vecs[-1], unit_vecs[0]), -1.0, 1.0)))] + \
                    [np.degrees(np.arccos(np.clip(np.dot(unit_vecs[x], unit_vecs[x + 1]), -1.0, 1.0))) \
                     for x in range(0, unit_vecs.shape[0] - 1)]
                    )

    degs = signs * degs

    # 平滑角度
    if convolve:
        cov_array = np.array([0.125, 0.225, 0.3, 0.225, 0.125])
        degs = np.convolve(degs, cov_array, mode='same')

    # 如果要获取凸点，则反转角度
    if get_convex == False:
        degs = -degs

    # 根据不同的过滤类型筛选顶点
    if filter_type == 'peak':
        peaks, _ = find_peaks(np.append(degs , degs), height=angle_threshold)
        peaks = np.where(peaks>=degs.shape[0], peaks - degs.shape[0], peaks)
        peaks = np.unique(peaks)
    else:
        peaks = np.where(degs > angle_threshold)[0]

    return peaks, degs


def find_concave_vertices(geom, angle_threshold=0, filter_type='peak', convolve=False, get_convex=False,
                          output_type='geopandas'):
    """
    寻找多边形外部和内部的凹顶点

    :param geom: shapely Polygon or MultiPolygon
        要查找凹顶点的多边形
    :param angle_threshold: number
        两个顶点之间的角度阈值，低于该值的顶点将不被考虑
    :param filter_type: str
        {'all', 'peak'}
        是否过滤所有高于 ``angle_threshold`` 的顶点或定位峰值顶点，默认为 'peak'
    :param convolve: boolean
        是否平滑角度（用于更细的峰值检测），默认为 False
    :param get_convex: boolean
        是否获取凸点而不是凹点，默认为 False
    :param output_type: str
        {"geopandas" , "list"}
        geopandas dataframe（默认）或点和角度的列表

    :return: geopandas dataframe（默认）或点和角度的元组
    """

    # 初始化空的凹顶点数据结构，根据输出类型选择创建 DataFrame 或列表
    if output_type == 'geopandas':
        concave_points = gpd.GeoDataFrame()
    else:
        concave_points = [[], []]

    # 如果输入几何对象是 MultiPolygon，则递归处理每个 Polygon
    if isinstance(geom, MultiPolygon):
        if output_type == 'geopandas':
            frames = []
            # 遍历每个 Polygon，并将结果添加到列表中
            for g in geom:
                frames.append(
                    find_concave_vertices(g, angle_threshold, filter_type, convolve, get_convex, output_type))

            # 将所有结果合并为一个 DataFrame
            concave_points = pd.concat(frames)
            return concave_points
        else:
            points = []
            angles = []
            # 遍历每个 Polygon，并将结果合并到一个列表中
            for g in geom:
                pts, angs = find_concave_vertices(g, angle_threshold, filter_type, convolve, get_convex, output_type)
                points += pts
                angles += angs
            return points, angles
    else:
        # 确保多边形是逆时针方向的
        geom = make_ccw(geom)

        # 遍历每个环（内部和外部），找到凹顶点
        for coord_seq in [np.array(i.coords) for i in geom.interiors] + [np.array(geom.exterior.coords)]:
            # 处理每个环的顶点，找到凹顶点
            peaks, degs = process_vertices(coord_seq, angle_threshold, filter_type, convolve, get_convex, output_type)

            # 根据输出类型处理凹顶点结果
            if output_type == 'geopandas':
                # 将凹顶点的几何信息和角度添加到 DataFrame 中
                concave_points = pd.concat([concave_points,
                                            gpd.GeoDataFrame({'geometry': [Point(i) for i in coord_seq[:-1][peaks]],
                                                              'angle': degs[peaks]})])
            else:
                # 将凹顶点的几何信息和角度添加到列表中
                concave_points[0] += [Point(i) for i in coord_seq[:-1][peaks]]
                concave_points[1] += degs[peaks].tolist()

        return concave_points



def find_convex_vertices(geom, angle_threshold, filter_type, convolve=False, get_concave=False, output_type='geopandas'):
    """
    寻找多边形外部和内部的凸顶点

    :param geom: shapely Polygon or MultiPolygon
        要查找凸顶点的多边形
    :param angle_threshold: number
        两个顶点之间的角度阈值，低于该值的顶点将不被考虑
    :param filter_type: str
        {'all', 'peak'}
        是否过滤所有高于 ``angle_threshold`` 的顶点或定位峰值顶点，默认为 'peak'
    :param convolve: boolean
        是否平滑角度（用于更细的峰值检测），默认为 False
    :param get_convex: boolean
        是否获取凸点而不是凹点，默认为 False
    :param output_type: str
        {"geopandas" , "list"}
        geopandas dataframe（默认）或点和角度的列表

    :return: geopandas dataframe（默认）或点和角度的元组
    """
    # 如果需要获取凹点，则调用 find_concave_vertices 函数
    if get_concave == False:
        get_convex = True
    else:
        get_convex = False

    return find_concave_vertices(geom, angle_threshold, filter_type, convolve, get_convex, output_type)



def plot_concave_hull(coords, concave_poly, figsize=(10, 10)):
    """
    绘制凹多边形外壳的函数

    :param coords: numpy.ndarray, shapely coordinate sequence
        计算凹多边形外壳时使用的点
    :param concave_poly: shapely Polygon
        输出的凹多边形
    :param figsize: tuple
        图形尺寸
    :return: None
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    legend_elements = []

    # 绘制凸多边形
    gpd.GeoSeries([MultiPoint(coords).convex_hull]).plot(ax=ax,
                                                         color='w',
                                                         edgecolor='#5ac5bc',
                                                         linewidth=5)
    legend_elements += [Patch(alpha=0.5,
                              facecolor='w',
                              edgecolor='#5ac5bc',
                              linewidth=2,
                              label='convex polygon')]

    # 绘制凹多边形
    gpd.GeoSeries([concave_poly]).plot(ax=ax,
                                       alpha=0.5,
                                       color='#ee65a3',
                                       edgecolor='r',
                                       linewidth=5)

    legend_elements += [Patch(alpha=0.5,
                              facecolor='#ee65a3',
                              edgecolor='r',
                              linewidth=2,
                              label='concave polygon')]

    # 绘制点
    gpd.GeoSeries([Point(c) for c in coords]).plot(ax=ax,
                                                   color='purple')

    legend_elements += [Line2D([0], [0], markersize=8, color='w',
                               markerfacecolor='purple',
                               marker='o',
                               label='points')]

    ax.legend(handles=legend_elements)
    plt.show()



def plot_vertices(geom, concave_vertices_df=None, convex_vertices_df=None, figsize=(10, 10)):
    """
    绘制多边形的凹/凸顶点

    :param geom: shapely Polygon
        计算凹/凸顶点的多边形
    :param concave_vertices_df: geopandas GeoDataFrame
        输出的凹顶点，默认为 None
    :param convex_vertices_df: geopandas GeoDataFrame
        输出的凸顶点，默认为 None
    :param figsize: tuple
        图形大小
    :return: None
    """
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    legend_elements = []

    # 绘制多边形
    gpd.GeoSeries([geom]).plot(ax=ax, color='#ee65a3', alpha=0.5, edgecolor='r')

    # 如果有凹顶点数据，则绘制凹顶点
    if concave_vertices_df is not None:
        concave_vertices_df.plot(ax=ax, color='#5ac5bc')
        legend_elements += [Line2D([0], [0], markersize=8, color='w',
                                   markerfacecolor='#5ac5bc',
                                   marker='o',
                                   label='concave vertices')]

    # 如果有凸顶点数据，则绘制凸顶点
    if convex_vertices_df is not None:
        convex_vertices_df.plot(ax=ax, color='purple')
        legend_elements += [Line2D([0], [0], markersize=8, color='w',
                                   markerfacecolor='purple',
                                   marker='o',
                                   label='convex vertices')]

    ax.legend(handles=legend_elements)
    plt.show()
