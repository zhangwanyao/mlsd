import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import *
from shapely.ops import polygonize
import warnings
import rtree

# 定义高斯平滑函数，接受三个参数：coords 是一个点集的数组，sigma 是高斯核的标准差，默认为 2，num_points_factor 是控制插值密度的因子，默认为 2。
def gaussian_smooth(coords, sigma=2, num_points_factor=2):

    # 将输入的坐标数组转换为 NumPy 数组。
    coords = np.array(coords)

    # 将坐标数组转置，并分别存储 x 和 y 坐标。
    x, y = coords.T

    # 生成一个从 0 到 1 等间隔的数组，长度与输入点集数组的行数相同。
    xp = np.linspace(0, 1, coords.shape[0])

    # 生成一个更密集的从 0 到 1 等间隔的数组，长度是原数组长度乘以 num_points_factor。
    interp = np.linspace(0, 1,coords.shape[0] * num_points_factor)

    # 对 x 和 y 坐标进行插值，使得坐标点的密度增加。这里使用的是线性插值。
    x = np.interp(interp, xp, x)
    y = np.interp(interp, xp, y)

    # 对插值后的 x 和 y 坐标应用一维高斯滤波器进行平滑处理，sigma 参数控制平滑程度，mode='wrap' 表示边界模式为循环。这里 gaussian_filter1d 函数来自于 scipy.ndimage 模块。
    x = gaussian_filter1d(x, sigma, mode='wrap')
    y = gaussian_filter1d(y, sigma, mode='wrap')

    # 返回平滑处理后的 x 和 y 坐标数组。
    return x, y


# 定义高斯平滑几何图形函数，接受三个参数：geom 是一个 shapely 的 LineString、Polygon、MultiLineString 或 MultiPolygon 对象，sigma 是高斯核的标准差，默认为 2，num_points_factor 是控制插值密度的因子，默认为 2。
def gaussian_smooth_geom(geom, sigma=2, num_points_factor=2):
    """
    :param geom: shapely 的 LineString、Polygon、MultiLineString 或 MultiPolygon 对象
    :param sigma: 高斯核的标准差
    :param num_points_factor: 确定顶点密度的点数 - 分辨率
    :return: 平滑处理后的 shapely 几何图形
    """
    # 检查几何图形的类型，如果是 Polygon 或 LineString，则进行平滑处理
    if isinstance(geom, (Polygon, LineString)):
        # 对多边形外环坐标进行高斯平滑处理
        x, y = gaussian_smooth(geom.exterior.coords)

        # 如果几何图形是 Polygon 类型
        if type(geom) == Polygon:
            # 将处理后的坐标添加到 x 和 y 坐标数组末尾，并形成闭合环
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            # 如果几何图形有内部环
            if len(list(geom.interiors)) > 0:
                l = []
                l.append(list(zip(x, y)))

                # 对每个内部环坐标进行高斯平滑处理，并添加到列表中
                for interior in list(geom.interiors):
                    x, y = gaussian_smooth(interior)
                    l.append(list(zip(x, y)))
                # 创建多边形对象，将处理后的坐标列表展开后作为参数传递
                return Polygon([item for sublist in l for item in sublist])
            else:
                # 创建多边形对象，将处理后的坐标列表作为参数传递
                return Polygon(list(zip(x, y)))
        else:
            # 如果几何图形是 LineString 类型，则创建线对象，将处理后的坐标列表作为参数传递
            return LineString(list(zip(x, y)))

    # 如果几何图形是 MultiPolygon 或 MultiLineString 类型
    elif isinstance(geom, (MultiPolygon, MultiLineString)):
        list_ = []
        # 对每个几何对象进行递归调用高斯平滑几何图形函数
        for g in geom:
            list_.append(gaussian_smooth_geom(g, sigma, num_points_factor))

        # 如果几何图形是 MultiPolygon 类型，则创建多边形集合对象，将处理后的几何对象列表作为参数传递
        if type(geom) == MultiPolygon:
            return MultiPolygon(list_)
        else:
            # 如果几何图形是 MultiLineString 类型，则创建线集合对象，将处理后的几何对象列表作为参数传递
            return MultiLineString(list_)
    else:
        # 如果几何图形不属于以上类型，则发出警告并返回原始几何图形
        warnings.warn(
            'geometry must be LineString, Polygon, MultiLineString or MultiPolygon, returning original geometry')
        return geom


# 定义使多边形有效的函数，接受两个参数：geom 是一个几何图形对象，precision 是比较准确度的小数位数，默认为 2。
def make_valid_polygon(geom, precision=2):
    # 如果几何图形已经有效或者不是多边形或多部分多边形，则直接返回几何图形
    if geom.is_valid == True or isinstance(geom, (Polygon, MultiPolygon)) == False:
        return geom

    # 如果几何图形是多边形类型
    if isinstance(geom, Polygon):
        # 计算原始多边形的面积
        area = geom.area
        # 对原始多边形进行零缓冲处理
        buff_geom = geom.buffer(0)
        # 计算零缓冲处理后的多边形的面积
        buff_area = buff_geom.area
        # 如果原始多边形面积与缓冲后多边形面积的四舍五入值相等，则返回缓冲后的多边形
        if round(area, precision) == round(buff_area, precision):
            return buff_geom
        else:
            # 否则，对多边形的外环和内环进行交集操作，并重新构建多边形
            exterior = geom.exterior
            interiors = geom.interiors
            exterior = exterior.intersection(exterior)
            interiors = [MultiPolygon(polygonize(i.intersection(i)))[0].exterior for i in interiors]
            result = MultiPolygon([Polygon(i.exterior, interiors) for i in polygonize(exterior)])
            return result
    # 如果几何图形是多部分多边形类型
    elif isinstance(geom, MultiPolygon):
        # 对多部分多边形中的每个部分递归调用使多边形有效的函数
        result = [make_valid_polygon(poly) for poly in geom]
        # 将结果展平并创建多部分多边形对象
        result = MultiPolygon(flatten([[i for i in j] if isinstance(j, MultiPolygon) else j for j in result]))
        # 如果结果只有一个多边形，则返回该多边形
        if len(result) == 1:
            return result[0]
        else:
            # 否则返回多部分多边形对象
            return result



# 定义使多边形顺时针方向的函数，接受一个几何图形对象作为参数
def make_ccw(geom):
    # 如果几何图形不是多边形或多部分多边形类型，则发出警告并返回原始几何图形
    if isinstance(geom, (Polygon, MultiPolygon)) == False:
        warnings.warn("Geometry is not a Polygon or MultiPolygon - returning the original geometry")
        return geom

    # 如果几何图形是多边形类型
    if isinstance(geom, Polygon):
        # 如果多边形为空，则直接返回
        if geom.is_empty == True:
            return geom
        # 如果多边形外环是顺时针方向，则直接返回多边形
        if geom.exterior.is_ccw:
            return geom
        else:
            # 否则，将多边形外环坐标反转，并对内环进行相应操作，以确保内环是顺时针方向
            exterior = LinearRing(geom.exterior.coords[::-1])
            interiors = geom.interiors
            interiors = [LinearRing(i.coords[::-1]) if i.is_ccw == True else i for i in interiors]
            result = Polygon(exterior, interiors)
            return result
    # 如果几何图形是多部分多边形类型
    elif isinstance(geom, MultiPolygon):
        # 对多部分多边形中的每个部分递归调用使多边形顺时针方向的函数
        result = [make_ccw(poly) for poly in geom]
        # 将结果展平并创建多部分多边形对象
        result = MultiPolygon(flatten([[i for i in j] if isinstance(j, MultiPolygon) else j for j in result]))
        # 如果结果只有一个多边形，则返回该多边形
        if len(result) == 1:
            return result[0]
        else:
            # 否则返回多部分多边形对象
            return result


# 定义一个展平嵌套容器的生成器函数
def flatten(container):
    # 遍历容器中的每个元素
    for i in container:
        # 如果当前元素是列表或元组类型，则递归调用展平函数，并遍历展平后的结果
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            # 如果当前元素不是列表或元组类型，则直接生成该元素
            yield i




class ConcaveHull(object):
    def __init__(self,
                 coords,
                 knn,
                 increase_knn=1):
        # 初始化函数，接受一系列参数，其中 coords 是坐标数组，knn 是最近邻数量，increase_knn 是增加的最近邻数量，默认为 1

        # 将输入的参数分别存储起来
        self.coords_orig = coords
        self.knn_orig = knn
        self.increase_knn_orig = increase_knn
        # 调用 _concave_hull 方法进行凹多边形计算
        self._concave_hull(self.coords_orig, self.knn_orig, self.increase_knn_orig)

    def get_concave_hull(self):
        # 返回计算得到的凹多边形
        return self.concave_hull

    def _set_params(self, current_idx, knn, new_dir, current_point):
        # 设置参数的私有方法，用于计算当前点的最近邻信息及相关参数

        # 获取最近邻索引
        self.idx = np.array(list(self.rt_idx.nearest(self.c_bounds[current_idx], knn)))
        self.idx = self.idx[self.idx != current_idx]
        self.points = self.coords[self.idx]

        # 计算当前点与其邻居之间的方向
        self.dir_vecs = np.array([x - self.coords[current_idx] for x in self.points])
        self.dir_vecs = np.array([i / np.linalg.norm(i) for i in self.dir_vecs])

        # 计算当前方向与邻居之间的角度
        self.signs = np.array([np.sign(np.cross(new_dir, i)) for i in self.dir_vecs])
        self.angles = np.array([np.degrees(np.arccos(np.clip(np.dot(new_dir, i), -1.0, 1.0))) for i in self.dir_vecs])
        self.angles[self.signs == 1] = 360 - self.angles[self.signs == 1]
        self.angles = np.array([90 - x if x < 90 else 360 + 90 - x for x in self.angles])

        # 计算当前点与所有其他点之间的距离
        self.dists = np.array([np.linalg.norm(current_point - i) for i in self.points])

        # 创建一个按最小角度和最大距离排序的邻居队列
        self.s_points = np.array([self.angles, self.dists, self.idx, np.arange(0, self.idx.shape[0])]).T
        self.s_points = self.s_points[np.lexsort((-self.s_points[:, 1], self.s_points[:, 0]))]

        # 选择最顶部的邻居点作为候选点
        self.new_idx = int(self.s_points[0][2])

    def _concave_hull(self, coords, knn, increase_knn):
        # 计算凹多边形的私有方法，采用迭代的方式逐步构建凹多边形

        self.coords = np.array(coords)
        self.knn = knn
        self.increase_knn = increase_knn

        # 三角形情况
        if np.unique(self.coords, axis=0).shape[0] < 3:
            self.concave_hull = Polygon()
        elif np.unique(self.coords, axis=0).shape[0] == 3:
            self.concave_hull = MultiPoint(self.coords).convex_hull
        # 无法绘制凹形状
        elif self.knn == np.unique(self.coords, axis=1).shape[0] - 1:
            self.concave_hull = MultiPoint(self.coords).convex_hull
        else:
            # 处理边界情况
            if self.knn_orig >= self.coords.shape[0] - 1:
                self.knn = self.coords.shape[0] - 2

            # 用于 rtree 索引的边界数组
            self.c_bounds = np.hstack([self.coords, self.coords])

            # rtree 索引
            self.rt_idx = rtree.index.Index()
            [self.rt_idx.insert(i[0], (i[1])) for i in enumerate(self.c_bounds)]

            # 设置初始参数
            self.start_knn = self.knn
            self.line = []
            self.min_y = self.coords[:, 1].min()
            self.min_y_idx = np.where(self.coords[:, 1] == self.min_y)
            self.first_idx = self.min_y_idx[0][0]
            self.line.append(self.coords[self.first_idx])
            self.first_point = self.coords[self.first_idx]

            # 设置参数
            self._set_params(self.first_idx, self.knn, np.array([.0, 1.]), self.first_point)
            self.current_idx = self.new_idx
            self.current_point = self.coords[self.current_idx]
            self.new_dir = self.dir_vecs[int(self.s_points[0][3])]
            self.line.append(self.current_point)

            # 从 rtree 索引中删除第一个点
            self.rt_idx.delete(self.first_idx, self.c_bounds[self.first_idx])

            # 当第一个点重新添加到索引时的安全距离
            self.check_dist = True
            self.max_dist = max(self.dists)

            while self.current_idx != self.first_idx:
                self.test_int = True
                count = 0
                self.knn = self.start_knn

                # 设置队列
                self._set_params(self.current_idx, self.knn, self.new_dir, self.current_point)
                while self.test_int == True:

                    # 停止迭代
                    if self.new_idx == self.first_idx:
                        self.test_int = False
                    else:
                        # 设置新方向
                        self.new_dir = self.dir_vecs[int(self.s_points[count][3])]
                        count += 1

                        # 检查新线段是否与现有线段创建自交
                        l = LineString(self.line)
                        relate = l.relate(LineString([self.current_point,
                                                      self.coords[self.new_idx]]))

                        # 如果没有，停止内部循环并添加到线段中
                        if relate == 'FF1F00102':
                            self.test_int = False
                        else:
                            self.test_int = True
                            self.rt_idx.delete(self.new_idx, self.c_bounds[self.new_idx])

                            # 如果队列为空，则重新填充
                            if count == self.s_points.shape[0]:
                                self._set_params(self.current_idx, self.knn, self.new_dir, self.current_point)
                                count = 0

                # 从 rtree 索引中删除当前点
                self.rt_idx.delete(self.current_idx, self.c_bounds[self.current_idx])

                # 设置候选点作为当前点
                self.current_idx = self.new_idx
                self.current_point = self.coords[self.current_idx]

                # 达到安全距离后将第一个点添加到 r-tree 索引中
                if self.check_dist:
                    if np.linalg.norm(self.first_point - self.current_point) >= self.max_dist:
                        self.rt_idx.add(self.first_idx, self.c_bounds[self.first_idx])
                        self.check_dist = False

                # 将新点添加到线段中
                self.line.append(self.current_point)

            # 底部两种情况是由于某些点太远以及邻居数量太少而失败的
            # 重新计算，增加 knn
            if len(self.line) < 4:
                return self._concave_hull(self.coords_orig, self.knn + self.increase_knn, self.increase_knn)
            else:
                poly = Polygon(self.line)
                if poly.buffer(0.1).contains(MultiPoint(self.coords)) and poly.is_valid:
                    self.concave_hull = poly
                else:
                    return self._concave_hull(self.coords_orig, self.knn + self.increase_knn, self.increase_knn)
