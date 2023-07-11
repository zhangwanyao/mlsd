# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

import numpy as np
from building_boundary.utils.error import ThresholdError
from building_boundary.utils.angle import min_angle_difference
from building_boundary.utils.geometry import distance


'''

以上是一个名为 BoundarySegment 的类，用于表示边界线段对象。这个类具有以下功能：

初始化：根据给定的点，计算拟合的直线参数，并计算线段的端点、长度和方向。
计算直线的斜率和截距。
计算点到拟合线的距离以及点与拟合线之间的最大距离。
设置直线参数，并根据新参数重新创建线段。
计算线段与另一条线的交点。
判断线段上的点位于哪一侧。
'''


def PCA(points):
    """
    对给定的3D点集进行主成分分析（PCA），通过计算点云的协方差矩阵的特征值和特征向量来完成。

    Parameters
    ----------
    points : (Mx3) array
        点的X、Y和Z坐标。

    Returns
    -------
    eigenvalues : (1x3) array
        对应于协方差矩阵的特征值。
    eigenvectors : (3x3) array
        协方差矩阵的特征向量。
    """
    # 计算点云的协方差矩阵
    cov_mat = np.cov(points, rowvar=False)

    # 对协方差矩阵进行特征值分解，得到特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    # 对特征值按从大到小进行排序，并相应地调整特征向量的顺序
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors



class BoundarySegment(object):
    def __init__(self, points):
        """
        初始化一条边界线段对象。

        Parameters
        ----------
        points : (Mx2) array
            点的X和Y坐标。

        Attributes
        ----------
        points : (Mx2) array
            点的X和Y坐标。
        a : float
            线的a系数 (ax + by + c = 0)。
        b : float
            线的b系数 (ax + by + c = 0)。
        c : float
            线的c系数 (ax + by + c = 0)。
        end_points : (2x2) array
            线段的端点坐标。
        length : float
            线段的长度。
        orientation : float
            线段的方向（弧度制）。
        """
        self.points = points
        self.points = np.array(self.points).reshape(-1, 2)
        self.fit_line()

    def slope(self):
        """
        计算线的斜率。

        Returns
        -------
        float
            线的斜率。
        """
        return -self.a / self.b

    def intercept(self):
        """
        计算线的截距。

        Returns
        -------
        float
            线的截距。
        """
        return -self.c / self.b

    # @property 是 Python 中用于将一个方法转换为属性的装饰器。在这种情况下，@property 装饰器用于定义一个属性，使得可以像访问属性一样访问该方法的返回值。
    @property
    def line(self):
        """
        返回表示线的 a、b 和 c 系数（ax + by + c = 0）的元组。

        Returns
        -------
        tuple
            表示线的参数元组。
        """
        return (self.a, self.b, self.c)

    # @line.setter 是 Python 中用于定义属性 setter 方法的装饰器。它允许在给属性赋值时执行自定义的逻辑。
    # 在这个上下文中，@line.setter 装饰器定义了一个 setter 方法，用于设置线段的系数 a, b, 和 c，并在设置完成后调用 _create_line_segment() 方法来重新创建线段。
    @line.setter
    def line(self, line):
        """
        设置线的参数并重新创建线段。

        Parameters
        ----------
        line : (1x3) array-like
            表示线的 a、b 和 c 系数（ax + by + c = 0）。

        """
        self.a, self.b, self.c = line
        self._create_line_segment()

    def fit_line(self, max_error=None):
        """
        将线拟合到对象的点集。

        Parameters
        ----------
        max_error : float or int
            拟合线允许的最大误差（点到线的最大距离）。
            如果超过了此最大误差，则会引发 ThresholdError。

        Raises
        ------
        ThresholdError
            如果拟合线的误差（点到线的最大距离）超过给定的最大误差。
        """
        if len(self.points) == 1:
            raise ValueError('Not enough points to fit a line.')
        elif len(self.points) == 2:
            # 对于只有两个点的情况，通过计算直线的斜率和截距来拟合直线
            dx, dy = np.diff(self.points, axis=0)[0]
            if dx == 0:
                self.a = 0
            else:
                self.a = dy / dx
            self.b = -1
            self.c = (np.mean(self.points[:, 1]) -
                      np.mean(self.points[:, 0]) * self.a)
        elif all(self.points[0, 0] == self.points[:, 0]):
            # 如果所有点的 x 坐标相同，则拟合一条竖直线
            self.a = 1
            self.b = 0
            self.c = -self.points[0, 0]
        elif all(self.points[0, 1] == self.points[:, 1]):
            # 如果所有点的 y 坐标相同，则拟合一条水平线
            self.a = 0
            self.b = 1
            self.c = -self.points[0, 1]
        else:
            # 对于其他情况，使用主成分分析（PCA）来拟合直线
            _, eigenvectors = PCA(self.points)
            self.a = eigenvectors[1, 0] / eigenvectors[0, 0]
            self.b = -1
            self.c = (np.mean(self.points[:, 1]) -
                      np.mean(self.points[:, 0]) * self.a)

            if max_error is not None:
                error = self.error()
                if error > max_error:
                    raise ThresholdError(
                        "Could not fit a proper line. Error: {}".format(error)
                    )

        self._create_line_segment()

    def _point_on_line(self, point):
        """
        从另一点找到拟合线上最接近的点。

        Parameters
        ----------
        point : (1x2) array
            一个点的X和Y坐标。

        Returns
        -------
        point : (1x2) array
            给定点在拟合线上最接近点的X和Y坐标。

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation  # noqa
        """
        if self.a == 0 and self.b == 0:
            raise ValueError('Invalid line. Line coefficients a and b '
                             '(ax + by + c = 0) cannot both be zero.')

        # 计算点到拟合线上最近点的坐标
        x = (self.b * (self.b * point[0] - self.a * point[1]) -
             self.a * self.c) / (self.a ** 2 + self.b ** 2)
        y = (self.a * (-self.b * point[0] + self.a * point[1]) -
             self.b * self.c) / (self.a ** 2 + self.b ** 2)
        return [x, y]

    def _create_line_segment(self):
        """
        通过创建端点、长度和方向来定义拟合线的线段。

        Raises
        ------
        ValueError
            如果没有足够的点来创建线段。
        """
        if len(self.points) == 1:
            raise ValueError('Not enough points to create a line.')
        else:
            # 计算线段的起始点和结束点
            start_point = self._point_on_line(self.points[0])
            end_point = self._point_on_line(self.points[-1])

            # 计算线段的长度和方向
            self.end_points = np.array([start_point, end_point])
            dx, dy = np.diff(self.end_points, axis=0)[0]
            self.length = math.hypot(dx, dy)
            self.orientation = math.atan2(dy, dx)

    def error(self):
        """
        计算点与拟合线之间的最大距离。

        Returns
        -------
        error : float
            点与拟合线之间的最大距离。
        """
        # 计算每个点到拟合线的距离
        self.dist_points_line()

        # 返回距离中的最大值
        return max(abs(self.distances))

    def dist_points_line(self):
        """
        计算每个点到拟合线的距离。

        Attributes
        ----------
        distances : (1xN) array
            每个点到拟合线的距离。

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        # 计算每个点到拟合线的距离
        self.distances = (abs(self.a * self.points[:, 0] +
                              self.b * self.points[:, 1] + self.c) /
                          math.sqrt(self.a ** 2 + self.b ** 2))

    def dist_point_line(self, point):
        """
        计算给定点到拟合线的距离。

        Parameters
        ----------
        point : (1x2) array
            点的 X 和 Y 坐标。

        Returns
        -------
        dist : float
            给定点到拟合线的距离。

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        # 计算给定点到拟合线的距离
        dist = (abs(self.a * point[0] + self.b * point[1] + self.c) /
                math.sqrt(self.a ** 2 + self.b ** 2))
        return dist

    def target_orientation(self, primary_orientations):
        """
        确定给定的主要方向中哪一个与此线段的方向最接近。

        Parameters
        ----------
        primary_orientations : list of float
            已确定的主要方向列表。

        Returns
        -------
        orientation : float
            最接近此线段方向的主要方向。
        """
        # 计算此线段方向与所有主要方向之间的最小角度差异
        po_diff = [min_angle_difference(self.orientation, o) for
                   o in primary_orientations]
        # 找到最小的角度差异
        min_po_diff = min(po_diff)
        # 返回与最小角度差异相对应的主要方向
        return primary_orientations[po_diff.index(min_po_diff)]

    def regularize(self, orientation, max_error=None):
        """
        根据给定的方向重新创建线段。

        Parameters
        ----------
        orientation : float or int
            线段应具有的方向。以弧度表示，从0到pi（逆时针方向的东到西），从0到-pi（顺时针方向的东到西）。
        max_error : float or int
            允许拟合线段的最大误差（点到线的最大距离）。
            如果超过了此最大误差，则会引发ThresholdError异常。

        Raises
        ------
        ThresholdError
            如果拟合线段的误差（点到线的最大距离）超过了给定的最大误差。

        .. [1] https://math.stackexchange.com/questions/1377716/how-to-find-a-least-squares-line-with-a-known-slope  # noqa
        """
        prev_a = self.a
        prev_b = self.b
        prev_c = self.c

        # 如果给定的方向与当前方向不一致，则重新计算线段参数
        if not np.isclose(orientation, self.orientation):
            if np.isclose(abs(orientation), math.pi / 2):
                # 如果方向为pi/2或-pi/2，则线段为竖直线
                self.a = 1
                self.b = 0
                self.c = np.mean(self.points[:, 0])
            elif (np.isclose(abs(orientation), math.pi) or
                  np.isclose(orientation, 0)):
                # 如果方向为0或pi，则线段为水平线
                self.a = 0
                self.b = 1
                self.c = np.mean(self.points[:, 1])
            else:
                # 否则根据给定的方向重新计算线段的参数
                self.a = math.tan(orientation)
                self.b = -1
                self.c = (sum(self.points[:, 1] - self.a * self.points[:, 0]) /
                          len(self.points))

            # 如果指定了最大误差，则检查拟合线段的误差是否超过了最大误差
            if max_error is not None:
                error = self.error()
                if error > max_error:
                    # 如果超过了最大误差，则恢复先前的线段参数，并引发ThresholdError异常
                    self.a = prev_a
                    self.b = prev_b
                    self.c = prev_c
                    raise ThresholdError(
                        "Could not fit a proper line. Error: {}".format(error)
                    )

            # 根据新的线段参数重新创建线段
            self._create_line_segment()

    def line_intersect(self, line):
        """
        计算此线段与另一条线的交点。

        Parameters
        ----------
        line : (1x3) array-like
            另一条线的a、b和c系数（ax + by + c = 0）。

        Returns
        -------
        point : (1x2) array
            交点的坐标。如果没有找到交点，则返回空数组。
        """
        a, b, c = line  # 获取另一条线的系数
        d = self.a * b - self.b * a  # 计算行列式的值
        if d != 0:
            dx = -self.c * b + self.b * c  # 计算交点的x坐标
            dy = self.c * a - self.a * c  # 计算交点的y坐标
            x = dx / float(d)
            y = dy / float(d)
            return np.array([x, y])  # 返回交点的坐标
        else:
            return np.array([])  # 如果行列式的值为0，表示两条线平行，返回空数组表示没有交点

    def side_point_on_line(self, point):
        """
        确定线段上的点位于哪一侧。

        Parameters
        ----------
        point : (1x2) array
            点的X和Y坐标，该点位于此线段的直线上。

        Returns
        -------
        side : int
            如果点在线段上，则返回0；
            如果点在起始点一侧，则返回1；
            如果点在终点一侧，则返回-1。
        """
        a = self.end_points[0]  # 获取线段的起点
        b = self.end_points[1]  # 获取线段的终点
        c = point  # 待判断的点
        if not np.isclose(np.cross(b - a, c - a), 0):
            # 如果点不在直线上，则抛出异常
            raise ValueError('Given point not on line.')

        dist_ab = distance(a, b)  # 计算线段的长度
        dist_ac = distance(a, c)  # 计算起始点到待判断点的距离
        dist_bc = distance(b, c)  # 计算终点到待判断点的距离
        if np.isclose(dist_ac + dist_bc, dist_ab):
            return 0  # 如果点在线段上，则返回0
        elif dist_ac < dist_bc:
            return 1  # 如果点在起始点一侧，则返回1
        elif dist_bc < dist_ac:
            return -1  # 如果点在终点一侧，则返回-1

