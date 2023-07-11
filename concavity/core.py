import math
import numpy as np
from concavity.utils import make_ccw, ConcaveHull

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