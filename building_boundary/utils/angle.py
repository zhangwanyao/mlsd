# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

'''
min_angle_difference: 计算两个角度之间的最小角度差异。它将考虑到角度的周期性，确保计算结果在 0 到 π 之间。

angle_difference: 计算两个角度之间的角度差异。与第一个函数不同，它返回的差异值在 0 到 π 之间，不会考虑最小差异。

to_positive_angle: 将角度转换为正值。如果角度为负，则添加 π 以确保结果为正。

perpendicular: 给定一个角度，返回其垂直角度。对给定角度加上 π/2 来获取垂直角度，如果结果大于 π，则减去 π。
'''


def min_angle_difference(a1, a2):
    """
    返回两个方向之间的最小角度差异。

    Parameters
    ----------
    a1 : float
        弧度表示的一个角度
    a2 : float
        弧度表示的另一个角度

    Returns
    -------
    angle : float
        弧度表示的最小角度差异
    """
    # 计算两个角度的绝对值差异
    diff1 = abs(math.pi - abs(abs(a1 - a2) - math.pi))
    # 计算 a1 小于 pi 时的另一种角度差异
    if a1 < math.pi:
        diff2 = abs(math.pi - abs(abs((a1 + math.pi) - a2) - math.pi))
    # 计算 a2 小于 pi 时的另一种角度差异
    elif a2 < math.pi:
        diff2 = abs(math.pi - abs(abs(a1 - (a2 + math.pi)) - math.pi))
    else:
        return diff1
    # 返回最小角度差异
    return diff1 if diff1 < diff2 else diff2


def angle_difference(a1, a2):
    """
    返回两个方向之间的角度差异。

    Parameters
    ----------
    a1 : float
        弧度表示的一个角度
    a2 : float
        弧度表示的另一个角度

    Returns
    -------
    angle : float
        弧度表示的角度差异
    """
    # 计算绝对值差异，取模以确保结果在 0 到 2*pi 之间
    diff = abs(a1 - a2) % (2 * math.pi)
    # 计算角度差异
    angle = math.pi - abs(math.pi - diff)
    return angle

    # # 举例
    # angle1 = math.pi / 4  # 45度
    # angle2 = 3 * math.pi / 4  # 135度
    # diff = angle_difference(angle1, angle2)
    # print("角度差异：", diff)


def to_positive_angle(angle):
    """
    将角度转换为正值。

    Parameters
    ----------
    angle : float
        弧度表示的角度

    Returns
    -------
    angle : float
        正值角度
    """
    # 取模以确保角度在0到pi之间
    angle = angle % math.pi
    # 如果角度为负，则加上pi以确保它为正
    if angle < 0:
        angle += math.pi
    return angle


def perpendicular(angle):
    """
    返回给定角度的垂直角度。

    Parameters
    ----------
    angle : float or int
        给定的角度，以弧度表示

    Returns
    -------
    perpendicular : float
        给定角度的垂直角度，以弧度表示
    """
    # 计算垂直角度
    perp = angle + math.pi/2
    # 如果垂直角度大于pi，则减去pi
    if perp > math.pi:
        perp = angle - math.pi/2
    return perp

    # 示例
    angle = math.pi / 6  # 30度的角度，以弧度表示
    perpendicular_angle = perpendicular(angle)
    print("垂直角度：", perpendicular_angle)
    # 垂直角度： 2.356194490192345
