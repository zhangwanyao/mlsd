#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/1 下午2:39
# @Author : WanYao Zhang
import math
from functional_domain.subsidiary import distance
import numpy as np

class Line:
    def __init__(self, start_point, end_point):
        self.start = start_point
        self.end = end_point
        self.middle = (np.array(start_point) + np.array(end_point)) / 2
        self.idx = 0
        self.num = 0
        self.ave = float('inf')
        self.angle = self.line_angle()

    def line_angle(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.angle = math.degrees(math.atan2(dy, dx))
        if self.angle < 0:
            self.angle += 180
        return self.angle

    def update_line_if_needed(self,thred = 6):
        if abs(self.angle) == 0 or abs(self.angle - 180) == 0 or abs(self.angle - 90) == 0:
            return self
        if abs(self.angle) < thred or abs(self.angle - 180) < thred:
            parallel_start = self.middle + np.array([-10, 0])
            parallel_end = self.middle + np.array([10, 0])
        elif (90 - thred) < self.angle < (90 + thred):
            parallel_start = self.middle + np.array([0, -10])
            parallel_end = self.middle + np.array([0, 10])
        else:
            return None

        self.start = self.projection_onto_line(self.start, parallel_start, parallel_end)
        self.end = self.projection_onto_line(self.end, parallel_start, parallel_end)
        self.angle = self.line_angle()

        return self

    @staticmethod
    def projection_onto_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_point = line_start + proj_length * line_unitvec
        return proj_point
def projection_onto_line(point, line):
    line_start, line_end = line.start, line.end
    line_vec_x = line_end[0] - line_start[0]
    line_vec_y = line_end[1] - line_start[1]
    point_vec_x = point[0] - line_start[0]
    point_vec_y = point[1] - line_start[1]
    line_len = distance(line_start, line_end)
    line_unitvec_x = line_vec_x / line_len
    line_unitvec_y = line_vec_y / line_len
    proj_length = point_vec_x * line_unitvec_x + point_vec_y * line_unitvec_y
    proj_point_x = line_start[0] + proj_length * line_unitvec_x
    proj_point_y = line_start[1] + proj_length * line_unitvec_y
    return proj_point_x, proj_point_y

def merge_lines(lines):
    points = []
    num_lines = len(lines)
    angles = 0
    for line in lines:
        angles += line.angle
        points.extend([line.start, line.end])
    angles_ave = angles / num_lines
    max_distance = 0
    new_start, new_end = None, None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = distance(points[i], points[j])
            if d > max_distance:
                max_distance = d
                new_start, new_end = points[i], points[j]
    if angles_ave < 45 or angles_ave > 135:
        # 接近0或180度的线段
        target_angle = 0 if angles_ave < 45 else 180
        closest_line = min(lines,key=lambda line: abs(line.angle - target_angle))
        new_start = Line.projection_onto_line(new_start, closest_line.start,closest_line.end)
        new_end = Line.projection_onto_line(new_end, closest_line.start,closest_line.end)
    elif 55 < angles_ave < 115:
        # 接近 90 度的线段
        closest_line = min(lines, key=lambda line: abs(line.angle - 90))
        new_start = Line.projection_onto_line(new_start, closest_line.start, closest_line.end)
        new_end = Line.projection_onto_line(new_end, closest_line.start, closest_line.end)
    return Line(new_start, new_end)

def point_to_line_dist(point, line_start, line_end):
    """
    计算点到线段的最短距离
    :param point: 点的坐标 (numpy array)
    :param line_start: 线段起点坐标 (numpy array)
    :param line_end: 线段终点坐标 (numpy array)
    :return: 点到线段的最短距离 (float)
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        print("存在长度为0的线段")
        dist = np.linalg.norm(point_vec)
        return dist
    else:
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = line_vec * t
        dist = np.linalg.norm(point_vec - nearest)
        return dist

def line_to_line_dist(start1, end1, start2, end2):
    """计算两条线段之间的最短距离"""
    dist1 = point_to_line_dist(start1, start2, end2)
    dist2 = point_to_line_dist(end1, start2, end2)
    dist3 = point_to_line_dist(start2, start1, end1)
    dist4 = point_to_line_dist(end2, start1, end1)
    return min(dist1, dist2, dist3, dist4)

def are_lines_collinear(line1, line2,line1_point, line2_point):
    # if abs(line1.angle) <10 or abs(line1.angle) >170: # 平行于x轴
    if abs(line1.angle) == 0 or abs(line1.angle) == 180:  # 平行于x轴
        value = abs(line1_point[1] - line2_point[1])
    # elif abs(line1.angle - 90) < 10: # 平行于y轴
    elif abs(line1.angle - 90) == 0:  # 平行于y轴
        value = abs(line1_point[0] - line2_point[0])
    else:
        value = 10
    if value < 3:
        return True
    else:
        return False


def intersect_point(line1, line2):
    xdiff = (line1.start[0] - line1.end[0], line2.start[0] - line2.end[0])
    ydiff = (line1.start[1] - line1.end[1], line2.start[1] - line2.end[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not intersect')

    d = (det(line1.start, line1.end), det(line2.start, line2.end))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])

def closest_point(line1, line2):
    start1, end1 = line1.start, line1.end
    start2, end2 = line2.start, line2.end
    distances = [
        (distance(end1, start2), (end1,1), (start2,0)),  # 第一条线段的末端与第二条线段的起点
        (distance(end1, end2), (end1,1), (end2,1)),  # 第一条线段的末端与第二条线段的末端
        (distance(start1, start2), (start1,0), (start2,0)),  # 第一条线段的起点与第二条线段的起点
        (distance(start1, end2), (start1,0), (end2,1))  # 第一条线段的起点与第二条线段的末端
    ]
    min_distance, (point1,idx1), (point2,idx2) = min(distances, key=lambda x: x[0])

    return idx1,idx2,point1,point2

def extend_line_to_perpendicular(line, point):
    p = np.array(point)
    a = line.start
    b = line.end
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    new_point = a + t * ab
    return new_point,Line(point, new_point)


def line_to_line(lines):
    line_i = 0
    while line_i < len(lines):
        line1 = lines[line_i]
        line2 = lines[(line_i + 1) % len(lines)]

        # if abs(line1.angle - line2.angle) < 5 or abs(line1.angle - line2.angle) > 175:
        if abs(line1.angle - line2.angle) == 0 or abs(line1.angle - line2.angle) == 180:
            idx1, idx2, line1_point, line2_point = closest_point(line1, line2)
            if are_lines_collinear(line1, line2,line1_point, line2_point): #接近共线合并
                # points = []
                # points.extend([line1.start, line1.end,line2.start,line2.end])
                # max_distance = 0
                # for i in range(len(points)):
                #     for j in range(i + 1, len(points)):
                #         d = distance(points[i], points[j])
                #         if d > max_distance:
                #             max_distance = d
                #             new_start, new_end = points[i], points[j]
                if line1.angle == 0 or line1.angle == 180:
                    new_midlle_y = (line2.start[1] + line1.start[1])/2
                    max_x = max(line1.start[0], line1.end[0],line2.start[0],line2.end[0])
                    min_x = min(line1.start[0], line1.end[0],line2.start[0],line2.end[0])
                    new_start = [min_x, new_midlle_y]
                    new_end = [max_x, new_midlle_y]
                else:
                    new_midlle_x = (line2.start[0] + line1.start[0])/2
                    max_y = max(line1.start[1], line1.end[1], line2.start[1], line2.end[1])
                    min_y = min(line1.start[1], line1.end[1], line2.start[1], line2.end[1])
                    new_start = [new_midlle_x, min_y]
                    new_end = [new_midlle_x, max_y]
                # Merge lines
                new_line = Line(new_start, new_end)
                lines[line_i] = new_line
                lines.pop((line_i + 1) % len(lines))
            else:
                # if line_i != len(lines) - 1:
                new_point,perp_line = extend_line_to_perpendicular(line1, line2_point) # 延伸线段到垂线
                if idx1 == 0:
                    lines[line_i] = Line(new_point,line1.end)
                elif idx1 == 1:
                    lines[line_i] = Line(line1.start,new_point)
                lines.insert(line_i + 1, perp_line)
                line_i += 1
        elif abs(line1.angle - line2.angle) == 90:
            try:
                idx1,idx2,line1_point,line2_point = closest_point(line1, line2)
                inter_point = intersect_point(line1, line2)
                if idx1 == 0:
                    lines[line_i] = Line(line1.end,inter_point)
                elif idx1 == 1:
                    lines[line_i] = Line(line1.start, inter_point)
                if idx2 == 0:
                    lines[(line_i + 1) % len(lines)] = Line(inter_point, line2.end)
                elif idx2 == 1:
                    lines[(line_i + 1) % len(lines)] = Line(inter_point,line2.start)
            except Exception as e:
                idx1,idx2,line1_point,line2_point = closest_point(line1, line2)
                new_point, perp_line = extend_line_to_perpendicular(line1, line2_point)
                lines.insert(line_i + 1, perp_line)
                line_i += 1

        line_i += 1
    return lines


def classification_doors_windows(Pseudo_walls,walls):
    '''
    param Pseudo_walls:list[array2*2]
    param walls:list[start、end、angle]
    '''
    lines = []
    doors = []
    windows = []

    for pseudo_wall in Pseudo_walls:
        line = Line(pseudo_wall[0], pseudo_wall[1])
        lines.append(line)
    points = [point for wall in walls for point in [wall.start,wall.end]]
    for line in lines:
        line_distance = distance(line.start,line.end)
        if line_distance > 400:
            continue
        elif 400 > line_distance > 100 or line_distance < 55:
            windows.append([line.start, line.end])
        else:
            left = False
            right = False
            up = False
            down = False
            if line.angle == 90:
                y_max = max(line.start[1], line.end[1]) + 60
                y_min = min(line.start[1], line.end[1]) - 60
                for point in points:
                    if y_min < point[1] < y_max:
                        if point[0] < line.start[0] - 20:
                            left = True
                        elif point[0] > line.end[0] + 20:
                            right = True
                if left and right:
                    doors.append([line.start,line.end])
                else:
                    windows.append([line.start,line.end])
            else:
                x_max = max(line.start[0], line.end[0]) + 60
                x_min = min(line.start[0], line.end[0]) - 60
                for point in points:
                    if x_min < point[0] < x_max:
                        if point[1] < line.start[1] - 20:
                            down = True
                        elif point[1] > line.end[1] + 20:
                            up = True
                if down and up:
                    doors.append([line.start, line.end])
                else:
                    windows.append([line.start, line.end])
    return doors, windows