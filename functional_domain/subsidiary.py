#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/1 下午2:27
# @Author : WanYao Zhang
import cv2
import numpy as np
import math

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_difference(line1, line2):
    def line_angle(line):
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        return math.atan2(dy, dx)
    angle1 = math.degrees(line_angle(line1))
    if angle1 < 0:
        angle1 += 180
    angle2 = math.degrees(line_angle(line2))
    if angle2 < 0:
        angle2 += 180
    return abs(angle1 - angle2)

# Perform Canny edge detection and find contours
def find_max_contour(image, threshold1=50, threshold2=200):
    '''
    Find the edge points of the outline to prevent unexpected situations and look for the largest edge points。
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        if len(max_contour) <= 100:
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea)

    else:
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
        else:
            max_contour = contours[0]
    return max_contour