#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/11 下午2:46
# @Author : WanYao Zhang
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines, pred_squares
import argparse
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import building_boundary

# Argument parser for model path and input size
parser = argparse.ArgumentParser('M-LSD demo')
parser.add_argument('--model_path', default='tflite_models/M-LSD_512_large_fp32.tflite', type=str,
                    help='path to tflite model')
parser.add_argument('--input_size', default=512, type=int, choices=[512, 320], help='input size')
args = parser.parse_args()

# Load tflite model
interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

class creat_line:
    def __init__(self,start_point,end_point,middle_point):
        self.start = start_point
        self.end = end_point
        self.middle = middle_point
        self.num = 0
        self.idx = 0
        self.ave = float('inf')
# 匹配直线端点和凸包点
def match_line_to_boundary(boundary_points, detected_lines, threshold=5):
    # boundary_points = np.array([(p[1], p[0]) for p in boundary_points])
    matched_lines = []
    for line in detected_lines:
        start, end = line
        middle = (start + end) / 2
        matched_lines.append(creat_line(start,end,middle))
    for idx,point in enumerate(boundary_points):
        middle_dist = [distance(point,line.middle) for line in matched_lines]
        closest_dists = np.argsort(middle_dist)[:2]  # 获取最小的两个索引

        if idx == 0:
            # 处理最小的两个索引点
            angles = []
            for closest_dist in closest_dists:
                angle = np.degrees(np.arctan2(matched_lines[closest_dist].middle[1] - point[1], matched_lines[closest_dist].middle[0] - point[0]))
                angles.append(angle)
            if angles[0] > angles[1]:
                if middle_dist[closest_dists[0]] < threshold:
                        matched_lines[closest_dists[0]].idx += idx
                        matched_lines[closest_dists[0]].num += 1
                if middle_dist[closest_dists[1]] < threshold:
                        matched_lines[closest_dists[1]].idx += 99
                        matched_lines[closest_dists[1]].num += 1
            else:
                if middle_dist[closest_dists[0]] < threshold:
                        matched_lines[closest_dists[0]].idx += 999999
                        matched_lines[closest_dists[0]].num += 1
                if middle_dist[closest_dists[1]] < threshold:
                        matched_lines[closest_dists[1]].idx += idx
                        matched_lines[closest_dists[1]].num += 1
        else:
            for closest_dist in closest_dists:
                if middle_dist[closest_dist] < threshold:
                        matched_lines[closest_dist].idx += idx
                        matched_lines[closest_dist].num += 1

    for line in matched_lines:
        if line.num != 0:
            line.ave = line.idx/line.num
        else:
            line.ave = float('inf')
        print(line.ave)
    matched_lines = sorted(matched_lines, key=lambda l: l.ave)

    return matched_lines

# Function to sort points in clockwise order
def sort_points_clockwise(points, centroid):
    # Calculate angles of each point with respect to the centroid
    angles = {}
    for point in points:
        diff = point - centroid
        angle = math.atan2(diff[1], diff[0])
        angles[tuple(point)] = angle if angle >= 0 else 2 * math.pi + angle

    # Sort points based on angles
    sorted_points = sorted(points, key=lambda p: angles[tuple(p)])

    return sorted_points

# Function to sort line segments based on proximity
def sort_neighborhood(LineParams):
    # 通过选择最近的线段来构建一个排序列表，使得每个线段的终点或起点尽可能接近下一个线段的起点或终点
    # Convert LineParams to numpy array for easier manipulation
    LineParams = np.array(LineParams)

    # Initialize a list to store sorted elements
    sorted_elements = [LineParams[0]]  # Start with the first element

    # Remove the starting element from LineParams
    LineParams = np.delete(LineParams, 0, axis=0)

    while len(LineParams) > 0:
        # Extract the end point of the last sorted element
        last_end_point = sorted_elements[-1][1]  # End point of the last sorted element
        last_start_point= sorted_elements[-1][0]
        # Initialize variables to store the closest start point and its corresponding distance
        closest_idx = None
        min_distance =9999


        # Compute distances between each start point and all other start points
        for idx, line in enumerate(LineParams) :
            for pt in line:
                distance = np.linalg.norm(last_end_point[:2] - pt[:2])
                distance2 = np.linalg.norm(last_start_point[:2] - pt[:2]) # Compute distance ignoring Z axis
                distance = min(distance,distance2)
                # Update closest start point if a closer one is found
                if distance < min_distance:
                    closest_idx = idx
                    min_distance = distance

        # Append the corresponding element to the sorted list
        sorted_elements.append(LineParams[closest_idx])

        # Remove the appended element from LineParams
        LineParams = np.delete(LineParams, closest_idx, axis=0)

    return sorted_elements

# Function to plot lines with numbers
def plot_lines_with_numbers(LineParams, image_height):
    """
    绘制带有编号的线条，y 轴坐标按照 OpenCV 方式翻转
    :param LineParams: 包含线条起点和终点坐标的列表
    :param image_height: 图像的高度，用于翻转 y 坐标
    """
    # Sort LineParams
    sorted_lines = sort_neighborhood(LineParams)

    # Plot each line segment and annotate with its index
    for idx, line in enumerate(sorted_lines):
        start_point = line[0][:2]
        end_point = line[1][:2]

        # 翻转 y 坐标
        start_point_flipped = (start_point[0], image_height - start_point[1])
        end_point_flipped = (end_point[0], image_height - end_point[1])

        # 绘制线段
        plt.plot([start_point_flipped[0], end_point_flipped[0]],
                 [start_point_flipped[1], end_point_flipped[1]],
                 label=f"Line {idx + 1}")

        # 标注线段编号
        plt.text((start_point_flipped[0] + end_point_flipped[0]) / 2,
                 (start_point_flipped[1] + end_point_flipped[1]) / 2,
                 str(idx + 1), horizontalalignment='center', verticalalignment='center')

    # Set plot title and legend
    plt.title('Lines with numbers')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    plt.grid(True)
    plt.show()

# Function to detect lines and trace boundaries
def process_for_LSD(img_path, score_thr, dist_thr):
    img_input = cv2.imread(img_path)
    if img_input is None:
        raise FileNotFoundError(f"Unable to load image at path: {img_path}")
    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[args.input_size, args.input_size], score_thr=score_thr, dist_thr=dist_thr)

    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    # Display the result
    # plt.imshow(gray,origin='upper')
    # plt.title("Processed Image")
    # plt.axis('off')  # Hide axes
    # plt.show()
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    # 使用 cv2.findContours 找到顺序连接的边缘点
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_points = contours[0].reshape(-1,2)
    # 4. 找到所有的边缘点（边缘值大于 0）
    # points = np.argwhere(edges > 0)
    # plt.figure(figsize=(10, 10))
    # plt.plot(points[:, 1], points[:, 0], 'o', label='edges Points')
    # plt.gca().invert_yaxis()
    # plt.show()
    # plt.figure(figsize=(10, 10))
    # plt.plot(contours_points[:, 0], contours_points[:, 1], 'o', label='contours Points')
    # plt.gca().invert_yaxis()
    # plt.show()
    detected_lines = []
    for idx, line in enumerate(lines):
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        start_point = np.array([x_start, y_start])
        end_point = np.array([x_end, y_end])
        detected_lines.append([start_point, end_point])
    # 匹配直线与凸包点
    matched_lines = match_line_to_boundary(contours_points, detected_lines, threshold=10)

    # 定义颜色映射
    colors = cm.get_cmap('tab20', len(matched_lines))  # 使用 'tab20' colormap 生成 20 种颜色

    # 绘制凸包边界
    plt.figure(figsize=(10, 10))
    # plt.plot(contours_points[:, 0], contours_points[:, 1], 'o', label='Random Points')  # 绘制所有点
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r--', lw=2, label='Convex Hull')  # 绘制凸包

    boundary_points_new = []
    # 绘制匹配后的直线段，并用不同颜色区分，显示编号
    for idx, line in enumerate(matched_lines):
        start, end = line.start, line.end
        boundary_points_new.append([start, end])
        color = colors(idx / len(matched_lines))  # 选择颜色
        plt.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2, label=f'Line {idx + 1}' if idx == 0 else "")

        # 在线段的中点处标注编号
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        plt.text(mid_x, mid_y, str(idx + 1), fontsize=12, ha='center', va='center', color=color)

    # 设置图形属性
    plt.title('Matched Lines with Convex Hull Points and Line Numbers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    img_output = img_input.copy()
    height, width = img_output.shape[:2]
    blank_image = np.full((height, width, 3), 255, dtype=np.uint8)
    print(f"Number of detected lines: {len(lines)}")


    # # 线段分割及排序
    # points = []
    # LineParams = []
    # # Draw lines on the image
    # for line in lines:
    #     x_start, y_start, x_end, y_end = [int(val) for val in line]
    #     points.append([x_start, y_start])
    #     points.append([x_end, y_end])
    #     LineParams.append([[x_start,y_start],[x_end,y_end]])
    #     cv2.line(blank_image, (x_start, y_start), (x_end, y_end), [0, 255, 255], 1)
    #
    # # 排序
    # center = np.mean(np.array(points), axis=0)
    # for idx,lineParam in enumerate(LineParams) :
    #     LineParams[idx]=sort_points_clockwise(lineParam, center)
    #
    # # 可视化过程结果
    # plot_lines_with_numbers(LineParams,height)
    # LineParams = sort_neighborhood(LineParams)
    #
    # # 数据格式转换
    # boundary_points_new = []
    # for idx, pts in enumerate(np.array(LineParams)):
    #     a = pts.tolist()[0][:2]
    #     b = pts.tolist()[1][:2]
    #     boundary_points_new.append([a, b])
    # # boundary_points = np.array([[[6272.994315602144, 2769.517578125], [6283.751909304447, 995.2520141601562]], [[6286.72216796875, 962.3297378014122], [7690.58349609375, 966.1963684957117]], [[7869.31005859375, 966.1767322108095], [8438.46875, 967.6881996419082]], [[8541.34145180593, 3844.921875], [8547.700278828617, 1200.83203125]], [[8537.677734375, 3816.2444731815604], [7965.89208984375, 3814.4087662685624]], [[7958.89990234375, 4244.850271421873], [6281.02880859375, 4238.168888806229]]])
    vertices = building_boundary.trace_boundary_face_points(boundary_points_new,
                                                            6,
                                                            max_error=1,
                                                            alpha=0.003,
                                                            k=math.ceil(180),
                                                            num_points=2,
                                                            angle_epsilon=0.2,
                                                            perp_dist_weight=4,
                                                            # #primary_orientations = [-1.5707963267948966, -3.141592653589793, -4.71238898038469, -6.283185307179586, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586],
                                                            merge_distance=4
                                                            )
    vertices = np.append(vertices, [vertices[0]], axis=0)
    print(f"Boundary vertices: {vertices}")
    for i in range(len(vertices)-1):
        start_point = (int(vertices[i][0]), int(vertices[i][1]))
        end_point = (int(vertices[i + 1][0]), int(vertices[i + 1][1]))
        cv2.line(blank_image, start_point, end_point, (0,0,255), 2)

    # Display the result
    plt.imshow(blank_image)
    plt.title("Processed Image")
    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == '__main__':
    img_path = r"src/img2.png"
    score_thr = 0.4
    dist_thr = 8.0
    process_for_LSD(img_path, score_thr, dist_thr)
