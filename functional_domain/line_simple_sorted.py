import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines
from sklearn.cluster import DBSCAN
import functional_domain.line_cloud_match as line_cloud_match
import functional_domain.points_png as points_png
from functional_domain.line_operations import Line, merge_lines,line_to_line_dist,line_to_line
from functional_domain.subsidiary import distance,find_max_contour
import matplotlib.cm as cm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()

# 构造模型文件的路径
model_path = os.path.join(current_dir, 'tflite_models', 'M-LSD_512_large_fp32.tflite')


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

input_size = 512

# Load tflite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def merge_lines_with_conditions(lines, distance_threshold=10, angle_threshold=10):
    merged_lines = []
    groups = []  # 用于存储线段分组

    for i, line1 in enumerate(lines):
        # 创建新分组
        group = [line1]
        for j in range(i + 1, len(lines)):
            line2 = lines[j]
            dist = line_to_line_dist(line1.start, line1.end, line2.start, line2.end)
            angle_diff = abs(line1.angle - line2.angle)
            if dist < distance_threshold and (angle_diff < angle_threshold or angle_diff > 180 - angle_threshold):
                group.append(line2)
        # 将新分组添加到 groups
        groups.append(group)

    def has_common_line(group1, group2):
        # 判断两个组是否有公共线段
        return any(line in group2 for line in group1)

    merged_groups = []
    while groups:
        group = groups.pop(0)
        merged = False
        if len(group) == 1:
            merged_lines.append(group[0])
        elif len(group) > 1:
            for merged_group in merged_groups:
                if has_common_line(group, merged_group):
                    merged_group.extend(group)
                    # set(merged_group)
                    merged = True
                    break
            if not merged:
                merged_groups.append(group)
    for group in merged_groups:
        merged_lines.append(merge_lines(group))

    return merged_lines
# def merge_lines_with_conditions(lines, distance_threshold=10, angle_threshold=10):
#
#     merged_lines = []
#     used = [False] * len(lines)
#
#     for i in range(len(lines)):
#         multi_lines = []
#         if used[i]:
#             continue
#         line1 = lines[i]
#         angle1 = line1.angle
#         if 10 < angle1 < 80 or 100 < angle1 < 170:
#             used[i] = True
#             continue
#         merged = False
#         for j in range(i+1,len(lines)):
#             if used[j]:
#                 continue
#             line2 = lines[j]
#             angle2 = line2.angle
#             dist = line_to_line_dist(line1.start,line1.end,line2.start, line2.end)
#             angle_diff = abs(angle1 - angle2)
#             if dist < distance_threshold and angle_diff < angle_threshold:
#                 if not used[i]:
#                     multi_lines.append(line1)
#                 multi_lines.append(line2)
#                 used[i] = used[j] = True
#                 merged = True
#         if merged:
#             merged_line = merge_lines(multi_lines)
#             merged_lines.append(merged_line)
#         else:
#             merged_lines.append(line1)
#     return merged_lines

class LineSegment:
    def __init__(self, start_point, end_point, length, angle):
        self.start_point = start_point
        self.end_point = end_point
        self.length = length
        self.angle = angle

# 线段补全
# def line_to_line(lines,threshold):
#     insert_lines = []
#     # 逐线段计算
#     for i in range(len(lines) - 1):
#         line1 = lines[i]
#         line2 = lines[i + 1]
#         # 获取两条线段的起点和终点
#         start1, end1 = line1.start, line1.end
#         start2, end2 = line2.start, line2.end
#
#         # 计算 4 个点之间的距离，并记录对应的点对
#         distances = [
#             (distance(end1, start2), end1, start2),  # 第一条线段的末端与第二条线段的起点
#             (distance(end1, end2), end1, end2),  # 第一条线段的末端与第二条线段的末端
#             (distance(start1, start2), start1, start2),  # 第一条线段的起点与第二条线段的起点
#             (distance(start1, end2), start1, end2)  # 第一条线段的起点与第二条线段的末端
#         ]
#
#         # 找到最小的距离以及对应的点对
#         min_distance, point1, point2 = min(distances, key=lambda x: x[0])
#         insert_lines.append(line1)
#         if min_distance >= threshold and abs(line1.angle - line2.angle) < 1:
#             # pass
#             insert_lines.append(Line(point1,point2))
#         if i == len(lines) - 2:
#             insert_lines.append(line2)
#     return insert_lines

def match_line_to_boundary(boundary_points, matched_lines, threshold=50):
    # 匹配直线端点和凸包点
    # boundary_points = np.array([(p[1], p[0]) for p in boundary_points])
    # matched_lines = []
    # for line in detected_lines:
    #     start, end = line
    #     matched_lines.append(Line(start,end))
    for idx,point in enumerate(boundary_points):
        middle_dist = [distance(point,line.middle) for line in matched_lines]
        closest_dists = np.argsort(middle_dist)[:2]  # 获取最小的两个索引

        if idx == 0 and len(closest_dists) == 2:
            # 处理最小的两个索引点
            angles = []
            for closest_dist in closest_dists:
                angle = np.degrees(np.arctan2(matched_lines[closest_dist].middle[1] - point[1], matched_lines[closest_dist].middle[0] - point[0]))
                angles.append(angle)
            if angles[0] > angles[1]:
                if middle_dist[closest_dists[0]] < threshold:
                        matched_lines[closest_dists[0]].idx += -10000
                        matched_lines[closest_dists[0]].num += 1
                if middle_dist[closest_dists[1]] < threshold:
                        matched_lines[closest_dists[1]].idx += 999999
                        matched_lines[closest_dists[1]].num += 1
            else:
                if middle_dist[closest_dists[0]] < threshold:
                        matched_lines[closest_dists[0]].idx += 999999
                        matched_lines[closest_dists[0]].num += 1
                if middle_dist[closest_dists[1]] < threshold:
                        matched_lines[closest_dists[1]].idx += idx
                        matched_lines[closest_dists[1]].num += -10000
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

# Function to detect lines and trace boundaries
def process_for_LSD(img_input,img_input2,score_thr, dist_thr,line_threshold,rt_slice_points):
    '''
    主要功能函数：图像匹配、直线拟合、多边形提取
    '''

    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[input_size, input_size], score_thr=score_thr, dist_thr=dist_thr)
    print("Image shape:",img_input2.shape)

    # cv2.imshow("Image1", img_input)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow("Image2",img_input2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    max_contour = find_max_contour(img_input2)

    # image_show = np.full((img_input2.shape[0], img_input2.shape[1]), 0, dtype=np.uint8)
    # # 显示轮廓，调整颜色和线宽
    # cv2.drawContours(image_show, [max_contour], -1, (255, 255, 255), 2)  # BGR格式颜色
    #
    # # 设置窗口名称和大小
    # window_name = 'Adaptive Threshold Contours'
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 800, 600)
    #
    # # 显示图像
    # cv2.imshow(window_name, image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours_points = max_contour.reshape(-1,2)
    detected_lines = [[np.array([int(val) for val in line[:2]]), np.array([int(val) for val in line[2:]])] for line in
                      lines]
    multi_lines = []
    print("1: Remove abnormal line segments and merge lines")
    for line in detected_lines:
        start, end = line
        multi_lines.append(Line(start, end))
    # # 可视化1
    # colors = cm.get_cmap('tab20', len(multi_lines))
    # for idx, line in enumerate(multi_lines):
    #     start, end = line.start, line.end
    #     color = colors(idx / len(multi_lines))  # 选择颜色
    #     plt.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2,
    #              label=f'Line {idx + 1}' if idx == 0 else "")
    #
    #     # 在线段的中点处标注编号
    #     mid_x = (start[0] + end[0]) / 2
    #     mid_y = (start[1] + end[1]) / 2
    #     plt.text(mid_x, mid_y, str(idx + 1), fontsize=12, ha='center', va='center', color=color)
    # # 设置图形属性
    # plt.title('Matched Lines with Convex Hull Points and Line Numbers')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().invert_yaxis()
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    merge_lines = merge_lines_with_conditions(multi_lines)


    matched_lines = match_line_to_boundary(contours_points, merge_lines, threshold=100)

    print("2: Sort roughly, in the direction of the contour map")
    print("3: Line segment normalization and completion")
    rect_lines = []
    for line in matched_lines:
        updated_line = line.update_line_if_needed()
        if updated_line:
            rect_lines.append(updated_line)

    # # 可视化2
    # colors = cm.get_cmap('tab20', len(rect_lines))
    # for idx, line in enumerate(rect_lines):
    #     start, end = line.start, line.end
    #     color = colors(idx / len(rect_lines))  # 选择颜色
    #     plt.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2,
    #              label=f'Line {idx + 1}' if idx == 0 else "")
    #
    #     # 在线段的中点处标注编号
    #     mid_x = (start[0] + end[0]) / 2
    #     mid_y = (start[1] + end[1]) / 2
    #     plt.text(mid_x, mid_y, str(idx + 1), fontsize=12, ha='center', va='center', color=color)
    # # 设置图形属性
    # plt.title('Matched Lines with Convex Hull Points and Line Numbers')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().invert_yaxis()
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    Contour_points = line_to_line(rect_lines)

    # 可视化3
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray_input >= 220)
    points = [[x, y] for x, y in zip(xs, ys)]
    points_array = np.array(points)
    plt.scatter(points_array[:, 0], points_array[:, 1], color='gray', s=3,alpha=0.6, label='Points near line')
    for idx, line in enumerate(Contour_points):
        start, end = line.start, line.end
        plt.plot([start[0], end[0]], [start[1], end[1]], color="red", lw=2,
                 label=f'Line {idx + 1}' if idx == 0 else "")
    # 设置图形属性
    plt.title('Matched Lines with Convex Hull Points and Line Numbers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Number of detected lines: {len(lines)}")
    boundary_points_new = []
    for idx, line in enumerate(Contour_points):
        start, end = line.start, line.end
        boundary_points_new.append([start, end])
    walls, doors = line_cloud_match.match_and_visualize(boundary_points_new, img_input,rt_slice_points)
    # # boundary_points = np.array([[[6272.994315602144, 2769.517578125], [6283.751909304447, 995.2520141601562]], [[6286.72216796875, 962.3297378014122], [7690.58349609375, 966.1963684957117]], [[7869.31005859375, 966.1767322108095], [8438.46875, 967.6881996419082]], [[8541.34145180593, 3844.921875], [8547.700278828617, 1200.83203125]], [[8537.677734375, 3816.2444731815604], [7965.89208984375, 3814.4087662685624]], [[7958.89990234375, 4244.850271421873], [6281.02880859375, 4238.168888806229]]])

    return walls,doors

def calculate_angle_diff(angle1, angle2):
    """计算两条直线的角度差"""
    diff = abs(angle1 - angle2)
    return min(diff, 180 - diff)

def rotation_matrix(theta):
    """计算三维旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

def rotate_point_cloud(point_cloud, rot_matrix):
    """对三维点云数据进行旋转"""
    return np.dot(point_cloud, rot_matrix.T)

def become_full_member(img):
    lines = pred_lines(img, interpreter, input_details, output_details,
                       input_shape=[input_size, input_size], score_thr=0.2, dist_thr=10)

    line_segments = []
    for line in lines:
        x_start, y_start, x_end, y_end = [val for val in line]
        start_point = np.array([x_start, y_start])
        end_point = np.array([x_end, y_end])
        l_dis = distance(start_point, end_point)
        angle = np.degrees(np.arctan2(y_end - y_start,x_end - x_start))
        if angle < 0:
            angle += 180
        line_segment = LineSegment(start_point, end_point,l_dis,angle)
        line_segments.append(line_segment)
    angles = np.array([line.angle for line in line_segments]).reshape(-1,1)
    dbscan = DBSCAN(eps=5, min_samples=2)
    clusters = dbscan.fit_predict(angles)
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    if len(unique_clusters) < 2:
        raise ValueError("DBSCAN 没有检测到足够的聚类")

    top_two_clusters = unique_clusters[np.argsort(counts)[-2:]]
    # 从每一类中各选取一条直线
    lines_in_cluster1 = [line for line, cluster in zip(line_segments, clusters) if cluster == top_two_clusters[0]]
    lines_in_cluster2 = [line for line, cluster in zip(line_segments, clusters) if cluster == top_two_clusters[1]]

    # 计算每个聚类的中值
    angles_cluster1 = np.array([line.angle for line in lines_in_cluster1])
    angles_cluster2 = np.array([line.angle for line in lines_in_cluster2])

    median_angle1 = np.median(angles_cluster1)
    median_angle2 = np.median(angles_cluster2)

    # 计算这两条直线的角度差
    angle_diff = calculate_angle_diff(median_angle1, median_angle2)

    # 如果角度差接近 90 度，选取其中一条直线
    if abs(angle_diff - 90) < 10:  # 接近 90 度的阈值
        selected_angle = median_angle1 if median_angle1 < median_angle2 else median_angle2
    else:
        raise ValueError("直线之间的角度差不接近 90 度")

    # 判断选中的直线是接近 X 轴还是 Y 轴
    if abs(selected_angle) < 45:  # 接近 X 轴
        theta = np.radians(-selected_angle)  # 计算旋转角度
        rot_matrix = rotation_matrix(theta)
    elif abs(selected_angle - 90) < 45:  # 接近 Y 轴
        theta = np.radians(90 - selected_angle)  # 计算旋转角度
        rot_matrix = rotation_matrix(theta)
    else:
        raise ValueError("选中的直线不接近 X 轴或 Y 轴")
    return rot_matrix


def line_process(Outer_contour,Inner_contour,rt_slice_points):
    """
    Extraction and processing of internal and external contours.
    """
    score_thr = 0.3
    dist_thr = 5.0
    line_threshold = 50
    walls, doors = process_for_LSD(Outer_contour, Inner_contour, score_thr, dist_thr, line_threshold,rt_slice_points)
    return walls, doors

def point_process(points):
    '''
    Point cloud turns positive.
    '''
    slice_points,multi_sliced_points,img = points_png.point_cloud_to_png(points)
    rt_matrix = become_full_member(img)
    rotated_point = rotate_point_cloud(multi_sliced_points, rt_matrix)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(rotated_point_cloud)
    # o3d.visualization.draw_geometries([point_cloud])
    return rt_matrix,rotated_point