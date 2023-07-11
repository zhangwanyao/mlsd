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

# 线段补全
def line_to_line(lines,threshold):
    insert_lines = []
    # 逐线段计算
    for i in range(len(lines) - 1):
        line1 = lines[i]
        line2 = lines[i + 1]
        # 获取两条线段的起点和终点
        start1, end1 = line1.start, line1.end
        start2, end2 = line2.start, line2.end

        # 计算 4 个点之间的距离，并记录对应的点对
        distances = [
            (distance(end1, start2), end1, start2),  # 第一条线段的末端与第二条线段的起点
            (distance(end1, end2), end1, end2),  # 第一条线段的末端与第二条线段的末端
            (distance(start1, start2), start1, start2),  # 第一条线段的起点与第二条线段的起点
            (distance(start1, end2), start1, end2)  # 第一条线段的起点与第二条线段的末端
        ]

        # 找到最小的距离以及对应的点对
        min_distance, point1, point2 = min(distances, key=lambda x: x[0])
        insert_lines.append(line1)
        if min_distance >= threshold:
            pass
            # insert_lines.append(creat_line(point1,point2,0))
        if i == len(lines) - 2:
            insert_lines.append(line2)
    return insert_lines

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

        if idx == 0 and len(closest_dists) == 2:
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

# Function to detect lines and trace boundaries
def process_for_LSD(img_path1,img_path2,score_thr, dist_thr,line_threshold):
    img_input = cv2.imread(img_path1)
    if img_input is None:
        raise FileNotFoundError(f"Unable to load image at path: {img_path1}")
    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[args.input_size, args.input_size], score_thr=score_thr, dist_thr=dist_thr)

    img_input2 = cv2.imread(img_path2)
    if img_input2 is None:
        raise FileNotFoundError(f"Unable to load image at path: {img_path2}")
    gray = cv2.cvtColor(img_input2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=200)
    # 使用 cv2.findContours 找到顺序连接的边缘点
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=len)
        if len(max_contour) <= 30:
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=len)
        else:
            max_contour = contours[0]
    contours_points = max_contour.reshape(-1,2)

    detected_lines = []
    for idx, line in enumerate(lines):
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        start_point = np.array([x_start, y_start])
        end_point = np.array([x_end, y_end])
        detected_lines.append([start_point, end_point])

    # 匹配直线与凸包点
    matched_lines = match_line_to_boundary(contours_points, detected_lines, threshold=60)
    Contour_points = line_to_line(matched_lines,line_threshold)
    # 定义颜色映射
    colors = cm.get_cmap('tab20', len(Contour_points))  # 使用 'tab20' colormap 生成 20 种颜色

    # 绘制凸包边界
    plt.figure(figsize=(10, 10))
    plt.plot(contours_points[:, 0], contours_points[:, 1], 'o', label='Random Points')  # 绘制所有点

    boundary_points_new = []
    # 绘制匹配后的直线段，并用不同颜色区分，显示编号
    for idx, line in enumerate(Contour_points):
        start, end = line.start, line.end
        boundary_points_new.append([start, end])
        color = colors(idx / len(Contour_points))  # 选择颜色
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

    # # boundary_points = np.array([[[6272.994315602144, 2769.517578125], [6283.751909304447, 995.2520141601562]], [[6286.72216796875, 962.3297378014122], [7690.58349609375, 966.1963684957117]], [[7869.31005859375, 966.1767322108095], [8438.46875, 967.6881996419082]], [[8541.34145180593, 3844.921875], [8547.700278828617, 1200.83203125]], [[8537.677734375, 3816.2444731815604], [7965.89208984375, 3814.4087662685624]], [[7958.89990234375, 4244.850271421873], [6281.02880859375, 4238.168888806229]]])
    vertices = building_boundary.trace_boundary_face_points(boundary_points_new,
                                                            max_error=1,
                                                            num_points=2,
                                                            angle_epsilon=0.2,
                                                            perp_dist_weight=4,
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
    img_path1 = r"img6.png"
    img_path2 = r"img7.png"
    score_thr = 0.4
    dist_thr = 10.0
    line_threshold = 200
    process_for_LSD(img_path1, img_path2, score_thr, dist_thr,line_threshold)
