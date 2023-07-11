from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines, pred_squares
import gradio as gr
import argparse
import math
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

# 线段顺时针排序
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

# 点组线段
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

# 可视化排序点云
def plot_lines_with_numbers(LineParams):
    # Sort LineParams
    sorted_lines = sort_neighborhood(LineParams)

    # Plot each line segment and annotate with its index
    for idx, line in enumerate(sorted_lines):
        start_point = line[0][:2]
        end_point = line[1][:2]
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], label=f"Line {idx+1}")
        plt.text((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2, str(idx+1),
                 horizontalalignment='center', verticalalignment='center')

    # Set plot title and legend
    plt.title('Lines with numbers')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    plt.grid(True)
    plt.show()

# Function to handle line detection
def gradio_wrapper_for_LSD(img_input, score_thr, dist_thr):
    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[args.input_size, args.input_size], score_thr=score_thr, dist_thr=dist_thr)
    img_output = img_input.copy()

    # # 角点检测
    # corners = cv2.goodFeaturesToTrack(img_output, maxCorners=200, qualityLevel=0.01, minDistance=20)
    # corners = np.intp(corners)
    # points = corners.reshape((-1, 2))

    height, width = img_output.shape[:2]
    blank_image = np.full((height, width, 3), 255, dtype=np.uint8)
    # 显示拟合出的直线数
    print(len(lines))

    # 线段分割及排序
    points = []
    LineParams = []
    # Draw lines on the image
    for line in lines:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        points.append([x_start, y_start])
        points.append([x_end, y_end])
        LineParams.append([[x_start,y_start],[x_end,y_end]])
        cv2.line(blank_image, (x_start, y_start), (x_end, y_end), [0, 255, 255], 1)

    # # 排序
    # center = np.mean(np.array(points), axis=0)
    # for idx,lineParam in enumerate(LineParams) :
    #     LineParams[idx]=sort_points_clockwise(lineParam, center)
    #
    # plot_lines_with_numbers(LineParams)
    # LineParams = sort_neighborhood(LineParams)
    #
    # # 数据格式转换
    # boundary_points_new = []
    # for idx, pts in enumerate(np.array(LineParams)):
    #     a = pts.tolist()[0][:2]
    #     b = pts.tolist()[1][:2]
    #     boundary_points_new.append([a, b])
    # # boundary_points = np.array([[[6272.994315602144, 2769.517578125], [6283.751909304447, 995.2520141601562]], [[6286.72216796875, 962.3297378014122], [7690.58349609375, 966.1963684957117]], [[7869.31005859375, 966.1767322108095], [8438.46875, 967.6881996419082]], [[8541.34145180593, 3844.921875], [8547.700278828617, 1200.83203125]], [[8537.677734375, 3816.2444731815604], [7965.89208984375, 3814.4087662685624]], [[7958.89990234375, 4244.850271421873], [6281.02880859375, 4238.168888806229]]])
    # vertices = building_boundary.trace_boundary_face_points(boundary_points_new,
    #                                                         6,
    #                                                         max_error=1,
    #                                                         alpha=0.003,
    #                                                         k=math.ceil(180),
    #                                                         num_points=2,
    #                                                         angle_epsilon=0.2,
    #                                                         perp_dist_weight=4,
    #                                                         # #primary_orientations = [-1.5707963267948966, -3.141592653589793, -4.71238898038469, -6.283185307179586, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586],
    #                                                         merge_distance=4
    #                                                         )
    # vertices = np.append(vertices, [vertices[0]], axis=0)
    # print(vertices)
    # for i in range(len(vertices)-1):
    #     start_point = (int(vertices[i][0]), int(vertices[i][1]))
    #     end_point = (int(vertices[i + 1][0]), int(vertices[i + 1][1]))
    #     # cv2.line(blank_image, start_point, end_point, (0,0,255), 2)
    return blank_image

# Use locally saved images1
sample_images = [["example1.jpg", 0.2, 10.0],
                 ["example2.jpg", 0.2, 10.0],
                 ["example3.jpg", 0.2, 10.0]]

iface = gr.Interface(gradio_wrapper_for_LSD,
                     ["image",
                      gr.Number(value=0.2, label='score_thr (0.0 ~ 1.0)'),
                      gr.Number(value=10.0, label='dist_thr (0.0 ~ 20.0)')
                     ],
                     "image",
                     title="Line segment detection with Mobile LSD (M-LSD)",
                     description="M-LSD is a light-weight and real-time deep line segment detector...",
                     examples=sample_images)

# Launch Gradio interface
iface.launch(share=True)
# server_name="0.0.0.0",



