import matplotlib.pyplot as plt
import numpy as np


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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

def merge_lines_with_conditions(lines, distance_threshold=15, angle_threshold=10):
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


# 定义一个 Line 类来表示线段
class Line:
    def __init__(self, start, end, angle):
        self.start = np.array(start)
        self.end = np.array(end)
        self.angle = angle

    @staticmethod
    def projection_onto_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_point = line_start + proj_length * line_unitvec
        return proj_point

# 定义两条线段之间的距离
def line_to_line_dist(start1, end1, start2, end2):
    """计算两条线段之间的最短距离"""
    dist1 = point_to_line_dist(start1, start2, end2)
    dist2 = point_to_line_dist(end1, start2, end2)
    dist3 = point_to_line_dist(start2, start1, end1)
    dist4 = point_to_line_dist(end2, start1, end1)
    return min(dist1, dist2, dist3, dist4)

def merge_lines(lines):
    # 将所有线段的端点提取出来
    points = []
    num_lines = len(lines)
    angles = 0
    for line in lines:
        angles += line.angle
        points.extend([line.start, line.end])
    angles_ave = angles / num_lines

    # 寻找距离最远的两个点，作为新的合并线段
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
    # 返回新的合并线段
    return Line(new_start, new_end,0)

# 创建测试数据
lines = [
    Line([695, 869], [696, 1043], 89.67071753615333),
    Line([694, 1045], [1101, 1044], 179.8592244122409),
    Line([700, 767], [1100, 772], 0.7161599454704085),
    Line([1102, 769], [1105, 844], 87.70938995736148),
    Line([1103, 984], [1104, 1047], 89.09061955080087),
    Line([1092, 769], [1096, 1053], 89.19307054489762),
]

# 调用合并函数
merged_lines = merge_lines_with_conditions(lines)

# 可视化函数
def plot_lines(lines, color="blue", label="Original"):
    for line in lines:
        plt.plot(
            [line.start[0], line.end[0]],
            [line.start[1], line.end[1]],
            color=color,
            label=label if label else None,
        )
        label = None  # 防止重复标签

# 绘制原始线段
plt.figure(figsize=(10, 10))
plot_lines(lines, color="blue", label="Original Lines")

# 绘制合并后的线段
plot_lines(merged_lines, color="red", label="Merged Lines")

# 图形设置
plt.legend()
plt.gca().invert_yaxis()  # 翻转 Y 轴方向，使其符合图像坐标系
plt.title("Line Segments Merging Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

