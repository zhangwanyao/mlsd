import os
import ezdxf
import math
from ezdxf.math import Vec2
import numpy as np
import json
import matplotlib.pyplot as plt
from functional_domain.line_operations import classification_doors_windows,line_to_line_dist,Line
from config import Config


def add_door(msp, start, end):
    center = Vec2(start)
    radius = center.distance(Vec2(end))

    # 计算从 start 到 end 的方向角度 (以度为单位)
    direction_angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))

    # 调整方向角度以确保正确的起始位置
    start_angle = direction_angle
    end_angle = start_angle + 90  # 1/4 圆弧

    # 添加 1/4 圆弧
    msp.add_arc(
        center=center,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        dxfattribs={"layer": "Doors", "color": 3}  # 绿色
    )

    # 计算连接线的终点
    line_end = center + Vec2.from_angle(math.radians(start_angle + 90)) * radius

    # 添加连接线
    msp.add_line(start=start, end=line_end, dxfattribs={"layer": "Walls", "color": 3})  # 绿色

def save_to_dxf(input_json, output_dxf,rt_slice_points):
    """
    从 JSON 文件中读取墙、门、窗和多边形数据，并将其保存为 DXF 文件。
    """
    # 加载 JSON 数据
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建 DXF 文档
    doc = ezdxf.new()
    msp = doc.modelspace()

    # 绘制墙
    for wall in data["walls"]:
        start, end = wall
        msp.add_line(start=start, end=end, dxfattribs={"layer": "Walls", "color": 1})  # 红色

    # 绘制门
    for door in data["doors"]:
        start, end = door
        add_door(msp, start, end)

        # 绘制窗
    for window in data["windows"]:
        start, end = window
        msp.add_line(start=start, end=end, dxfattribs={"layer": "Windows", "color": 5})  # 蓝色

    # 绘制多边形
    for polygon in data["polygons"]:
        points = polygon + [polygon[0]]  # 闭合多边形
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Polygons", "color": 8})  # 灰色填充

    # 添加点
    for point in rt_slice_points:
        x, y = point[:2]*1000
        msp.add_point((x, y), dxfattribs={"layer": "PointCloud", "color": 7})  # 白色点
    # 保存 DXF 文件
    doc.saveas(output_dxf)
    print(f"DXF 文件已保存到 {output_dxf}")

class Wall:
    def __init__(self,wall):
        self.start = wall[0]
        self.end = wall[1]
        self.angle = self.line_angle()
    def line_angle(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.angle = math.degrees(math.atan2(dy, dx))
        if self.angle < 0:
            self.angle += 180
        return self.angle

def points_to_line(points, x1, y1, x2, y2):
    """
    计算点数组points (px, py) 投影到线段 (x1, y1), (x2, y2) 的点
    """
    ptolines = []
    ptolines.append(np.array([x1,y1]))
    ptolines.append(np.array([x2, y2]))
    for point in points:
        p = np.array(point)
        a = np.array([x1,y1])
        b = np.array([x2,y2])
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        new_point = a + t * ab
        ptolines.append(new_point)
    return ptolines

def sort_stend(points, x1, y1):
    '''
    将点数组points从(x1,y1)的顺序进行排序
    '''
    def distance_from_start(point):
        return math.sqrt((point[0] - x1) ** 2 + (point[1] - y1) ** 2)
    sorted_points = sorted(points, key=distance_from_start)
    return sorted_points


def distance_points(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def cluster_walls_and_doors(points, distance_threshold):
    walls = []
    doors = []
    current_wall = [points[0]]
    for i in range(1, len(points)):
        distance = distance_points(points[i - 1], points[i])
        if distance <= distance_threshold:
            current_wall.append(points[i])
        else:
            if len(current_wall) >= 2 and np.linalg.norm(current_wall[0] - current_wall[-1]) > 20:
                wall = (current_wall[0], current_wall[-1])
                if not any(np.allclose(wall, w) for w in walls):
                    walls.append(wall)
            door = (current_wall[-1], points[i])
            doors.append(door)
            current_wall = [points[i]]
    if len(current_wall) >= 5 and np.linalg.norm(current_wall[0] - current_wall[-1]) > 20:
        # Store the last wall
        wall = (current_wall[0], current_wall[-1])
        if not any(np.allclose(wall, w) for w in walls):
            walls.append(wall)
    return walls, doors


def is_parallel_and_within_distance(wall1, wall2, distance_threshold):
    if abs(wall1.angle - 90) < 1:
        angle_diff = abs(wall1.angle - wall2.angle)
        if angle_diff > 170 or angle_diff < 10:
            if 5 < abs(wall1.start[0] - wall2.start[0]) < distance_threshold:
                dist = line_to_line_dist(wall1.start, wall1.end, wall2.start, wall2.end)
                if dist <= distance_threshold:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        angle_diff = abs(wall1.angle - wall2.angle)
        if angle_diff > 170 or angle_diff < 10:
            if 5 < abs(wall1.start[1] - wall2.start[1]) < distance_threshold:
                dist = line_to_line_dist(wall1.start, wall1.end, wall2.start, wall2.end)
                if dist <= distance_threshold:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    # dist = line_to_line_dist(wall1.start,wall1.end, wall2.start,wall2.end)
    # if dist <= distance_threshold:
    #     if angle_diff > 170 or angle_diff < 10:
    #         return True
    #     else:
    #         return False
    # else:
    #     return False

def sort_vertices_clockwise(vertices):
    """
    将角点按顺时针顺序排序
    :param vertices: 角点的数组，形状为 (n, 2)
    :return: 按顺时针顺序排序后的角点数组
    """
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    return sorted_vertices


def convert_to_point_cloud(x, y, off_size, Image_rows,minx,miny):
    """
    根据公式将像素坐标 (x, y) 转换为点云坐标。
    """
    point_cloud_x = x * 5 + minx
    point_cloud_y = (Image_rows - y) * 5 + miny
    return [point_cloud_x, point_cloud_y]

def save_to_json(all_walls, all_doors, all_windows,polygons, output_json):
    """
    将墙、门、窗的数据转换为点云坐标并保存为 JSON 文件。
    """
    off_size = Config.off_size
    Image_rows = Config.Image_rows
    minx = Config.minx
    miny = Config.miny

    def convert_items(items):
        """
        转换墙体或门窗数据为点云坐标。
        """
        return [
            [convert_to_point_cloud(start[0], start[1], off_size, Image_rows,minx,miny),
             convert_to_point_cloud(end[0], end[1], off_size, Image_rows,minx,miny)]
            for start, end in items
        ]

    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # 转换数据
    walls_data = convert_items(all_walls)
    doors_data = convert_items(all_doors)
    windows_data = convert_items(all_windows)
    polygons_data = [
            [convert_to_point_cloud(polygon[0][0], polygon[0][1], off_size, Image_rows,minx,miny),
             convert_to_point_cloud(polygon[1][0], polygon[1][1], off_size, Image_rows,minx,miny),
             convert_to_point_cloud(polygon[2][0], polygon[2][1], off_size, Image_rows,minx,miny),
             convert_to_point_cloud(polygon[3][0], polygon[3][1], off_size, Image_rows,minx,miny)]
            for polygon in polygons
        ]

    # 组装为 JSON 结构
    data = {
        "walls": walls_data,
        "doors": doors_data,
        "windows": windows_data,
        "polygons":polygons_data
    }

    serializable_data = json.loads(json.dumps(data, default=convert_to_serializable))

    # 保存为 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)

def draw_room(all_walls,all_doors,output_filename,output_json,rt_slice_points):
    '''
    param all_walls:array 2*2
    param all_doors:array 2*2
    param output_filename:string.
    '''
    walls = [] # 需要填充的墙体
    fig, ax = plt.subplots(1,1)
    # 绘制墙体（去掉门窗部分）
    for wall in all_walls:
        walls.append(Wall(wall))
        # ax.plot([wall[0][0][0], wall[0][1][0]], [wall[0][0][1], wall[0][1][1]], color='black', linewidth=4)
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=1)
        # 绘制门洞）
    # for door in all_doors:
    #     ax.plot([door[0][0], door[1][0]], [door[0][1], door[1][1]], color='cyan', linewidth=1)
    polygons = []
    for i, wall1 in enumerate(walls):
        for j in range(i + 1, len(walls)):
            wall2 = walls[j]
            if i != j and is_parallel_and_within_distance(wall1, wall2,40):
                # 多边形绘制
                # vertices = np.array([wall1.start, wall1.end,wall2.start,wall2.end])
                # new_vertices = sort_vertices_clockwise(vertices)
                # polygon = Polygon(new_vertices,closed=True,fill=True,edgecolor='black',facecolor='black',alpha=0.5)
                # ax.add_patch(polygon)

                # 填充的图形形状为矩形
                if abs(wall1.angle - 90) < 10:
                    x = np.array([wall1.start[0], wall2.start[0]])
                    arr = np.array([wall1.start[1],wall1.end[1],wall2.start[1],wall2.end[1]])
                    y = np.argsort(arr)
                    y1 = np.array([arr[y[1]],arr[y[1]]])
                    y2 = np.array([arr[y[2]], arr[y[2]]])
                else:
                    arr = np.array([wall1.start[0],wall1.end[0],wall2.start[0],wall2.end[0]])
                    sort_x = np.argsort(arr)
                    x = np.array([arr[sort_x[1]],arr[sort_x[2]]])
                    y1 = np.array([wall1.start[1],wall1.end[1]])
                    y2 = np.array([wall2.start[1],wall2.end[1]])
                polygons.append([[x[0], y1[0]], [x[0], y2[0]], [x[1], y1[1]], [x[1], y2[1]]])
                ax.fill_between(x,y1,y2,color='k', alpha=0.3)
    doors,windows = classification_doors_windows(all_doors,walls)

    # merge two doors
    to_remove = [False] * len(doors)
    for i, door1 in enumerate(doors):
        line1 = Line(door1[0], door1[1])
        for j in range(i + 1, len(doors)):
            door2 = doors[j]
            line2 = Line(door2[0], door2[1])
            if i != j and is_parallel_and_within_distance(line1, line2,40):
                to_remove[j] = True
    doors = [door for i, door in enumerate(doors) if not to_remove[i]]

    for door in doors:
        ax.plot([door[0][0], door[1][0]], [door[0][1], door[1][1]], color='blue', linewidth=1)
    for window in windows:
        ax.plot([window[0][0], window[1][0]], [window[0][1], window[1][1]], color='cyan', linewidth=2)
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()
    ax.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    # plt.show()

    # 图片中(x,y)换算点云坐标公式为
    # 像素x对应点云的 （x-off_size）*10
    # 像素y对应点云的 （Image_rows -y-off_size）*10
    polygons = [sort_vertices_clockwise(np.array(ploygon)) for ploygon in polygons]

    save_to_json(all_walls, doors, windows,polygons,output_json)
    output_dxf = os.path.join(os.path.dirname(output_filename),"output.dxf")
    save_to_dxf(output_json, output_dxf,rt_slice_points)