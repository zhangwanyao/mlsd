import ctypes
import numpy as np
import laspy
import sys
import os
import open3d as o3d
from config import Config
from functional_domain import line_door_walls, line_simple_sorted as process

def save_matrix_to_file_as_txt(matrix, las_file_path):
    save_dir = os.path.join(os.path.dirname(las_file_path), "result")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rotation_matrix.txt')
    np.savetxt(save_path, matrix, fmt="%.6f", delimiter="\t")
    print(f"旋转矩阵已保存到: {save_path}")

def load_matrix_from_file(txt_file_path):
    try:
        matrix = np.loadtxt(txt_file_path, delimiter=" ")
        print(f"旋转矩阵已从文件加载: {txt_file_path}")
        return matrix
    except Exception as e:
        print(f"读取旋转矩阵失败: {e}")
        return None


class PointCloudCache:
    def __init__(self):
        self.point_cloud_data = None

    def load_point_cloud(self, path):
        if self.point_cloud_data is None:
            with laspy.open(path) as las_file:
                las_data = las_file.read()
                points = las_data.xyz
            self.point_cloud_data = np.array(points,np.float32)
        return self.point_cloud_data

    def apply_rotation(self, rotation_matrix):
        if self.point_cloud_data is None:
            raise ValueError("Point cloud data not loaded.")
        rotated_points = np.dot(self.point_cloud_data, rotation_matrix.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotated_points)
        # o3d.io.write_point_cloud("rt.pcd", pcd)
        # o3d.visualization.draw_geometries([pcd])
        return np.ascontiguousarray(rotated_points.flatten(), dtype=np.float32), rotated_points.shape[0]

class MyData(ctypes.Structure):
    _fields_ = [
        ('contours', ctypes.POINTER(ctypes.c_ubyte)),  # Mat 指针
        ('edges', ctypes.POINTER(ctypes.c_ubyte)),      # Mat 指针
        ("contoursRow", ctypes.c_int),
        ("contoursCol", ctypes.c_int),
        ("edgesRow",  ctypes.c_int),
        ("edgesCol",  ctypes.c_int),
        ("minx", ctypes.c_double),
        ("miny", ctypes.c_double),
    ]

if hasattr(sys, '_MEIPASS'):  # PyInstaller 的临时目录
    current_dir = sys._MEIPASS
else:
    current_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
dll_path = os.path.join(current_dir, 'RoomSegment', 'RoomSegment.dll')
print("dll_path", dll_path)
try:
    my_dll = ctypes.CDLL(dll_path)
except Exception as e:
    print(f"filed to load share Resegment_dll{e}.")

my_dll.RoomSegment.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int)
my_dll.RoomSegment.restype = None

my_dll.get_data.argtypes = (ctypes.POINTER(ctypes.c_int),)
my_dll.get_data.restype = ctypes.POINTER(MyData)

my_dll.clear_data.argtypes = []
my_dll.clear_data.restype = None


if __name__ == '__main__':
    # las_file_path = sys.argv[1]
    # rt_file_path = sys.argv[2]
    # rt_matrix = load_matrix_from_file(rt_file_path)
    las_file_path = r"C:\Users\wanyao.zhang\Downloads\户型二精细1.las"
    pcd_cache = PointCloudCache()
    points = pcd_cache.load_point_cloud(las_file_path)
    print("turn_to_postive")

    rt_matrix,rt_slice_points = process.point_process(points)
    save_matrix_to_file_as_txt(rt_matrix, las_file_path)

    points_flat, num_points = pcd_cache.apply_rotation(rt_matrix)
    # points_flat = points.flatten()
    # num_points = points.shape[0]
    float_p = ctypes.POINTER(ctypes.c_float)
    points_ctypes = points_flat.ctypes.data_as(float_p)
    door_size = 90
    Config.off_size = door_size

    my_dll.RoomSegment(points_ctypes, num_points,0,door_size)
    count = ctypes.c_int()
    data_array = my_dll.get_data(ctypes.byref(count))

    walls = []
    doors = []

    for i in range(count.value):
        data = data_array[i]
        contoursRow = data.contoursRow
        contoursCol = data.contoursCol
        Config.minx = data.minx
        Config.miny = data.miny

        contours_size = (data.contoursRow, data.contoursCol,3)
        edges_size = (data.edgesRow, data.edgesCol,3)
        print(f"contoursRow: {contoursRow}, contoursCol: {contoursCol}")

        Config.Image_rows = data.edgesRow
        contours_matrix = np.ctypeslib.as_array(data.contours, shape=contours_size) #Inner contour
        edges_matrix = np.ctypeslib.as_array(data.edges, shape=edges_size) # Outer contour
        try:
            wall,door = process.line_process(edges_matrix,contours_matrix,rt_slice_points)
            if wall is not None and door is not None:
                walls.extend(wall)
                doors.extend(door)
            else:
                print("Warning: process.line_process returned None for either wall or door.")
        except Exception as e:
            print(f"Error processing lines: {e}")
    save_dir = os.path.join(os.path.dirname(las_file_path), "result")
    os.makedirs(save_dir, exist_ok=True)
    png_save_path = os.path.join(save_dir, 'output.png')
    json_save_path = os.path.join(save_dir, 'floor_plan.json')
    line_door_walls.draw_room(walls, doors,png_save_path, json_save_path,rt_slice_points)
    my_dll.clear_data()