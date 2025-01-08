import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 读取 XML 文件
file_path = "output.xml"  # 替换为你的 XML 文件路径
tree = ET.parse(file_path)
root = tree.getroot()

# 获取 WallInfo 节点
wall_info = root.find("WallInfo")
if wall_info is None:
    raise ValueError("XML 文件中缺少 WallInfo 节点")

# 提取墙壁线段信息
lines = []
for line_data in wall_info.findall("LineData"):
    start_x = float(line_data.get("StartX"))
    start_y = float(line_data.get("StartY"))
    end_x = float(line_data.get("EndX"))
    end_y = float(line_data.get("EndY"))
    lines.append(((start_x, start_y), (end_x, end_y)))

# 绘制线段
plt.figure(figsize=(10, 10))
for line in lines:
    (start_x, start_y), (end_x, end_y) = line
    plt.plot([start_x, end_x], [start_y, end_y], marker="o")

# 配置图表
plt.title("Wall Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
