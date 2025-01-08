import os
import cv2
import numpy as np
import open3d as o3d


def export_scene(outpath, density_map, normals_map):
    def export_density():
        density_path = os.path.join(outpath, "density.png")
        density_uint8 = (density_map * 255).astype(np.uint8)
        cv2.imwrite(density_path, density_uint8)

    def export_normals():
        normals_path = os.path.join(outpath, "normals.png")
        normals_uint8 = (np.clip(normals_map, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(normals_path, normals_uint8)

    export_density()
    export_normals()

def generate_density(pcd,width=256, height=256):

    ps = np.array(pcd.points)
    pcd.estimate_normals()
    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res

    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = \
        np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None, :2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                             image_res - 1)
    
    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # print(np.unique(counts))
    # counts = np.minimum(counts, 1e2)

    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)
    # print(np.unique(density))

    normals = np.array(pcd.normals)
    normals_map = np.zeros((density.shape[0], density.shape[1], 3))

    import time
    start_time = time.time()
    for i, unique_coord in enumerate(unique_coordinates):
        # print(normals[unique_ind])
        normals_indcs = np.argwhere(np.all(coordinates[::10] == unique_coord, axis=1))[:, 0]
        normals_map[unique_coordinates[i, 1], unique_coordinates[i, 0], :] = np.mean(normals[::10][normals_indcs, :],
                                                                                     axis=0)

    print("Time for normals: ", time.time() - start_time)

    normals_map = (np.clip(normals_map, 0, 1) * 255).astype(np.uint8)

    # plt.figure()
    # plt.imshow(normals_map)
    # plt.show()

    return density, normals_map, normalization_dict

if __name__ == '__main__':
    path = r"C:\Users\wanyao.zhang\Desktop\降采样\rt_points.pcd"
    out_path = "C:\\Users\\wanyao.zhang\\Desktop\\"
    pcd = o3d.io.read_point_cloud(path)
    density, normals_map, normalization_dict = generate_density(pcd)
    export_scene(out_path, density, normals_map)