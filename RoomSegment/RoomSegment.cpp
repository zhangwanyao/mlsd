#define POINTCLOUDPROCESSING_EXPORTS
#include "RoomSegment.h"

//pcl::PointCloud<pcl::PointXYZ>::Ptr Load3DPtCloudData(const std::string& filename, std::vector<cv::Point3f>& scene_xyz_out) {
//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//
//	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
//		PCL_ERROR("Failed to read file %s\n", filename.c_str());
//		exit(-1);
//	}
//
//	for (const auto& point : *cloud) {
//		cv::Point3f p(point.x, point.y, point.z);
//		scene_xyz_out.push_back(p);
//	}
//
//	return cloud;
//}



std::vector<std::vector<cv::Point3f>> VoxelizePlanePoints(const std::vector<cv::Point3f>& plane_points, float voxelWidthMm, int theshold, cv::Mat& outputImage)
{
	std::vector<std::vector<cv::Point3f>> voxels;
	float minX = plane_points[0].x, maxX = plane_points[0].x;
	float minY = plane_points[0].y, maxY = plane_points[0].y;

	for (const auto& point : plane_points) {
		minX = min(minX, point.x);
		maxX = max(maxX, point.x);
		minY = min(minY, point.y);
		maxY = max(maxY, point.y);
	}

	int voxel_num_in_x = static_cast<int>((maxX - minX) / voxelWidthMm) + 1;
	int voxel_num_in_y = static_cast<int>((maxY - minY) / voxelWidthMm) + 1;

	voxels.resize(voxel_num_in_x * voxel_num_in_y);

	outputImage = cv::Mat(voxel_num_in_y * voxelWidthMm, voxel_num_in_x * voxelWidthMm, CV_8UC3, cv::Scalar(0, 0, 0));

	for (const auto& point : plane_points) {
		int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
		int y_idx = static_cast<int>((point.y - minY) / voxelWidthMm);
		int voxel_idx = y_idx * voxel_num_in_x + x_idx;

		voxels[voxel_idx].push_back(point);

		int x_pixel = x_idx * voxelWidthMm;
		int y_pixel = y_idx * voxelWidthMm;

		if (voxels[voxel_idx].size() >= theshold) {
			cv::Rect roi(x_pixel, y_pixel, voxelWidthMm, voxelWidthMm);
			outputImage(roi).setTo(cv::Scalar(255, 255, 255));
		}
	}

	return voxels;
}

void FindMinMaxCoordinates(const std::vector<cv::Point3f>& plane_points, float& minX, float& maxX, float& minY, float& maxY, float& minZ, float& maxZ) {
	minX = maxX = plane_points[0].x;
	minY = maxY = plane_points[0].y;
	minZ = maxZ = plane_points[0].z;

	for (const auto& point : plane_points) {
		minX = std::min(minX, point.x);
		maxX = std::max(maxX, point.x);
		minY = std::min(minY, point.y);
		maxY = std::max(maxY, point.y);
		minZ = std::min(minZ, point.z);
		maxZ = std::max(maxZ, point.z);
	}
	std::cout << "minX: " << minX << ", maxX: " << maxX << std::endl;
	std::cout << "minY: " << minY << ", maxY: " << maxY << std::endl;
	std::cout << "minZ: " << minZ << ", maxZ: " << maxZ << std::endl;
}

void VoxelizePoints(const std::vector<cv::Point3f>& plane_points, float minX, float minY, float voxelWidthMm,
	std::vector<std::vector<cv::Point3f>>& voxels, int voxel_num_in_x) {
	for (const auto& point : plane_points) {
		int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
		int y_idx = static_cast<int>((point.y - minY) / voxelWidthMm);
		int voxel_idx = y_idx * voxel_num_in_x + x_idx; // Use the correct X dimension size

		// Ensure the index is valid before accessing the vector
		if (voxel_idx >= 0 && voxel_idx < voxels.size()) {
			voxels[voxel_idx].push_back(point);
		}
		else {
			std::cerr << "Error: voxel_idx out of bounds: " << voxel_idx << std::endl;
		}
	}
}

void FilterVoxelsByDensity(const std::vector<std::vector<cv::Point3f>>& voxels, float radio, float voxelWidthMm, std::vector<std::vector<cv::Point3f>>& top_voxels) {
	std::vector<int> voxel_point_counts(voxels.size());
	for (size_t i = 0; i < voxels.size(); ++i) {
		voxel_point_counts[i] = static_cast<int>(voxels[i].size() / voxelWidthMm);
	}

	std::vector<int> non_empty_voxel_counts;
	for (int count : voxel_point_counts) {
		if (count > 0) {
			non_empty_voxel_counts.push_back(count);
		}
	}

	std::sort(non_empty_voxel_counts.begin(), non_empty_voxel_counts.end(), std::greater<int>());
	int num_voxels = non_empty_voxel_counts.size();
	int threshold_index = std::max(static_cast<int>(num_voxels * radio), 1);
	int threshold_point_count = non_empty_voxel_counts[threshold_index] * voxelWidthMm;

	for (size_t i = 0; i < voxels.size(); ++i) {
		if (voxels[i].size() >= threshold_point_count) {
			top_voxels.push_back(voxels[i]);
		}
	}
}

void UpdateBoundsForFilteredVoxels(const std::vector<std::vector<cv::Point3f>>& top_voxels, int door_size, float voxelWidth_filter, float& minX, float& maxX, float& minY, float& maxY, float& minZ, float& maxZ, double& minx, double& miny) {
	minX = maxX = top_voxels[0][0].x;
	minY = maxY = top_voxels[0][0].y;
	minZ = maxZ = top_voxels[0][0].z;

	for (const auto& voxel : top_voxels) {
		for (const auto& point : voxel) {
			minX = std::min(minX, point.x);
			maxX = std::max(maxX, point.x);
			minY = std::min(minY, point.y);
			maxY = std::max(maxY, point.y);
			minZ = std::min(minZ, point.z);
			maxZ = std::max(maxZ, point.z);
		}
	}

	minx = minX;
	miny = minY;
	maxX += door_size * voxelWidth_filter * 2;
	maxY += door_size * voxelWidth_filter * 2;
	minY -= door_size * voxelWidth_filter * 2;
	minX -= door_size * voxelWidth_filter * 2;
}


//top_voxels可以是不同体素建立的，返回的是预设的minX下，voxelWidth_filter建立的体素
void CreateFilteredVoxelGrid(const std::vector<std::vector<cv::Point3f>>& top_voxels, float maxX, float maxY, float minX, float minY, float voxelWidth_filter, std::vector<std::vector<cv::Point3f>>& voxels_filter) {
	int voxel_num_in_x_filter = static_cast<int>((maxX - minX) / voxelWidth_filter);
	int voxel_num_in_y_filter = static_cast<int>((maxY - minY) / voxelWidth_filter);
	voxels_filter.resize(voxel_num_in_y_filter * voxel_num_in_x_filter);


	for (const auto& voxel : top_voxels) {
		for (const auto& point : voxel) {
			int x_idx = static_cast<int>((point.x - minX) / voxelWidth_filter);
			int y_idx = static_cast<int>((point.y - minY) / voxelWidth_filter);
			int voxel_idx = y_idx * voxel_num_in_x_filter + x_idx;
			voxels_filter[voxel_idx].push_back(point);
		}
	}
}

void CleanUpPointsByZThreshold(std::vector<std::vector<cv::Point3f>>& voxels_filter, float z_floor, float z_ceil, double z_threshold, float voxelwidth_filter,
	std::vector<std::vector<cv::Point3f>>& voxels_filter_out, std::vector<std::vector<cv::Point3f>>& voxels_filter_window) {
	// Implement logic to classify points by Z values
	voxels_filter_out = voxels_filter;
	voxels_filter_window = voxels_filter;

	int zno = 0;
	std::cout << "z_floor: " << z_floor << "z_ceiling: " << z_ceil << std::endl;
	for (int i = 0; i < voxels_filter.size(); i++)
	{
		if (voxels_filter[i].size() > 0)
		{
			float z_max = voxels_filter[i][0].z;
			float z_min = voxels_filter[i][0].z;
			vector<cv::Point3f> pt_windows;
			vector<cv::Point3f> pt_windows1;
			vector<cv::Point3f> pt_windows2;
			vector<cv::Point3f> pt_windows_up;
			vector<cv::Point3f> pt_windows_down;

			for (const auto& point : voxels_filter[i]) {
				z_min = min(z_min, point.z);
				z_max = max(z_max, point.z);
				if (point.z > z_floor + 1000 && point.z < z_floor + 1250)
				{
					pt_windows.push_back(point);
				}
				else if (point.z > z_floor + 1250 && point.z < z_floor + 1500)
				{
					pt_windows1.push_back(point);
				}
				else if (point.z > z_floor + 1600 && point.z < z_floor + 1850)
				{
					pt_windows2.push_back(point);
				}
				if (point.z > z_floor + (z_ceil - z_floor) / 2)
				{
					pt_windows_up.push_back(point);
				}
				else if (point.z <= z_floor + (z_ceil - z_floor) / 2)
				{
					pt_windows_down.push_back(point);
				}
			}
			//门窗关键筛选
			// 检查是否有两个点集合为空
			if ((pt_windows.empty() && pt_windows1.empty()) ||
				(pt_windows.empty() && pt_windows2.empty()) ||
				(pt_windows1.empty() && pt_windows2.empty())) {

				voxels_filter_out[i].clear();
				zno++;
				//continue;
			}
			else if ((!pt_windows.empty() && !pt_windows1.empty()) ||
				(!pt_windows.empty() && !pt_windows2.empty()) ||
				(!pt_windows1.empty() && !pt_windows2.empty()))
			{
				voxels_filter_window[i].clear();
			}
			float z_size = z_max - z_min;
			//噪点关键筛选,手持扫描的无法通过 z_threshold/2.5
			if (z_size < z_threshold / 2.5)
			{
				voxels_filter_out[i].clear();
			}
			else
			{
				if (pt_windows_up.size() > pt_windows_down.size() * 20)
				{
					voxels_filter[i].clear();
				}
				int  count = static_cast<int>(z_size / voxelwidth_filter) + 1;
				std::vector < std::vector<int> > counts_z(count);
				for (const auto& point : voxels_filter[i]) {
					int z_idx = static_cast<int>((point.z - z_min) / voxelwidth_filter);
					counts_z[z_idx].push_back(z_idx);
				}
				int voxel_point_counts_z_sum = 0;
				for (int j = 0; j < counts_z.size(); j++)
				{
					if (counts_z[j].size() > 0)
					{
						voxel_point_counts_z_sum += 1;
					}
				}
				// 如果体素点数小于阈值，则清除该体素,完全是噪点
				if (voxel_point_counts_z_sum < count * 0.1)
				{
					voxels_filter[i].clear();
					voxels_filter_out[i].clear();
					voxels_filter_window[i].clear();
				}
				//voxels_filter用于户型分割，而out负责后期生图，前则不严格，后者严格，户型分割至少要是门窗点，如果是梁不行，门窗点占比至少也有0.3了
				/*else if (voxel_point_counts_z_sum < count * 0.3)
				{
					voxels_filter_out[i].clear();
				}*/

			}

		}
	}

}

void FinalizeOutputImages(const std::vector<std::vector<cv::Point3f>>& voxels_filter_window, float minX, float minY, float  voxelWidth_filter, cv::Mat& outputImage_window) {
	// Implement logic to finalize output images based on filtered voxels
	for (int i = 0; i < voxels_filter_window.size(); i++)
	{
		if (voxels_filter_window[i].size() > 0)
		{
			auto point = voxels_filter_window[i][0];
			int x_idx = static_cast<int>((point.x - minX) / voxelWidth_filter);
			int y_idx = static_cast<int>((point.y - minY) / voxelWidth_filter);
			int x_pixel = x_idx;
			int y_pixel = y_idx;
			cv::Rect roi(x_pixel, y_pixel, 1, 1);
			if (x_pixel < outputImage_window.cols && y_pixel < outputImage_window.rows)
			{
				outputImage_window(roi).setTo(cv::Scalar(255, 255, 255));
			}
			else {
				/*std::cout << "out of range" << std::endl;
				std::cout << "x_pixel: " << x_pixel << ", y_pixel: " << y_pixel << std::endl;*/
			}
		}
	}
}

cv::Mat  ExtractRepeatedPixelsAndSave(
	const std::vector<cv::Mat>& slice_images,
	int repeat_threshold,
	const std::string& output_path)
{
	if (slice_images.empty()) {
		std::cerr << "No images provided for processing." << std::endl;
	}

	// 确保所有图像尺寸相同
	int rows = slice_images[0].rows;
	int cols = slice_images[0].cols;
	for (const auto& img : slice_images) {
		if (img.rows != rows || img.cols != cols) {
			std::cerr << "All images must have the same dimensions." << std::endl;
		}
	}

	// 创建计数矩阵
	cv::Mat count_matrix = cv::Mat::zeros(rows, cols, CV_32S);

	// 遍历每张图像，累加非黑像素
	for (const auto& img : slice_images) {
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (img.at<cv::Vec3b>(r, c) != cv::Vec3b(0, 0, 0)) {
					count_matrix.at<int>(r, c)++;
				}
			}
		}
	}

	// 创建输出图像
	cv::Mat output_image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
	// 创建输出图像
	cv::Mat output_image_re(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
	// 根据计数矩阵生成输出图像
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			if (count_matrix.at<int>(r, c) >= repeat_threshold) {
				output_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0); // 绘制绿色点
			}
			else if (count_matrix.at<int>(r, c) < repeat_threshold && count_matrix.at<int>(r, c) > 1)
			{
				output_image_re.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0); // 绘制绿色点

			}

		}
	}

	// 保存输出图像
	//cv::imwrite(output_path, output_image);
	//cv::imwrite("combined_output_image_re.png", output_image_re);

	std::cout << "Saved combined image: " << output_path << std::endl;
	return output_image;
}
void ClearVoxelsBasedOnOutputImage(
	const cv::Mat& output_image,
	std::vector<std::vector<cv::Point3f>>& voxels,
	float voxelWidthMm,
	int voxel_num_in_x)
{
	// 遍历output_image的每个像素
	for (int y = 0; y < output_image.rows; ++y) {
		for (int x = 0; x < output_image.cols; ++x) {
			// 获取当前像素的颜色值
			cv::Vec3b pixel_value = output_image.at<cv::Vec3b>(y, x);
			//std::cout << "pixel_value: " << pixel_value << std::endl;

			// 检查像素值是否为0, 0, 0（黑色）
			if (pixel_value == cv::Vec3b(0, 0, 0)) {
				// 计算对应的体素索引
				int voxel_idx = y * voxel_num_in_x + x;
				//std::cout << "voxel_idx: " << voxel_idx << std::endl;

				// 确保索引有效，清空对应的voxels元素
				if (voxel_idx >= 0 && voxel_idx < voxels.size()) {
					voxels[voxel_idx].clear(); // 清空该体素中的点
					//std::cout << "voxels[voxel_idx]: " << voxels[voxel_idx] << std::endl;

				}
				else {
					std::cerr << "Error: voxel_idx out of bounds: " << voxel_idx << std::endl;
				}
			}
		}
	}
}
// 示例调用函数
void GenerateAndSaveOverheadImages(
	const std::vector<cv::Point3f>& plane_points,
	std::vector<std::vector<cv::Point3f>>& voxels,
	float voxelWidthMm,
	float z_floor,
	float z_ceil,
	int repeat_threshold,
	const std::string& output_folder)
{

	if ((z_ceil - z_floor) > 3700)
	{
		z_ceil -= 1150;
		z_floor += 900;
	}
	else
	{
		z_ceil -= 50;
		z_floor += 900;
	}

	int num_slices = 10; // 每10%一个切片
	float z_interval = (z_ceil - z_floor) / num_slices;

	// 找到点云的最小和最大X、Y值
	float minX = plane_points[0].x, maxX = plane_points[0].x;
	float minY = plane_points[0].y, maxY = plane_points[0].y;

	for (const auto& point : plane_points) {
		minX = std::min(minX, point.x);
		maxX = std::max(maxX, point.x);
		minY = std::min(minY, point.y);
		maxY = std::max(maxY, point.y);
	}
	int voxel_num_in_x = static_cast<int>((maxX - minX) / voxelWidthMm) + 10;
	int voxel_num_in_y = static_cast<int>((maxY - minY) / voxelWidthMm) + 10;
	// 计算图像尺寸
	int image_width = static_cast<int>((maxX - minX) / voxelWidthMm) + 10;
	int image_height = static_cast<int>((maxY - minY) / voxelWidthMm) + 10;

	std::vector<cv::Mat> slice_images;

	for (int i = 0; i <= num_slices; ++i) {
		float z_min = z_floor + i * z_interval;
		float z_max = z_min + z_interval;

		// 创建空白图像
		cv::Mat overhead_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));

		// 绘制点云到图像
		for (const auto& point : plane_points) {
			if (point.z >= z_min && point.z < z_max) {
				int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
				int y_idx = static_cast<int>((point.y - minY) / voxelWidthMm);
				int voxel_idx = y_idx * voxel_num_in_x + x_idx; // Use the correct X dimension size
				if (x_idx >= 0 && x_idx < image_width && y_idx >= 0 && y_idx < image_height) {
					overhead_image.at<cv::Vec3b>(y_idx, x_idx) = cv::Vec3b(0, 255, 0); // 绘制绿色点
				}
			}
		}

		// 保存切片图像
		std::string file_name = output_folder + "slice_" + std::to_string(i) + ".png";
		//cv::imwrite(file_name, overhead_image);
		//std::cout << "Saved slice image: " << file_name << std::endl;

		// 收集切片图像
		slice_images.push_back(overhead_image);
	}

	// 合并多张切片图像
	std::string combined_image_path = output_folder + "combined.png";
	cv::Mat output_image = ExtractRepeatedPixelsAndSave(slice_images, repeat_threshold, combined_image_path);

	// 清空体素中的点
	ClearVoxelsBasedOnOutputImage(output_image, voxels, voxelWidthMm, voxel_num_in_x);

}

void GenerateAndSaveOverheadImages(
	const std::vector<cv::Point3f>& plane_points,
	float voxelWidthMm,
	float z_floor,
	float z_ceil,
	const std::string& output_folder)
{
	int num_slices = 10; // 每10%一个切片
	float z_interval = (z_ceil - z_floor) / num_slices;

	// 找到点云的最小和最大X、Y值
	float minX = plane_points[0].x, maxX = plane_points[0].x;
	float minY = plane_points[0].y, maxY = plane_points[0].y;
	for (const auto& point : plane_points) {
		minX = std::min(minX, point.x);
		maxX = std::max(maxX, point.x);
		minY = std::min(minY, point.y);
		maxY = std::max(maxY, point.y);
	}

	// 计算图像尺寸
	int image_width = static_cast<int>((maxX - minX) / voxelWidthMm) + 1;
	int image_height = static_cast<int>((maxY - minY) / voxelWidthMm) + 1;

	for (int i = 0; i <= num_slices; ++i) {
		float z_min = z_floor + i * z_interval;
		float z_max = z_min + z_interval;

		// 创建空白图像
		cv::Mat overhead_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));

		// 绘制点云到图像
		for (const auto& point : plane_points) {
			if (point.z >= z_min && point.z < z_max) {
				int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
				int y_idx = static_cast<int>((point.y - minY) / voxelWidthMm);
				if (x_idx >= 0 && x_idx < image_width && y_idx >= 0 && y_idx < image_height) {
					overhead_image.at<cv::Vec3b>(y_idx, x_idx) = cv::Vec3b(0, 255, 0); // 绘制绿色点
				}
			}
		}

		// 保存图像
		std::string file_name = output_folder + "slice_" + std::to_string(i) + ".png";
		//cv::imwrite(file_name, overhead_image);
		//std::cout << "Saved slice image: " << file_name << std::endl;
	}
}


std::vector<std::vector<cv::Point3f>> VoxelizePlanePointsByDensity_02(
	const std::vector<cv::Point3f>& plane_points,
	int door_size,
	float voxelWidthMm,
	float voxelWidth_filter,
	float radio,
	double z_threshold,
	cv::Mat& outputImage,
	cv::Mat& outputImage_nowindow,
	cv::Mat& outputImage_window,
	float& z_floor,
	float& z_ceil,
	cv::Point& base_point,
	double& minx,
	double& miny
) {
	std::vector<std::vector<cv::Point3f>> voxels, top_voxels, top_voxels2, voxels_filter, voxels_filter_out, voxels_filter_window;
	std::vector<std::vector<cv::Point3f>> voxels_filter1, voxels_filter_out1, voxels_filter_window1;

	// Step 1: Find min and max coordinates
	float minX, maxX, minY, maxY, minZ, maxZ;
	FindMinMaxCoordinates(plane_points, minX, maxX, minY, maxY, minZ, maxZ);

	//std::cout << " Step 2: Voxelization: "<< std::endl;
	int voxel_num_in_x = static_cast<int>((maxX - minX) / voxelWidth_filter) + 10;
	int voxel_num_in_y = static_cast<int>((maxY - minY) / voxelWidth_filter) + 10;
	voxels.resize(voxel_num_in_x * voxel_num_in_y);
	VoxelizePoints(plane_points, minX, minY, voxelWidth_filter, voxels, voxel_num_in_x);
	GenerateAndSaveOverheadImages(plane_points, voxels, voxelWidth_filter, z_floor, z_ceil, 3, "output_images");

	////生成唯有水平面的俯视图
	//cv::Mat outputImage0;
	//outputImage0 = cv::Mat(voxel_num_in_y, voxel_num_in_x, CV_8UC3, cv::Scalar(0, 0, 0));
	//FinalizeOutputImages(voxels, minX, minY, voxelWidth_filter, outputImage0);
	//std::string file_name =  "outputImage0.png";
	//cv::imwrite(file_name, outputImage0);
	//std::cout << "outputImage0 slice image: " << "outputImage0.png" << std::endl;

	std::cout << "Step 3: Filter voxels based on density " << std::endl;
	//FilterVoxelsByDensity(voxels, 0.9, voxelWidthMm, top_voxels);
	//top_voxels = voxels;
	//top_voxels = voxels;
	for (size_t i = 0; i < voxels.size(); ++i) {
		if (voxels[i].size() > 5) {
			top_voxels.push_back(voxels[i]);
		}
	}
	std::cout << "Step 4: Calculate new bounds and prepare for filtered voxelization " << std::endl;
	UpdateBoundsForFilteredVoxels(top_voxels, door_size, voxelWidth_filter, minX, maxX, minY, maxY, minZ, maxZ, minx, miny);


	std::cout << "Step 5: Create filtered voxel grid" << std::endl;
	CreateFilteredVoxelGrid(top_voxels, maxX, maxY, minX, minY, voxelWidth_filter * 5, top_voxels2);
	//CreateFilteredVoxelGrid(top_voxels, maxX, maxY, minX, minY, voxelWidth_filter, voxels_filter);

	std::cout << " Step 6: Classify and clean up points based on Z values" << std::endl;
	CleanUpPointsByZThreshold(top_voxels2, z_floor, z_ceil, z_threshold, voxelWidth_filter * 5, voxels_filter_out, voxels_filter_window);
	//voxels_filter = top_voxels;
	//voxels_filter_out = top_voxels;
	//voxels_filter_window = top_voxels;

	std::cout << "Step 5: Create filtered voxel grid" << std::endl;
	CreateFilteredVoxelGrid(top_voxels2, maxX, maxY, minX, minY, voxelWidth_filter, voxels_filter1);
	CreateFilteredVoxelGrid(voxels_filter_out, maxX, maxY, minX, minY, voxelWidth_filter, voxels_filter_out1);
	CreateFilteredVoxelGrid(voxels_filter_window, maxX, maxY, minX, minY, voxelWidth_filter, voxels_filter_window1);



	std::cout << "Step 7: Finalize output images and return filtered voxels " << std::endl;
	int voxel_num_in_x_filter = static_cast<int>((maxX - minX) / voxelWidth_filter);
	int voxel_num_in_y_filter = static_cast<int>((maxY - minY) / voxelWidth_filter);
	outputImage = cv::Mat(voxel_num_in_y_filter, voxel_num_in_x_filter, CV_8UC3, cv::Scalar(0, 0, 0));
	outputImage_nowindow = cv::Mat(voxel_num_in_y_filter, voxel_num_in_x_filter, CV_8UC3, cv::Scalar(0, 0, 0));
	outputImage_window = cv::Mat(voxel_num_in_y_filter, voxel_num_in_x_filter, CV_8UC3, cv::Scalar(0, 0, 0));
	FinalizeOutputImages(voxels_filter1, minX, minY, voxelWidth_filter, outputImage);
	FinalizeOutputImages(voxels_filter_out1, minX, minY, voxelWidth_filter, outputImage_nowindow);
	FinalizeOutputImages(voxels_filter_window1, minX, minY, voxelWidth_filter, outputImage_window);
	return voxels_filter;
}


std::pair< std::vector<Vec3b>, std::vector<Point>> checkFourDirections(const Mat& image, Point center) {
	Vec3b blackColor = Vec3b(0, 0, 0);
	Vec3b whiteColor = Vec3b(255, 255, 255);

	std::vector<Vec3b> directions(4);
	std::vector<Point> locations(4);

#pragma omp parallel for schedule(static)
	for (int dir = 0; dir < 4; ++dir) {
		int dy[] = { -1, 1, 0, 0 };
		int dx[] = { 0, 0, -1, 1 };

		int y = center.y;
		int x = center.x;
		while (true) {
			y += dy[dir];
			x += dx[dir];

			if (y < 0 || y >= image.rows || x < 0 || x >= image.cols) {
				directions[dir] = blackColor;
				locations[dir] = Point(x, y);
				break;
			}

			Vec3b pixel = image.at<Vec3b>(y, x);
			if (pixel != blackColor) {
				directions[dir] = pixel;
				locations[dir] = Point(x, y);
				break;
			}
			else
			{
				locations[dir] = Point(x, y);
				continue;
			}
		}

	}

	return std::make_pair(directions, locations);
}
double distanceBetweenPoints(const Point& p1, const Point& p2) {
	return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

Vec3b processDirections(const std::vector<Vec3b>& directions, const std::vector<Point>& locations, Point center, int distanceThreshold, int distance_min) {
	Vec3b blackColor = Vec3b(0, 0, 0);
	Vec3b whiteColor = Vec3b(255, 255, 255);

	int whiteCount = 0;
	int nonBlackNonWhiteCount = 0;
	std::vector<Vec3b> nonBlackNonWhiteColors;
	std::vector<double> nonBlackNonWhiteDistances;
	std::vector<double> BlackWhiteDistances;

#pragma omp parallel for schedule(static) reduction(+:whiteCount,nonBlackNonWhiteCount)
	for (size_t i = 0; i < directions.size(); ++i) {
		const Vec3b& d = directions[i];
		const Point& loc = locations[i];

		if (d == whiteColor)
#pragma omp critical
		{
			{
				++whiteCount;
				BlackWhiteDistances.push_back(distanceBetweenPoints(loc, center));
			}

		}
		else if (d != whiteColor && d != blackColor)
#pragma omp critical
		{
			++nonBlackNonWhiteCount;
			nonBlackNonWhiteColors.push_back(d);
			nonBlackNonWhiteDistances.push_back(distanceBetweenPoints(loc, center));
		}
	}

	if (whiteCount == 3 && nonBlackNonWhiteCount == 1) {
		bool allWithinDistance = true;
		for (size_t i = 0; i < directions.size(); ++i) {
			if (directions[i] == whiteColor) {
				if (nonBlackNonWhiteDistances[i] > distanceThreshold) {
					allWithinDistance = false;
				}
			}
		}

		if (allWithinDistance) {
			for (const Vec3b& d : nonBlackNonWhiteColors) {
				if (d != whiteColor && d != blackColor) {
					return d;
				}
			}
		}
	}
	else if (nonBlackNonWhiteCount >= 3) {
		Vec3b firstColor = nonBlackNonWhiteColors.front();
		bool allSameColor = std::all_of(nonBlackNonWhiteColors.begin(), nonBlackNonWhiteColors.end(),
			[&firstColor](const Vec3b& c) { return c == firstColor; });

		if (allSameColor) {
			return firstColor;
		}
		else
		{
			double minDistance = 60;
			Vec3b closestColor = blackColor;
			for (size_t i = 0; i < nonBlackNonWhiteDistances.size(); ++i) {
				if (nonBlackNonWhiteDistances[i] < distanceThreshold) {
					minDistance = nonBlackNonWhiteDistances[i];
					closestColor = nonBlackNonWhiteColors[i];
				}
			}
			return closestColor;
		}
	}
	else if (whiteCount == 2 && nonBlackNonWhiteCount == 2) {
		double minDistance = 60;
		Vec3b closestColor = blackColor;
		for (size_t i = 0; i < nonBlackNonWhiteDistances.size(); ++i) {
			if (nonBlackNonWhiteDistances[i] < distanceThreshold) {
				minDistance = nonBlackNonWhiteDistances[i];
				closestColor = nonBlackNonWhiteColors[i];
			}
		}
		return closestColor;
	}

	return blackColor;
}


Vec3b processDirections_only(const std::vector<Vec3b>& directions, const std::vector<Point>& locations, Point center, int distanceThreshold, int distance_min) {
	Vec3b blackColor = Vec3b(0, 0, 0);
	Vec3b whiteColor = Vec3b(255, 255, 255);

	int whiteCount = 0;
	int nonBlackNonWhiteCount = 0;
	std::vector<Vec3b> nonBlackNonWhiteColors;
	std::vector<double> nonBlackNonWhiteDistances;
	std::vector<double> BlackWhiteDistances;


	for (size_t i = 0; i < directions.size(); ++i) {
		const Vec3b& d = directions[i];
		const Point& loc = locations[i];

		if (d == whiteColor) {
			++whiteCount;
			BlackWhiteDistances.push_back(distanceBetweenPoints(loc, center));
		}
		else if (d != whiteColor && d != blackColor) {
			++nonBlackNonWhiteCount;
			nonBlackNonWhiteColors.push_back(d);
			nonBlackNonWhiteDistances.push_back(distanceBetweenPoints(loc, center));
		}
	}


	if (nonBlackNonWhiteCount == 4) {
		Vec3b firstColor = nonBlackNonWhiteColors.front();
		bool allSameColor = std::all_of(nonBlackNonWhiteColors.begin(), nonBlackNonWhiteColors.end(),
			[&firstColor](const Vec3b& c) { return c == firstColor; });
		if (allSameColor)
		{
			return firstColor;

		}
	}
	return blackColor;
}


bool CheckInputDataUnit(const std::vector<cv::Point3f>& data)
{
	int64_t size = data.size() > 10000 ? 10000 : data.size();
	int out = 0;
	for (int i = 0; i < size; i++)
	{
		if (std::abs(data[i].x) > 50.f || std::abs(data[i].y) > 50.f || std::abs(data[i].z) > 50.f)
		{
			out++;
		}
	}

	return out > (0.8 * size) ? true : false;
}

bool MeterToMinimeterConvertion(const std::vector<cv::Point3f>& data, std::vector<cv::Point3f>& out)
{
	bool ret = CheckInputDataUnit(data);
	if (!ret)
	{
		out = data;
#pragma omp parallel for
		for (int i = 0; i < out.size(); i++) {
			out[i].x *= 1000.f;
			out[i].y *= 1000.f;
			out[i].z *= 1000.f;
		}
	}
	return ret;
}


bool isNonIsolatedPixel(const cv::Mat& input_image, int row, int col, int n, int num, int image_type) {
	if (row < 0 || row >= input_image.rows || col < 0 || col >= input_image.cols) {
		return false;
	}
	if (image_type == 1)
	{
		int count = 0;

		for (int i = -n; i <= n; ++i) {
			for (int j = -n; j <= n; ++j) {
				int newRow = row + i;
				int newCol = col + j;

				if (newRow < 0 || newRow >= input_image.rows || newCol < 0 || newCol >= input_image.cols) {
					continue;
				}

				if (input_image.at<uchar>(newRow, newCol) != 0) {
					count++;
				}
			}
		}

		return count > num;
	}
	else
	{
		int count = 0;

		for (int i = -n; i <= n; ++i) {
			for (int j = -n; j <= n; ++j) {
				int newRow = row + i;
				int newCol = col + j;

				if (newRow < 0 || newRow >= input_image.rows || newCol < 0 || newCol >= input_image.cols) {
					continue;
				}

				if (input_image.at<cv::Vec3b>(newRow, newCol)[0] != 0 &&
					input_image.at<cv::Vec3b>(newRow, newCol)[1] != 0 &&
					input_image.at<cv::Vec3b>(newRow, newCol)[2] != 0) {
					count++;
				}
			}
		}

		return count > num;
	}

}

cv::Mat removeIsolatedPixels(const cv::Mat& input_image, int n, int num, int image_type) {
	cv::Mat new_image(input_image.size(), input_image.type(), cv::Scalar(0, 0, 0));
	if (image_type == 1)
	{
		for (int row = 0; row < input_image.rows - 1; ++row) {
			for (int col = 0; col < input_image.cols - 1; ++col) {

				cv::Vec3b pixel = input_image.at<uchar>(row, col);
				if (pixel[0] != 0) {
					if (isNonIsolatedPixel(input_image, row, col, n, num, image_type)) {
						new_image.at<uchar>(row, col) = input_image.at<uchar>(row, col);
					}
				}

			}
		}
	}
	else
	{
		for (int row = 0; row < input_image.rows - 1; ++row) {
			for (int col = 0; col < input_image.cols - 1; ++col) {

				cv::Vec3b pixel = input_image.at<cv::Vec3b>(row, col);
				if (pixel[0] != 0 && pixel[1] != 0 && pixel[2] != 0) {
					if (isNonIsolatedPixel(input_image, row, col, n, num, image_type)) {
						new_image.at<cv::Vec3b>(row, col) = input_image.at<cv::Vec3b>(row, col);
					}
				}

			}
		}
	}


	return new_image;
}


vector<RoomStruct> updateWhitePixelsWithNearestColor(Mat binaryImage, Mat colorImage, Mat& outputImage, int dis) {

	vector<RoomStruct> colorToPointMap;

	outputImage = Mat::zeros(colorImage.size(), colorImage.type());

	vector<Point> whitePixels;
	if (binaryImage.type() != CV_8UC1) {
		if (binaryImage.channels() > 1) {
			cv::Mat grayImage;
			cv::cvtColor(binaryImage, binaryImage, cv::COLOR_BGR2GRAY);
			cv::threshold(binaryImage, binaryImage, 127, 255, cv::THRESH_BINARY);
		}
		else {
			binaryImage.convertTo(binaryImage, CV_8UC1);
		}
	}
	else {
		binaryImage = binaryImage;
	}


	findNonZero(binaryImage, whitePixels);
	std::cout << "whitePixels.size():" << whitePixels.size() << std::endl;

	for (const Point& whitePixel : whitePixels) {
		int bestMatchX = -1;
		int bestMatchY = -1;
		double minDistance = numeric_limits<double>::max();

		for (int dy = -dis; dy <= dis; ++dy) {
			for (int dx = -dis; dx <= dis; ++dx) {
				int y = whitePixel.y + dy;
				int x = whitePixel.x + dx;

				if (y < 0 || y >= colorImage.rows || x < 0 || x >= colorImage.cols) {
					continue;
				}

				Vec3b pixel = colorImage.at<Vec3b>(y, x);

				if (pixel != Vec3b(0, 0, 0) && pixel != Vec3b(255, 255, 255)) {
					double dist = distanceBetweenPoints(Point(x, y), whitePixel);
					if (dist < minDistance) {
						minDistance = dist;
						bestMatchX = x;
						bestMatchY = y;
					}
				}
			}
		}

		if (bestMatchX != -1 && bestMatchY != -1) {
			Vec3b bestMatchColor = colorImage.at<Vec3b>(bestMatchY, bestMatchX);
			outputImage.at<Vec3b>(whitePixel.y, whitePixel.x) = bestMatchColor;
			if (colorToPointMap.size() == 0)
			{
				RoomStruct room;
				room.roomPts.push_back(whitePixel);
				room.roomColor = bestMatchColor;
				colorToPointMap.push_back(room);
			}
			else
			{
				bool isExist = false;
				for (int i = 0; i < colorToPointMap.size(); i++)
				{
					if (colorToPointMap[i].roomColor == bestMatchColor) {
						colorToPointMap[i].roomPts.push_back(whitePixel);
						isExist = true;
						break;
					}
				}
				if (!isExist)
				{
					RoomStruct room;
					room.roomPts.push_back(whitePixel);
					room.roomColor = bestMatchColor;
					colorToPointMap.push_back(room);
				}

			}
		}
	}
	return colorToPointMap;
}


std::vector<std::vector<cv::Point3f>> VoxelizePlanePointsByDensity_XZ(
	const std::vector<cv::Point3f>& plane_points,
	float voxelWidthMm,
	float radio,
	cv::Mat& outputImage,
	bool reverse, float& z_floor,
	float& z_ceil)

{
	std::vector<std::vector<cv::Point3f>> voxels;
	std::vector<std::vector<cv::Point3f>> voxels_out;

	float minX = plane_points[0].x, maxX = plane_points[0].x;
	float minZ = plane_points[0].z, maxZ = plane_points[0].z;
	float minY = plane_points[0].y, maxY = plane_points[0].y;

	for (const auto& point : plane_points) {
		minX = min(minX, point.x);
		maxX = max(maxX, point.x);
		minZ = min(minZ, point.z);
		maxZ = max(maxZ, point.z);
		minY = min(minY, point.y);
		maxY = max(minY, point.y);

	}

	/*std::cout << "minX: " << minX << ", maxX: " << maxX << std::endl;
	std::cout << "minZ: " << minZ << ", maxZ: " << maxZ << std::endl;*/

	int voxel_num_in_x = static_cast<int>((maxX - minX) / voxelWidthMm) + 10;
	int voxel_num_in_y = static_cast<int>((maxZ - minZ) / voxelWidthMm) + 10;
	int voxel_num_in_z = static_cast<int>((maxY - minY) / voxelWidthMm) + 10;

	//std::cout << "voxel_num_in_x: " << voxel_num_in_x << std::endl;
	//std::cout << "voxel_num_in_y: " << voxel_num_in_y << std::endl;

	voxels.resize(voxel_num_in_x * voxel_num_in_y);
	voxels_out.resize(voxel_num_in_x * voxel_num_in_z);
	outputImage = cv::Mat(voxel_num_in_z, voxel_num_in_x, CV_8UC3, cv::Scalar(0, 0, 0));



	for (const auto& point : plane_points) {
		int y_idx = static_cast<int>((point.z - minZ) / voxelWidthMm);
		int voxel_idx = y_idx * voxel_num_in_x;
		voxels[voxel_idx].push_back(point);
	}

	std::vector<int> voxel_point_counts(voxels.size());
	for (size_t i = 0; i < voxels.size(); ++i) {

		voxel_point_counts[i] = static_cast<int>(voxels[i].size() / voxelWidthMm);
	}
	std::vector<int> non_empty_voxel_counts;
	for (int count : voxel_point_counts) {
		if (count > 50) {
			non_empty_voxel_counts.push_back(count);
		}
	}

	std::sort(non_empty_voxel_counts.begin(), non_empty_voxel_counts.end(), std::greater<int>());

	int num_voxels = non_empty_voxel_counts.size();
	int threshold_index = max(static_cast<int>(num_voxels * radio), 1);
	int threshold_point_count = non_empty_voxel_counts[threshold_index] * voxelWidthMm;
	/*std::cout << "num_voxels: " << num_voxels << std::endl;
	std::cout << "Threshold index: " << threshold_index << std::endl;
	std::cout << "Threshold point count: " << threshold_point_count << std::endl;*/
	std::vector<std::vector<cv::Point3f>> top_voxels;
	std::vector<std::vector<cv::Point3f>> bottom_voxels;

	if (reverse)
	{
		//for (size_t i = 0; i < voxels.size(); ++i) {
		//	if (voxels[i].size() <= threshold_point_count) {
		//		top_voxels.push_back(voxels[i]);
		//	}
		//	else if (voxels[i].size() > threshold_point_count)
		//	{
		//		bottom_voxels.push_back(voxels[i]);
		//	}
		//}

		//for (size_t i = 0; i < bottom_voxels.size(); i++)
		//{
		//	auto point = bottom_voxels[i][0];
		//	{
		//		int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
		//		int y_idx = static_cast<int>((point.z - minZ) / voxelWidthMm);
		//		int voxel_idx = y_idx * voxel_num_in_x + x_idx;

		//		int x_pixel = x_idx * voxelWidthMm;
		//		int y_pixel = y_idx * voxelWidthMm;

		//		cv::Rect roi(x_pixel, y_pixel, voxelWidthMm, voxelWidthMm);
		//		outputImage(roi).setTo(cv::Scalar(0, 255, 0));

		//	}
		//}

		//std::cout << "top_voxels size: " << top_voxels.size() << std::endl;
	}
	else
	{
		for (size_t i = 0; i < voxels.size(); ++i) {
			if (voxels[i].size() >= threshold_point_count) {
				top_voxels.push_back(voxels[i]);
			}
			else
			{
				bottom_voxels.push_back(voxels[i]);
			}
		}

		/*for (const auto& point : plane_points) {
			int x_idx = static_cast<int>((point.x - minX) / voxelWidthMm);
			int z_idx = static_cast<int>((point.z - minZ) / voxelWidthMm);
			int y_idx = static_cast<int>((point.y - minY) / voxelWidthMm);

			int voxel_idx = z_idx * voxel_num_in_x + x_idx;

			int x_pixel = x_idx * voxelWidthMm;
			int z_pixel = z_idx * voxelWidthMm;
			int y_pixel = y_idx * voxelWidthMm;


			if (voxels[voxel_idx].size() >= threshold_point_count) {
				cv::Rect roi(x_pixel, y_pixel, voxelWidthMm, voxelWidthMm);
				outputImage(roi).setTo(cv::Scalar(0, 255, 0));
			}
		}*/

		//std::cout << "top_voxels size: " << top_voxels.size() << std::endl;
	}
	//cv::imwrite("top_voxels_output_image.png", outputImage);
	//std::cout << "----VoxelizePlanePointsByDensity_XZ end---- " << std::endl;

	/*std::vector<cv::Point3f> all_voxels;
	for (size_t i = 0; i < top_voxels.size(); i++)
	{
		all_voxels.insert(all_voxels.end(), top_voxels[i].begin(), top_voxels[i].end());
	}*/
	//std::cout << "points size: " << all_voxels.size() << std::endl;
	z_floor = top_voxels[0][0].z;
	z_ceil = top_voxels[0][0].z;
	for (size_t i = 0; i < top_voxels.size(); i++)
	{
		z_floor = min(top_voxels[i][0].z, z_floor);
		z_ceil = max(top_voxels[i][0].z, z_ceil);
	}

	return bottom_voxels;
}


Mat preprocessImage(const Mat& src, int thresholdValue) {
	Mat gray, binaryImage;
	if (src.channels() > 1) {
		cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		gray = src.clone();
	}
	threshold(gray, binaryImage, thresholdValue, 255, THRESH_BINARY);
	return binaryImage;
}

Mat binarizeImageWithGrid(const Mat& binaryImage, int gridSize) {
	Mat resultImage = Mat::zeros(binaryImage.size(), CV_8UC1);
	for (int y = 0; y < binaryImage.rows - gridSize; y += gridSize) {
		for (int x = 0; x < binaryImage.cols - gridSize; x += gridSize) {
			cv::Rect cell(x, y, gridSize, gridSize);
			if (cv::countNonZero(binaryImage(cell)) > 0) {
				resultImage(cell).setTo(255);
			}
		}
	}
	return resultImage;
}

Mat applyMorphology(const Mat& input, int doorSize, int erodeSize) {
	Mat dilated, eroded;
	Mat doorElement = getStructuringElement(MORPH_RECT, Size(doorSize, doorSize));
	Mat erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(erodeSize, erodeSize));

	dilate(input, dilated, doorElement);
	//cv::imwrite(" 03 dilatedImage.png", dilated);
	erode(dilated, eroded, erodeElement);
	return eroded;
}

vector<vector<Point>> processContours(const Mat& image, vector<Vec4i>& hierarchy) {
	vector<vector<Point>> contours;
	findContours(image, contours, hierarchy, RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, Point());
	return contours;
}

vector<RoomStruct> updateRoomMap(
	const Mat& srcNowindowImage, const Mat& segmentedImage, Mat& outputImage,
	const vector<pair<Scalar, vector<Point>>>& contourColorPairs, int doorHalfSize)
{
	vector<RoomStruct> RoomMap = updateWhitePixelsWithNearestColor(srcNowindowImage, segmentedImage, outputImage, doorHalfSize);

	for (size_t i = 0; i < RoomMap.size(); i++) {
		RoomMap[i].roomId = i;
	}
	for (size_t i = 0; i < RoomMap.size(); i++) {
		for (const auto& pair : contourColorPairs) {
			auto color = pair.first;
			auto points = pair.second;
			if (color[0] == RoomMap[i].roomColor[0] && color[1] == RoomMap[i].roomColor[1] && color[2] == RoomMap[i].roomColor[2]) {
				RoomMap[i].roomContours.insert(RoomMap[i].roomContours.end(), points.begin(), points.end());
				break;
			}
		}
	}
	return RoomMap;
}
// 检查单个区域是否全是黑色
bool isCellBlack(const cv::Mat& image, const cv::Rect& cell) {
	// 确保矩形区域不会超出图像边界
	cv::Rect validCell = cell & cv::Rect(0, 0, image.cols, image.rows);

	// 遍历该区域的每个像素
	for (int y = validCell.y; y < validCell.y + validCell.height; ++y) {
		for (int x = validCell.x; x < validCell.x + validCell.width; ++x) {
			// 检查像素值是否为非黑色
			if (image.at<uchar>(y, x) != 0) {
				return false; // 如果发现非黑像素，立即返回 false
			}
		}
	}

	// 如果所有像素都为黑色，返回 true
	return true;
}
// Main segmentation function
vector<RoomStruct> region_seg_image_02(
	Mat src, Mat src_nowindow, int thresholdValue = 30, int gridSize = 5, int door_half_size = 60, int eroded_img_size = 10)
{
	// Step 1: Preprocess input images
	Mat binaryImage = preprocessImage(src, thresholdValue);
	Mat binaryImageNowindow = preprocessImage(src_nowindow, thresholdValue);

	// Step 2: Binarize image with grid
	Mat gridBinarized = binarizeImageWithGrid(binaryImage, 1);

	// Step 3: Apply morphology to extract regions
	Mat erodedImage = applyMorphology(gridBinarized, door_half_size, eroded_img_size);
	//cv::imwrite(" 04 eroded_img.png", erodedImage);

	// Step 4: Detect and process contours
	vector<Vec4i> hierarchy;
	vector<vector<Point>> contours = processContours(erodedImage, hierarchy);
	vector<double> areas(contours.size());
	// 删除最大轮廓
	//for (size_t i = 0; i < contours.size(); ++i) {
	//	areas[i] = contourArea(contours[i]);
	//}
	//vector<pair<int, double>> indexAreaPairs;
	//for (size_t i = 0; i < contours.size(); ++i) {
	//	indexAreaPairs.emplace_back(i, areas[i]);
	//}
	//sort(indexAreaPairs.begin(), indexAreaPairs.end(),
	//	[](const pair<int, double>& a, const pair<int, double>& b) {
	//		return a.second > b.second;
	//	});
	//int largestIndex = -1, secondLargestIndex = -1;
	//if (!indexAreaPairs.empty()) {
	//	largestIndex = indexAreaPairs[0].first;
	//}
	//if (indexAreaPairs.size() > 1) {
	//	secondLargestIndex = indexAreaPairs[1].first;
	//}

	// Step 5: Segment regions and assign colors
	Mat segmentedImage = Mat::zeros(erodedImage.size(), CV_8UC3);
	vector<pair<Scalar, vector<Point>>> contourColorPairs;

	// Assign random colors to contours
	vector<Scalar> colors(contours.size());
	for (int i = 0; i < contours.size(); ++i) {
		if (hierarchy[i][3] != -1)
		{
			colors[i] = Scalar(rand() % 256, rand() % 256, rand() % 256);
			if (contourArea(contours[i]) > door_half_size * door_half_size / 2) {
				vector<Point> points;
				fillPoly(segmentedImage, vector<vector<Point>>{contours[i]}, colors[i]);
				for (int y = 0; y < segmentedImage.rows - 1; y += 1) {
					for (int x = 0; x < segmentedImage.cols - 1; x += 1) {
						Point2f pt(y, x); // 测试点
						Point2f pt2(x, y); // 测试点

						if (Scalar(segmentedImage.at<Vec3b>(y, x)[0], segmentedImage.at<Vec3b>(y, x)[1], segmentedImage.at<Vec3b>(y, x)[2]) == colors[i])
						{
							points.push_back(pt2);
						}
					}
				}
				contourColorPairs.emplace_back(colors[i], points);

			}
		}
	}

	//cv::imwrite(" 05 imageContours.png", segmentedImage);


	for (int y = 0; y < binaryImage.rows - 1; y += 1) {
		for (int x = 0; x < binaryImage.cols - 1; x += 1) {
			cv::Rect cell(x, y, 1, 1);
			try
			{
				bool hasNonZero = cv::countNonZero(binaryImage(cell)) > 0;
				if (hasNonZero) {
					segmentedImage(cell).setTo(cv::Scalar(255, 255, 255));
				}
			}
			catch (const std::exception&)
			{

			}
		}
	}
	//cv::imwrite(" 05 imageContours2.png", segmentedImage);
	Vec3b blackColor = Vec3b(0, 0, 0);
	Vec3b whiteColor = Vec3b(255, 255, 255);
	Mat resultImage_output = Mat::zeros(segmentedImage.size(), CV_8UC3);
	for (int y = 0; y < segmentedImage.rows; y += 1) {
		for (int x = 0; x < segmentedImage.cols; x += 1) {
			Vec3b& pixel = segmentedImage.at<Vec3b>(y, x);
			Vec3b& pixel2 = resultImage_output.at<Vec3b>(y, x);
			if (pixel != blackColor && pixel != whiteColor) {

				pixel2.val[0] = pixel.val[0];
				pixel2.val[1] = pixel.val[1];
				pixel2.val[2] = pixel.val[2];
				continue;
			}
			else if (pixel == whiteColor) {
				continue;
			}
		}
	}


	for (int y = 0; y < segmentedImage.rows - 3; y += 3) {
		for (int x = 0; x < segmentedImage.cols - 3; x += 3) {
			cv::Rect cell(x, y, 3, 3);
			Vec3b& pixel = segmentedImage.at<Vec3b>(y, x);
			Vec3b& pixel2 = resultImage_output.at<Vec3b>(y, x);
			if (isCellBlack(resultImage_output, cell))
			{
				if (pixel != blackColor && pixel != whiteColor) {

					pixel2.val[0] = pixel.val[0];
					pixel2.val[1] = pixel.val[1];
					pixel2.val[2] = pixel.val[2];
					continue;
				}
				else if (pixel == blackColor) {
					Point currentPixel(x, y);
					auto a = checkFourDirections(segmentedImage, currentPixel);
					Vec3b status = processDirections(a.first, a.second, currentPixel, door_half_size, eroded_img_size);

					if (status != blackColor && status != whiteColor)
					{
						pixel2 = status;
						if (status != blackColor && status != whiteColor)
						{
							resultImage_output(cell).setTo(cv::Scalar(status.val[0], status.val[1], status.val[2]));
							//pixel2 = status;
						}
					}
				}
				else if (pixel == whiteColor) {
					continue;
				}
			}
		}
	}
	//for (int y = 0; y < resultImage_output.rows - 5; y += 5) {
	//	for (int x = 0; x < resultImage_output.cols - 5; x += 5) {
	//		cv::Rect cell(x, y, 5, 5);
	//		if (isCellBlack(resultImage_output, cell))
	//		{
	//				Point currentPixel(x, y);
	//				auto a = checkFourDirections(resultImage_output, currentPixel);
	//				Vec3b status = processDirections(a.first, a.second, currentPixel, door_half_size, eroded_img_size);

	//				if (status != blackColor && status != whiteColor)
	//				{
	//					resultImage_output(cell).setTo(cv::Scalar(status.val[0], status.val[1], status.val[2]));
	//					//pixel2 = status;
	//				}
	//			
	//		}
	//		else
	//		{
	//			
	//		}
	//		
	//	}
	//}

	for (int y = 0; y < binaryImage.rows - 1; y += 1) {
		for (int x = 0; x < binaryImage.cols - 1; x += 1) {
			cv::Rect cell(x, y, 1, 1);
			try
			{
				bool hasNonZero = cv::countNonZero(binaryImage(cell)) > 0;
				if (hasNonZero) {
					resultImage_output(cell).setTo(cv::Scalar(255, 255, 255));
				}
			}
			catch (const std::exception&)
			{
				//std::cout << "Could not cell( " << x << "," << y << " ) " << std::endl;
			}
		}
	}

	Mat result_seg_image(resultImage_output.size(), CV_8UC3, Scalar(255, 255, 255));
	//cv::imwrite(" 05 imageContours3.png", resultImage_output);


	//for (int y = 0; y < resultImage_output.rows - 5; y += 5) {
//	for (int x = 0; x < resultImage_output.cols - 5; x += 5) {
//		cv::Rect cell(x, y, 5, 5);
//		if (isCellBlack(resultImage_output, cell))
//		{
//				Point currentPixel(x, y);
//				auto a = checkFourDirections(resultImage_output, currentPixel);
//				Vec3b status = processDirections(a.first, a.second, currentPixel, door_half_size, eroded_img_size);

//				if (status != blackColor && status != whiteColor)
//				{
//					resultImage_output(cell).setTo(cv::Scalar(status.val[0], status.val[1], status.val[2]));
//					//pixel2 = status;
//				}
//			
//		}
//		else
//		{
//			
//		}
//		
//	}
//}
	/*for (int y = 0; y < resultImage_output.rows - 5; y += 5) {
		for (int x = 0; x < resultImage_output.cols - 5; x += 5) {
			cv::Rect cell(x, y, 5, 5);
			Vec3b& pixel = resultImage_output.at<Vec3b>(y, x);
			Vec3b& pixel2 = resultImage_output.at<Vec3b>(y, x);
			if (isCellBlack(resultImage_output, cell))
			{
				Point currentPixel(x, y);
				auto a = checkFourDirections(resultImage_output, currentPixel);
				Vec3b status = processDirections(a.first, a.second, currentPixel, door_half_size, eroded_img_size);

				if (status != blackColor && status != whiteColor)
				{
					resultImage_output(cell).setTo(cv::Scalar(status.val[0], status.val[1], status.val[2]));
				}
			}
		}
	}
	cv::imwrite(" 05 imageContours4.png", resultImage_output);*/

	// Step 6: Update room map with segmented regions
	Mat finalOutput(segmentedImage.size(), CV_8UC3, Scalar(255, 255, 255));
	vector<RoomStruct> RoomMap = updateRoomMap(binaryImageNowindow, resultImage_output, finalOutput, contourColorPairs, door_half_size);
	//cv::imwrite(" 06 区域分割.png", finalOutput);
	// Return final room map
	return RoomMap;
}


cv::Mat* createMatrix() {
	cv::Mat* mat = new cv::Mat(3, 3, CV_8UC1, cv::Scalar(0));
	return mat;
}

// 将 RoomStruct 转换为 MyData

//bool RoomSegment()
//{
//	//-----------------------------------------------0.参数设置----------------------------------------------------
//	// 测量时间
//	int thresholdValue = 30;//二值化阈值
//	int gridSize = 10;//网格大小
//	int door_half_size = 81;//膨胀门宽，如果有面宽小于door_half_size*2 的房间将无法识别
//	int eroded_img_size = 10;//腐蚀大小
//	float vox_xz = 30.0f;
//	float z_ratio = 0.2f;
//	float vox_xy = 25.0f;
//	float xy_ratio = 0.7f;
//	float z_height = 1000.0f;
//	//if (model == 1) {
//	//	thresholdValue = 30;//二值化阈值
//	//	gridSize = 10;//网格大小
//	//	door_half_size = 75;//膨胀门宽，如果有面宽小于door_half_size*2 的房间将无法识别
//	//	eroded_img_size = 10;//腐蚀大小
//	//	vox_xz = 100.0f;
//	//	z_ratio = 0.2f;
//	//	vox_xy = 100.0f;
//	//	xy_ratio = 0.4f;
//	//	z_height = 600.0f;
//	//}
//
//	//-----------------------------------------------1.读取点云----------------------------------------------------
//
//	//E:\\mydata\\0823_data_room\\merge_color_pcd(2)_jiahuiniubi_zhuangzheng.pcd
//	/*scenePath = "E:\\mydata\\0823_data_room\\户一二楼_有家具和床_20240327164710\\merge_color_pcd_zhuangzheng_hand.pcd";*/
//	const std::string scenePath = "E:\\mydata\\0823_data_room\\merge_color_pcd(2)_jiahuiniubi_zhuangzheng.pcd";//_segoneroom
//
//	std::vector<cv::Point3f> scence_xyz;
//	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_xyz = Load3DPtCloudData(scenePath, scence_xyz);
//
//	//std::cout << "Load3DPtCloudData start" << std::endl;
//	//for (int i = 0; i < num_points; ++i) {
//	//	float x = points[3 * i + 0];
//	//	float y = points[3 * i + 1];
//	//	float z = points[3 * i + 2];
//	//	scence_xyz.push_back(cv::Point3f(x, y, z));
//	//}
//
//	std::vector<cv::Point3f> scene_unit_copy, cad_unit_copy;
//	scene_unit_copy.clear();
//	cad_unit_copy.clear();
//	bool is_meter;
//	if (!MeterToMinimeterConvertion(scence_xyz, scene_unit_copy))
//	{
//		is_meter = true;
//		//std::cout << "scene data is meter" << std::endl;
//	}
//
//
//	cv::Mat outputImage_xz;
//	//毛坯房设置  30.0f, 0.2
//	//装修房设置  100.0f, 0.2
//	float z_floor;
//	float z_ceil;
//	std::vector<std::vector<cv::Point3f>> voxels_xz = VoxelizePlanePointsByDensity_XZ(scene_unit_copy, vox_xz, z_ratio, outputImage_xz, false, z_floor, z_ceil);
//	std::cout << "z_floor: " << z_floor << "z_ceiling: " << z_ceil << std::endl;
//
//	std::vector<cv::Point3f> scence_xyz2;
//
//	for (const auto& voxel : voxels_xz) {
//		for (const auto& point : voxel) {
//
//			scence_xyz2.push_back(point);
//		}
//	}
//
//	// 生成俯视图
//	//std::vector<cv::Point3f> scene_xy;
//	//for (size_t i = 0; i < voxels_xz.size(); i++)
//	//{
//	//	scene_xy.insert(scene_xy.end(), voxels_xz[i].begin(), voxels_xz[i].end());
//	//}
//	//float minX1, maxX1, minY1, maxY1, minZ1, maxZ1;
//	//FindMinMaxCoordinates(scene_xy, minX1, maxX1, minY1, maxY1, minZ1, maxZ1);
//	//int voxel_num_in_x_filter1 = static_cast<int>((maxX1 - minX1) / gridSize)+10;
//	//int voxel_num_in_y_filter1 = static_cast<int>((maxY1 - minY1) / gridSize)+10;
//	//std::vector<std::vector<cv::Point3f>> voxels_xzy(voxel_num_in_x_filter1 * voxel_num_in_y_filter1);
//	//VoxelizePoints(scene_xy, minX1, minY1, gridSize, voxels_xzy, voxel_num_in_x_filter1);
//	//outputImage_xz = cv::Mat(voxel_num_in_y_filter1, voxel_num_in_x_filter1, CV_8UC3, cv::Scalar(0, 0, 0));
//	//FinalizeOutputImages(voxels_xzy, minX1, minY1, gridSize, outputImage_xz);
//	//cv::imwrite("00 outputImage_xz.png", outputImage_xz);
//
//
//	cv::Mat outputImage_xy;
//	cv::Mat outputImage_xy_nowindow;
//	cv::Mat outputImage_window;
//	//毛坯房设置 25.0f, gridSize, 0.7, 1000,
//	//装修房设置 100.0f,gridSize,0.4,1000
//	cv::Point base_point;
//	double minx, miny;
//	std::vector<std::vector<cv::Point3f>> voxels = VoxelizePlanePointsByDensity_02(scence_xyz2, door_half_size, vox_xy, gridSize, xy_ratio, z_height, outputImage_xy, outputImage_xy_nowindow, outputImage_window, z_floor, z_ceil, base_point, minx, miny);
//
//	// 翻转图像的Y轴
//	cv::Mat flippedImage;
//	cv::flip(outputImage_xy, flippedImage, 0);  // 0 表示垂直翻转
//	cv::imwrite("00 outputImage_xy.png", flippedImage);
//
//	cv::Mat flippedImage2;
//	cv::flip(outputImage_xy_nowindow, flippedImage2, 0);  // 0 表示垂直翻转
//	cv::imwrite("00 outputImage_xy_nowindow.png", flippedImage2);
//
//	cv::Mat flippedImage3;
//	cv::flip(outputImage_window, flippedImage3, 0);  // 0 表示垂直翻转
//	cv::imwrite("00 outputImage_xy_window.png", flippedImage3);
//
//	//毛坯房设置 30, 20
//	//装修房设置 30，30
//	/*if (model== 1)
//	{
//		outputImage_xy = removeIsolatedPixels(outputImage_xy, 30, 30,3);
//
//	}*/
//	//-----------------------------------------------3.户型分割----------------------------------------------------
//	//std::cout << "户型分割 start" << std::endl;
//	vector<RoomStruct>RoomMaps = region_seg_image_02(flippedImage, flippedImage2, thresholdValue, gridSize * 2, door_half_size, eroded_img_size);
//	int voxel_num_in_x = outputImage_xy.rows;
//	int voxel_num_in_y = outputImage_xy.cols;
//
//	/*for (size_t j = 0; j < RoomMaps.size(); j++)
//	{
//		auto room = RoomMaps[j];
//		for (size_t i = 0; i < room.roomPts.size(); i++)
//		{
//			cv::Point pt = room.roomPts[i];
//			int x_idx = pt.x;
//			int y_idx = pt.y;
//			int voxel_idx = y_idx * voxel_num_in_y + x_idx;
//			RoomMaps[j].roomPointClouds.insert(RoomMaps[j].roomPointClouds.end(), voxels[voxel_idx].begin(), voxels[voxel_idx].end());
//		}
//
//	}*/
//	//convertRoomsToMyData(RoomMaps, outputImage_xy);
//	// 创建并填充 MyData 结构体
//	for (size_t i = 0; i < RoomMaps.size(); i++)
//	{
//		MyData data;
//		Mat* contours = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
//		Mat* edges = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
//		RoomStruct room = RoomMaps[i];
//		//用room的成员值填充 contours
//		for (size_t j = 0; j < room.roomPts.size(); j++)
//		{
//			edges->at<Vec3b>(room.roomPts[j]) = Vec3b(255, 255, 255);
//		}
//		//用room的成员值填充 edges
//		for (size_t j = 0; j < room.roomContours.size(); j++)
//		{
//			contours->at<Vec3b>(room.roomContours[j]) = Vec3b(255, 255, 255);
//		}
//		data.minx = minx;
//		data.miny = miny;
//		data.contours = contours->data; // 替换为外部轮廓图
//		data.edges = edges->data; // 替换为外部轮廓图
//		data.contours_Row = contours->rows;
//		data.contours_Col = contours->cols;
//		data.edges_Row = edges->rows;
//		data.edges_Col = edges->cols;
//		int index = 0;
//		//vector<float> points_out(room.roomPointClouds.size()*3);
//		//float* points_out_ptr = points_out.data();
//		//for (cv::Point3f pt: room.roomPointClouds)
//		//{
//		//	points_out[3 *  index + 0]= pt.x;
//		//	points_out[3 * index + 1] = pt.y;
//		//	points_out[3 * index + 2] = pt.z;
//		//	index++;
//		//}
//		//data.points_out = points_out_ptr;
//		//data.points_out_num = room.roomPointClouds.size();
//		// 保存到全局变量
//		dataVec.push_back(data);
//	}
//	return true;
//
//	////-----------------------------------------------2.体素化点云，2D图生成----------------------------------------------------
//	////std::cout << "体素化点云 start" << std::endl;
//
//	//cv::Mat outputImage_xz;
//	////毛坯房设置  30.0f, 0.2
//	////装修房设置  100.0f, 0.2
//	//float z_floor;
//	//float z_ceil;
//	//std::vector<std::vector<cv::Point3f>> voxels_xz = VoxelizePlanePointsByDensity_XZ(scene_unit_copy, vox_xz, z_ratio, outputImage_xz, true, z_floor, z_ceil);
//	//std::vector<cv::Point3f> scene_xy;
//	//for (size_t i = 0; i < voxels_xz.size(); i++)
//	//{
//	//	scene_xy.insert(scene_xy.end(), voxels_xz[i].begin(), voxels_xz[i].end());
//	//}
//	//cv::Mat outputImage_xy;
//	////毛坯房设置 25.0f, gridSize, 0.7, 1000,
//	////装修房设置 100.0f,gridSize,0.4,1000
//	//cv::Point base_point;
//	//std::vector<std::vector<cv::Point3f>> voxels = VoxelizePlanePointsByDensity(scene_xy, door_half_size, vox_xy, gridSize, xy_ratio, z_height, outputImage_xy, z_floor, z_ceil, base_point);
//	//cv::Mat resizedImage_dilatedImage;
//	//cv::resize(outputImage_xy, resizedImage_dilatedImage, cv::Size(900, 900));
//	//cv::Mat resizedImage_outputImage_xz;
//	//cv::resize(outputImage_xz, resizedImage_outputImage_xz, cv::Size(900, 900));
//	//// 翻转图像的Y轴
//	//cv::Mat flippedImage;
//	//cv::flip(outputImage_xy, flippedImage, 0);  // 0 表示垂直翻转
//	//cv::imwrite("00 outputImage_xy.png", flippedImage);
//
//	////毛坯房设置 30, 20
//	////装修房设置 30，30
//	///*if (model== 1)
//	//{
//	//	outputImage_xy = removeIsolatedPixels(outputImage_xy, 30, 30,3);
//
//	//}*/
//	////-----------------------------------------------3.户型分割----------------------------------------------------
//	////std::cout << "户型分割 start" << std::endl;
//	//vector<RoomStruct>RoomMaps = region_seg_image(flippedImage, thresholdValue, gridSize * 2, door_half_size, eroded_img_size);
//	//int voxel_num_in_x = outputImage_xy.rows;
//	//int voxel_num_in_y = outputImage_xy.cols;
//
//	//for (size_t j = 0; j < RoomMaps.size(); j++)
//	//{
//	//	auto room = RoomMaps[j];
//	//	for (size_t i = 0; i < room.roomPts.size(); i++)
//	//	{
//	//		cv::Point pt = room.roomPts[i];
//	//		int x_idx = pt.x;
//	//		int y_idx = pt.y;
//	//		int voxel_idx = y_idx * voxel_num_in_y + x_idx;
//	//		RoomMaps[j].roomPointClouds.insert(RoomMaps[j].roomPointClouds.end(), voxels[voxel_idx].begin(), voxels[voxel_idx].end());
//	//	}
//	//}
//	////convertRoomsToMyData(RoomMaps, outputImage_xy);
//	//// 创建并填充 MyData 结构体
//	//for (size_t i = 0; i < RoomMaps.size(); i++)
//	//{
//	//	MyData data;
//	//	Mat* contours = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
//	//	Mat* edges = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
//	//	RoomStruct room = RoomMaps[i];
//	//	//用room的成员值填充 contours
//	//	for (size_t j = 0; j < room.roomPts.size(); j++)
//	//	{
//	//		edges->at<Vec3b>(room.roomPts[j]) = Vec3b(255, 255, 255);
//	//	}
//	//	//用room的成员值填充 edges
//	//	for (size_t j = 0; j < room.roomContours.size(); j++)
//	//	{
//	//		contours->at<Vec3b>(room.roomContours[j]) = Vec3b(255, 255, 255);
//	//	}
//	//	data.contours = contours->data; // 替换为外部轮廓图
//	//	data.edges = edges->data; // 替换为外部轮廓图
//	//	data.contours_Row = contours->rows;
//	//	data.contours_Col = contours->cols;
//	//	data.edges_Row = edges->rows;
//	//	data.edges_Col = edges->cols;
//	//	// 保存到全局变量
//	//	dataVec.push_back(data);
//	//}
//
//
//
//
//
//	//return true;
//}



bool RoomSegment(float* points, int num_points, int model = 0, int door_half_size = 75)
	{
		//-----------------------------------------------0.参数设置----------------------------------------------------
		// 测量时间
		if (door_half_size < 75)
		{
			door_half_size = 75;
		}
		int thresholdValue = 30;//二值化阈值
		int gridSize = 10;//网格大小
		int eroded_img_size = door_half_size * 0.6;//腐蚀大小
		float vox_xz = 30.0f;
		float z_ratio = 0.2f;
		float vox_xy = 25.0f;
		float xy_ratio = 0.7f;
		float z_height = 1000.0f;
		if (model == 1) {
			thresholdValue = 30;//二值化阈值
			gridSize = 10;//网格大小
			eroded_img_size = door_half_size * 0.6;//腐蚀大小
			vox_xz = 100.0f;
			z_ratio = 0.2f;
			vox_xy = 100.0f;
			xy_ratio = 0.4f;
			z_height = 600.0f;
		}
		//-----------------------------------------------1.读取点云----------------------------------------------------

		//E:\\mydata\\0823_data_room\\merge_color_pcd(2)_jiahuiniubi_zhuangzheng.pcd
		/*scenePath = "E:\\mydata\\0823_data_room\\户一二楼_有家具和床_20240327164710\\merge_color_pcd_zhuangzheng_hand.pcd";*/
		//const std::string scenePath = "E:\\mydata\\0823_data_room\\merge_color_pcd(2)_jiahuiniubi_zhuangzheng.pcd";
		std::vector<cv::Point3f> scence_xyz;

		//std::cout << "Load3DPtCloudData start" << std::endl;
		for (int i = 0; i < num_points; ++i) {
			float x = points[3 * i + 0];
			float y = points[3 * i + 1];
			float z = points[3 * i + 2];
			scence_xyz.push_back(cv::Point3f(x, y, z));
		}

		std::vector<cv::Point3f> scene_unit_copy, cad_unit_copy;
		scene_unit_copy.clear();
		cad_unit_copy.clear();
		bool is_meter;
		if (!MeterToMinimeterConvertion(scence_xyz, scene_unit_copy))
		{
			is_meter = true;
			//std::cout << "scene data is meter" << std::endl;
		}
		//-----------------------------------------------2.体素化点云，2D图生成----------------------------------------------------
		//std::cout << "体素化点云 start" << std::endl;

		cv::Mat outputImage_xz;
		//毛坯房设置  30.0f, 0.2
		//装修房设置  100.0f, 0.2
		float z_floor;
		float z_ceil;
		std::vector<std::vector<cv::Point3f>> voxels_xz = VoxelizePlanePointsByDensity_XZ(scene_unit_copy, vox_xz, z_ratio, outputImage_xz, false, z_floor, z_ceil);
		std::cout << "z_floor: " << z_floor << "z_ceiling: " << z_ceil << std::endl;

		std::vector<cv::Point3f> scence_xyz2;

		for (const auto& voxel : voxels_xz) {
			for (const auto& point : voxel) {

				scence_xyz2.push_back(point);
			}
		}

		// 生成俯视图
		//std::vector<cv::Point3f> scene_xy;
		//for (size_t i = 0; i < voxels_xz.size(); i++)
		//{
		//	scene_xy.insert(scene_xy.end(), voxels_xz[i].begin(), voxels_xz[i].end());
		//}
		//float minX1, maxX1, minY1, maxY1, minZ1, maxZ1;
		//FindMinMaxCoordinates(scene_xy, minX1, maxX1, minY1, maxY1, minZ1, maxZ1);
		//int voxel_num_in_x_filter1 = static_cast<int>((maxX1 - minX1) / gridSize)+10;
		//int voxel_num_in_y_filter1 = static_cast<int>((maxY1 - minY1) / gridSize)+10;
		//std::vector<std::vector<cv::Point3f>> voxels_xzy(voxel_num_in_x_filter1 * voxel_num_in_y_filter1);
		//VoxelizePoints(scene_xy, minX1, minY1, gridSize, voxels_xzy, voxel_num_in_x_filter1);
		//outputImage_xz = cv::Mat(voxel_num_in_y_filter1, voxel_num_in_x_filter1, CV_8UC3, cv::Scalar(0, 0, 0));
		//FinalizeOutputImages(voxels_xzy, minX1, minY1, gridSize, outputImage_xz);
		//cv::imwrite("00 outputImage_xz.png", outputImage_xz);


		cv::Mat outputImage_xy;
		cv::Mat outputImage_xy_nowindow;
		cv::Mat outputImage_window;
		//毛坯房设置 25.0f, gridSize, 0.7, 1000,
		//装修房设置 100.0f,gridSize,0.4,1000
		cv::Point base_point;
		double minx, miny;
		std::vector<std::vector<cv::Point3f>> voxels = VoxelizePlanePointsByDensity_02(scence_xyz2, door_half_size, vox_xy, gridSize, xy_ratio, z_height, outputImage_xy, outputImage_xy_nowindow, outputImage_window, z_floor, z_ceil, base_point, minx, miny);

		// 翻转图像的Y轴
		cv::Mat flippedImage;
		cv::flip(outputImage_xy, flippedImage, 0);  // 0 表示垂直翻转
		//cv::imwrite("00 outputImage_xy.png", flippedImage);

		cv::Mat flippedImage2;
		cv::flip(outputImage_xy_nowindow, flippedImage2, 0);  // 0 表示垂直翻转
		//cv::imwrite("00 outputImage_xy_nowindow.png", flippedImage2);

		cv::Mat flippedImage3;
		cv::flip(outputImage_window, flippedImage3, 0);  // 0 表示垂直翻转
		//cv::imwrite("00 outputImage_xy_window.png", flippedImage3);

		//毛坯房设置 30, 20
		//装修房设置 30，30
		/*if (model== 1)
		{
			outputImage_xy = removeIsolatedPixels(outputImage_xy, 30, 30,3);

		}*/
		//-----------------------------------------------3.户型分割----------------------------------------------------
		//std::cout << "户型分割 start" << std::endl;
		vector<RoomStruct>RoomMaps = region_seg_image_02(flippedImage, flippedImage2, thresholdValue, gridSize * 2, door_half_size, eroded_img_size);
		int voxel_num_in_x = outputImage_xy.rows;
		int voxel_num_in_y = outputImage_xy.cols;

		/*for (size_t j = 0; j < RoomMaps.size(); j++)
		{
			auto room = RoomMaps[j];
			for (size_t i = 0; i < room.roomPts.size(); i++)
			{
				cv::Point pt = room.roomPts[i];
				int x_idx = pt.x;
				int y_idx = pt.y;
				int voxel_idx = y_idx * voxel_num_in_y + x_idx;
				RoomMaps[j].roomPointClouds.insert(RoomMaps[j].roomPointClouds.end(), voxels[voxel_idx].begin(), voxels[voxel_idx].end());
			}

		}*/
		//convertRoomsToMyData(RoomMaps, outputImage_xy);
		// 创建并填充 MyData 结构体
		for (size_t i = 0; i < RoomMaps.size(); i++)
		{
			MyData data;
			Mat* contours = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
			Mat* edges = new Mat(outputImage_xy.size(), CV_8UC3, Scalar(0, 0, 0));
			RoomStruct room = RoomMaps[i];
			//用room的成员值填充 contours
			for (size_t j = 0; j < room.roomPts.size(); j++)
			{
				edges->at<Vec3b>(room.roomPts[j]) = Vec3b(255, 255, 255);
			}
			//用room的成员值填充 edges
			for (size_t j = 0; j < room.roomContours.size(); j++)
			{
				contours->at<Vec3b>(room.roomContours[j]) = Vec3b(255, 255, 255);
			}
			data.minx = minx;
			data.miny = miny;
			data.contours = contours->data; // 替换为外部轮廓图
			data.edges = edges->data; // 替换为外部轮廓图
			data.contours_Row = contours->rows;
			data.contours_Col = contours->cols;
			data.edges_Row = edges->rows;
			data.edges_Col = edges->cols;
			int index = 0;
			//vector<float> points_out(room.roomPointClouds.size()*3);
			//float* points_out_ptr = points_out.data();
			//for (cv::Point3f pt: room.roomPointClouds)
			//{
			//	points_out[3 *  index + 0]= pt.x;
			//	points_out[3 * index + 1] = pt.y;
			//	points_out[3 * index + 2] = pt.z;
			//	index++;
			//}
			//data.points_out = points_out_ptr;
			//data.points_out_num = room.roomPointClouds.size();
			// 保存到全局变量
			dataVec.push_back(data);
		}
		return true;
	}

	// 获取存储的数据
 MyData* get_data(int* count) {
		if (count == nullptr) {
			std::cerr << "Invalid count pointer" << std::endl;
			return nullptr;
		}

		*count = static_cast<int>(dataVec.size());  // 返回数据的数量
		if (*count == 0) {
			return nullptr;  // 如果没有数据，返回空指针
		}

		return dataVec.data();  // 返回数据指针
	}

	// 清空保存的数据
 void clear_data() {
		dataVec.clear();  // 清空数据
	}



//int main()
//{
//	std::cout << "z_floor: " << std::endl;
//	RoomSegment();
//	return 0;
//}