#ifndef POINTCLOUDPROCESSING_H
#define POINTCLOUDPROCESSING_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Define visibility for shared library
#ifdef _WIN32
#ifdef POINTCLOUDPROCESSING_EXPORTS
#define POINTCLOUDPROCESSING_API __declspec(dllexport)
#else
#define POINTCLOUDPROCESSING_API __declspec(dllimport)
#endif
#else
#define POINTCLOUDPROCESSING_API __attribute__((visibility("default"))) // For Linux
#endif

using namespace cv;
using namespace std;

// Structure to hold data for processing
struct MyData {
    uchar* contours;
    uchar* edges;
    int contours_Row;
    int contours_Col;
    int edges_Row;
    int edges_Col;
    double minx;
    double miny;
};

std::vector<MyData> dataVec;

struct RoomStruct {
    int roomId = 0;
    vector<Point> roomPts;
    vector<Point> roomContours;
    vector<Point3f> roomPointClouds;
    Vec3b roomColor;
};

struct Rooms {
    Mat input_image;
    vector<RoomStruct> rooms;
};

// Function declarations
std::vector<std::vector<cv::Point3f>> VoxelizePlanePoints(const std::vector<cv::Point3f>& plane_points, float voxelWidthMm, int threshold, cv::Mat& outputImage);

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
);

std::vector<std::vector<cv::Point3f>> VoxelizePlanePointsByDensity_XZ(
    const std::vector<cv::Point3f>& plane_points,
    float voxelWidthMm,
    float radio,
    cv::Mat& outputImage,
    bool reverse, float& z_floor,
    float& z_ceil
);

std::pair<std::vector<Vec3b>, std::vector<Point>> checkFourDirections(const Mat& image, Point center);
double distanceBetweenPoints(const Point& p1, const Point& p2);
Vec3b processDirections(const std::vector<Vec3b>& directions, const std::vector<Point>& locations, Point center, int distanceThreshold, int distance_min);
Vec3b processDirections_only(const std::vector<Vec3b>& directions, const std::vector<Point>& locations, Point center, int distanceThreshold, int distance_min);

bool CheckInputDataUnit(const std::vector<cv::Point3f>& data);
bool MeterToMinimeterConvertion(const std::vector<cv::Point3f>& data, std::vector<cv::Point3f>& out);
bool isNonIsolatedPixel(const cv::Mat& input_image, int row, int col, int n, int num, int image_type);
cv::Mat removeIsolatedPixels(const cv::Mat& input_image, int n, int num, int image_type);

vector<RoomStruct> region_seg_image(Mat src, Mat src_nowindow, int thresholdValue, int gridSize, int door_half_size, int eroded_img_size);
vector<RoomStruct> updateWhitePixelsWithNearestColor(Mat binaryImage, Mat colorImage, Mat& outputImage, int dis);

extern "C" {
    POINTCLOUDPROCESSING_API bool RoomSegment(float* points, int num_points, int model, int door_half_size);
    POINTCLOUDPROCESSING_API MyData* get_data(int* count);
    POINTCLOUDPROCESSING_API cv::Mat* createMatrix();
    POINTCLOUDPROCESSING_API void clear_data();
}

#endif  // POINTCLOUDPROCESSING_H
