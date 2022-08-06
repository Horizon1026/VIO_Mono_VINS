#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

/* 定义相机模型的基类 */
class CameraBase {
/* 构造函数与析构函数 */
public:
    CameraBase() {}
    virtual ~CameraBase() {}

/* 相关参数 */
public:
    // 图像尺寸
    int rows, cols;
    // 图像畸变参数
    float k1, k2, p1, p2;
    // 图像屏蔽域，被屏蔽的地方不参与特征点检测与追踪
    cv::Mat mask;
private:
    // 反畸变映射表，以及其粒度
    int mapScale;
    std::vector<std::vector<std::pair<float, float>>> distortedToCorrectMap;

/* 相机模型基础参数初始化 */
public:
    void InitBaseParams(int rows, int cols, float k1, float k2, float p1, float p2);
    bool SetMask(const cv::Mat &mask);

/* 像素平面坐标与归一化平面坐标转换 */
public:
    /* 由像素平面坐标，变化为归一化平面坐标 */
    virtual Eigen::Vector2f PixelPlane_NormlizedPlane(cv::Point2f p) = 0;
    /* 由归一化平面坐标，变化为像素平面坐标 */
    virtual cv::Point2f NormlizedPlane_PixelPlane(Eigen::Vector2f p) = 0;

/* 畸变与反畸变转换 */
public:
    /* 根据畸变参数和图像尺寸，构造反畸变映射表。粒度 scale 越大，反畸变映射越精确 */
    bool CreateUndistortedMap(int scale);
    /* 由非畸变像素平面坐标，变化为畸变像素平面坐标 */
    virtual cv::Point2f Undistortion_Distortion(cv::Point2f p) = 0;
    /* 由畸变像素平面坐标，变化为非畸变像素平面坐标 */
    cv::Point2f Distortion_Undistortion(cv::Point2f p);
};