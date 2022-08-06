#pragma once

#include <include/camera_model/camera_base.hpp>

/* 定义针孔相机模型 */
class PinholeCamera : public CameraBase {
/* 构造函数与析构函数 */
public:
    PinholeCamera() : CameraBase() {}
    ~PinholeCamera() {}

/* 此相机模型的参数 */
public:
    float fx, fy;
    float cx, cy;

/* 相机投影模型参数初始化 */
public:
    void InitParams(float fx, float fy, float cx, float cy, int mapScale);

/* 重写基类方法 */
public:
    /* 由像素平面坐标，变化为归一化平面坐标 */
    virtual Eigen::Vector2f PixelPlane_NormlizedPlane(cv::Point2f p) override;
    /* 由归一化平面坐标，变化为像素平面坐标 */
    virtual cv::Point2f NormlizedPlane_PixelPlane(Eigen::Vector2f p) override;
    /* 由非畸变像素平面坐标，变化为畸变像素平面坐标 */
    virtual cv::Point2f Undistortion_Distortion(cv::Point2f p) override;
};