#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

/* 定义输入图像信息的格式 */
class ImageMessage {
public:
    // 灰度图像信息
    cv::Mat image;
    // 时间戳，单位为秒
    double timeStamp;
public:
    ImageMessage() {}
    ~ImageMessage() {}
};

/* 定义输入 IMU 信息的格式 */
class IMUMessage {
public:
    // 加速度量测信息
    Eigen::Vector3f accel;
    // 角速度量测信息
    Eigen::Vector3f gyro;
    // 时间戳，单位为秒
    double timeStamp;
public:
    IMUMessage() {}
    ~IMUMessage() {}
};