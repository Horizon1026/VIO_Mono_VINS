#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class PerspectiveNPointClass {
private:
    // PnP 迭代优化过程中，位姿李代数调整量的模长达到此阈值，则认为收敛
    double CONVERGE_THRESHOLD = 0.01;
    // RANSAC 方法中，归一化平面上的重投影误差的阈值
    double RANSAC_THRESHOLD = 0.1;
    // 
    int USED_INLIERS_NUMS = 20;

public:
    /* 采用 RANSAC 方法代入 PnP 模型来估计相机位姿 */
    void EstimateRotationAndTranslation(std::vector<cv::Point3f> &wordPoints,   // 对应世界坐标系中的 3D 点坐标
                                        std::vector<cv::Point2f> &pixelPoints,  // 像素平面坐标
                                        Eigen::Matrix3d &CameraMatrix,          // 相机内参矩阵
                                        Eigen::Matrix3d &R_cw,          // 求解输出的相机姿态
                                        Eigen::Vector3d &t_cw,          // 求解输出的相机位置
                                        std::vector<uchar> &status);    // 判断 3D 和 2D 点对是否为 inliers 或者 outliers
    void EstimateRotationAndTranslation(std::vector<Eigen::Vector3d> &wordPoints,  // 对应世界坐标系中的 3D 点坐标
                                        std::vector<Eigen::Vector3d> &normPoints,  // 归一化平面坐标
                                        Eigen::Matrix3d &R_cw,          // 求解输出的相机姿态
                                        Eigen::Vector3d &t_cw,          // 求解输出的相机位置
                                        std::vector<uchar> &status);    // 判断 3D 和 2D 点对是否为 inliers 或者 outliers

    /* 采用 RANSAC 方法代入 PnP 模型来判断哪些点对是 inliers */

public:
    /* 利用所有的 3D 点和 2D 点对，估计相机位姿 */
    void SolvePnP(std::vector<cv::Point3f> &wordPoints,     // 对应世界坐标系中的 3D 点坐标
                  std::vector<cv::Point2f> &pixelPoints,    // 像素平面坐标
                  Eigen::Matrix3d &CameraMatrix,            // 相机内参
                  Eigen::Matrix3d &R_cw,    // 求解输出的相机姿态
                  Eigen::Vector3d &t_cw);   // 求解输出的相机位置
    void SolvePnP(std::vector<Eigen::Vector3d> &wordPoints,     // 对应世界坐标系中的 3D 点坐标
                  std::vector<Eigen::Vector3d> &normPoints,     // 归一化平面坐标
                  Eigen::Matrix3d &R_cw,    // 求解输出的相机姿态
                  Eigen::Vector3d &t_cw);   // 求解输出的相机位置

private:
    /* 像素坐标变换为归一化坐标 */
    std::vector<Eigen::Vector3d> GetNormPointsFromPixelPoints(std::vector<cv::Point2f> &pixelPoints,
                                                              Eigen::Matrix3d &CameraMatrix);
    /* 判断 3 个 2D 点是否共线 */
    bool IsCollineation(std::vector<Eigen::Vector3d> &points2D);

    /* 指定位姿的情况下，为每一对点对计算重投影误差 */
    std::vector<double> ComputeReprojectionError(const std::vector<Eigen::Vector3d> &points3D,
                                                 const std::vector<Eigen::Vector3d> &points2D,
                                                 const Eigen::Matrix3d &R,
                                                 const Eigen::Vector3d &t);
};