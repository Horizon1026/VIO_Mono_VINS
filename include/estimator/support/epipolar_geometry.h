#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class EpipolarGeometryClass {
private:
    // RANSAC 方法评价一个模型作用在一对匹配点上的误差阈值
    double RANSAC_THRESHOLD = 0.0005;
    // 使用 RANSAC 方法确定了一组 inliers 之后，从中随机选择多少点对来精细估计本质矩阵
    int USED_INLIERS_NUMS = 15;

public:
    /* 不带参数的构造函数 */
    EpipolarGeometryClass() {}

    /* 带参数的构造函数 */
    EpipolarGeometryClass(double threshold, int usedInliers);

    /* 析构函数 */
    ~EpipolarGeometryClass() {}

public:
    /* 根据两对匹配的特征点的坐标，估计出两帧之间的相对位姿 */
    bool EstimateRotationAndTranslation(std::vector<cv::Point2f> &pixelPoints0,   // 图像 0 中的特征点像素坐标
                                        std::vector<cv::Point2f> &pixelPoints1,   // 图像 1 中的特征点像素坐标
                                        Eigen::Matrix3d &CameraMatrix,            // 相机内参矩阵
                                        std::vector<uchar> &status,               // 图像 0 和图像 1 的特征点匹配情况
                                        Eigen::Matrix3d &R_c1c0,                  // 输出分解出来的相对姿态
                                        Eigen::Vector3d &t_c1c0);                 // 输出分解出来的相对位置
    bool EstimateRotationAndTranslation(std::vector<Eigen::Vector3d> &normPoints0,// 图像 0 中的归一化平面坐标
                                        std::vector<Eigen::Vector3d> &normPoints1,// 图像 1 中的归一化平面坐标
                                        std::vector<uchar> &status,               // 图像 0 和图像 1 的特征点匹配情况
                                        Eigen::Matrix3d &R_c1c0,                  // 输出分解出来的相对姿态
                                        Eigen::Vector3d &t_c1c0);                 // 输出分解出来的相对位置

    /* 根据两对匹配的特征点的坐标，判断出哪些是 inliers */
    bool FindInliersWithEssentialMatrix(std::vector<cv::Point2f> &pixelPoints0,   // 图像 0 中的特征点像素坐标
                                        std::vector<cv::Point2f> &pixelPoints1,   // 图像 1 中的特征点像素坐标
                                        Eigen::Matrix3d &CameraMatrix,            // 相机内参矩阵
                                        std::vector<uchar> &status);              // 图像 0 和图像 1 的特征点匹配情况
    bool FindInliersWithEssentialMatrix(std::vector<Eigen::Vector3d> &normPoints0,// 图像 0 中的归一化平面坐标
                                        std::vector<Eigen::Vector3d> &normPoints1,// 图像 1 中的归一化平面坐标
                                        std::vector<uchar> &status);              // 图像 0 和图像 1 的特征点匹配情况

public:
    /* 使用全部的匹配点，估计本质矩阵 */
    Eigen::Matrix3d FindEssentialMatrix(std::vector<cv::Point2f> &pixelPoints0,
                                        std::vector<cv::Point2f> &pixelPoints1,
                                        Eigen::Matrix3d &CameraMatrix,
                                        double &error);
    Eigen::Matrix3d FindEssentialMatrix(std::vector<Eigen::Vector3d> &normPoints0,
                                        std::vector<Eigen::Vector3d> &normPoints1,
                                        double &error);
    /* 从 E 矩阵中分解出 4 种位姿 */
    void DecomposeEssentialMatrix(Eigen::Matrix3d &E,       // 输入本质矩阵
                                  Eigen::Matrix3d &R0,      // 输出可能的两种相对位姿 R_c1_c0
                                  Eigen::Matrix3d &R1,
                                  Eigen::Vector3d &t0,      // 输出可能的两种相对位置 t_c1_c0
                                  Eigen::Vector3d &t1);
    /* 基于两幅图像的位姿，对一对匹配的特征点进行三角测量，得到其3D坐标。输入的点坐标是矫正后归一化平面的坐标 */
    Eigen::Vector3d TriangulateOnePoint(Eigen::Matrix3d &R0, Eigen::Vector3d &t0,           // 前一帧图像的位姿，R0_cw，t0_cw
                                        Eigen::Matrix3d &R1, Eigen::Vector3d &t1,           // 后一帧图像的位姿，R1_cw，t1_cw
                                        Eigen::Vector3d &point0, Eigen::Vector3d &point1);  // 两帧图像中对应点的归一化平面坐标（矫正后）

    /* 从本质矩阵中恢复出帧间位姿 R_c1c0 和 t_c1c0，并返回不符合这一相对位姿的特征点对的个数 */
    int RecoverPoseFromEssentialMatrix(Eigen::Matrix3d &E,                     // 输入本质矩阵
                                       std::vector<cv::Point2f> &pixelPoints0, // 输入图像 0 的特征点像素坐标
                                       std::vector<cv::Point2f> &pixelPoints1, // 输入图像 1 的特征点像素坐标
                                       Eigen::Matrix3d &CameraMatrix,          // 输入相机内参矩阵
                                       Eigen::Matrix3d &R_c1c0,                // 输出分解出来的相对姿态
                                       Eigen::Vector3d &t_c1c0);               // 输出分解出来的相对位置
    int RecoverPoseFromEssentialMatrix(Eigen::Matrix3d &E,                          // 输入本质矩阵
                                       std::vector<Eigen::Vector3d> &normPoints0,   // 输入图像 0 的归一化平面坐标
                                       std::vector<Eigen::Vector3d> &normPoints1,   // 输入图像 1 的归一化平面坐标
                                       Eigen::Matrix3d &R_c1c0,                     // 输出分解出来的相对姿态
                                       Eigen::Vector3d &t_c1c0);                    // 输出分解出来的相对位置

private:
    /* 用五点法估计所有可能的本质矩阵 */
    std::vector<Eigen::Matrix3d> FindAllEssentialMatriceFivePoints(std::vector<cv::Point2f> &pixelPoints0,
                                                                   std::vector<cv::Point2f> &pixelPoints1,
                                                                   Eigen::Matrix3d &CameraMatrix,
                                                                   std::vector<double> &meanErrors);
    std::vector<Eigen::Matrix3d> FindAllEssentialMatriceFivePoints(std::vector<Eigen::Vector3d> &normPoints0,
                                                                   std::vector<Eigen::Vector3d> &normPoints1,
                                                                   std::vector<double> &meanErrors);
    /* 五点法计算系数多项式 */
    void GetCoeffMatrixFivePoints(double *e, double *A);
    /* 计算对极约束误差 */
    std::vector<double> ComputeEpipolarGeometryError(std::vector<Eigen::Vector3d> &normPoints0,
                                                     std::vector<Eigen::Vector3d> &normPoints1,
                                                     Eigen::Matrix3d &E);
    /* 像素坐标变换为归一化坐标 */
    std::vector<Eigen::Vector3d> GetNormPointsFromPixelPoints(std::vector<cv::Point2f> &pixelPoints,
                                                              Eigen::Matrix3d &CameraMatrix);

};