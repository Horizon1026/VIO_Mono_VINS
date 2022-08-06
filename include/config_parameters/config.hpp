#pragma once

#include <opencv2/opencv.hpp>

/* 定义读取、存储和备份的系统配置参数管理器 */
class Config {
/* 构造函数与析构函数 */
public:
    Config() {}
    ~Config() {}

/* 存储配置信息 */
public:
    // 存取文件路径信息
    struct FilePathes {
        std::string savePoses;
        std::string globalMask;
        std::string saveCostTimes;
    } pathes;

    // 选择求解器的类型
    int solverType;

    // 相机内参信息
    struct CameraIntrinsic {
        int rows;
        int cols;
        float fx;
        float fy;
        float cx;
        float cy;
        float k1;
        float k2;
        float p1;
        float p2;
        int mapScale;
    } cameraIntrinsic;
    
    // IMU 噪声信息
    struct IMUNoise {
        float gyro_n;
        float accel_n;
        float gyro_w;
        float accel_w;
    } imuNoise;

    // 相机外参信息
    struct CameraExtrinsic {
        cv::Mat R_bc;
        cv::Mat t_bc;
    } cameraExtrinsic;

    // VIO-Mono 系统宏观配置参数
    int windowSize;

    // VIO-Mono 前端配置参数
    struct TrackerParams {
        int featureNum;
        float minDistance;
    } trackerParams;

    // VIO-Mono 初始化配置参数
    struct InitParams {
        float minIMUMotion;
        float minMeanParallax;
        int minCovisibleNum;
    } initParams;

    // VIO-Mono 后端优化配置参数
    struct OptimizeParams {
        float minMeanParallax;
        int minCovisibleNum;
        float maxImuSumTime;
    } optimizeParams;

public:
    /* 读取 config 配置文件 */
    bool LoadConfigFile(std::string &filePath);

    /* 将配置结果写入新的 config 文件 */
    bool CreateConfigFile(std::string &filePath);
};