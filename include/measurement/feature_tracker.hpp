#pragma once
#include <memory>
#include <Eigen/Core>

#include <include/camera_model/pinhole_camera.hpp>

/* 定义特征点追踪管理器 */
class FeatureTracker {
/* 构造函数与析构函数 */
public:
    FeatureTracker() {}
    ~FeatureTracker() {}

/* 相机模型 */
private:
    std::shared_ptr<PinholeCamera> camera;
public:
    bool SetCameraModel(std::shared_ptr<PinholeCamera> &camera);

/* 可配置参数 */
public:
    // 如果由一次追踪后，有效特征点数量不足于此，则需要补充检测
    int minFeatureNum = 120;
    // 追踪到的特征点之间允许最小距离
    float minDistance = 25;
public:
    bool SetParams(int minFeatureNum, float minDistance);

/* 内部参数 */
private:
    // 关键点与图像帧的全局计数器，用于编号
    static size_t maxFeatureID;
    static size_t maxImageID;
    // 上一帧与最新帧输入的图像、图像时间戳、检测或追踪到的特征点
    struct ImageWithFeatures {
        cv::Mat image;
        double timeStamp;
        std::vector<cv::Point2f> rawPoints;
        std::vector<cv::Point2f> undistortedPoints;
    } pre, next;
    // 特征点被追踪到的次数
    std::vector<size_t> trackedTimes;
    // 特征点在最新一帧与上一帧之间的非畸变光流像素速度
    std::vector<cv::Point2f> pixelVelocity;

/* 对外输出参数 */
public:
    // 特征点的 ID 号，及其在最新一帧中的像素坐标与归一化平面坐标
    std::vector<size_t> tracked_IDs;
    std::vector<Eigen::Vector2f> tracked_pixel;
    std::vector<Eigen::Vector2f> tracked_norm;
    // 最新一帧图像的 ID 号
    size_t currentImageID;

/* 对外接口方法 */
public:
    /* 输入一帧新的图像，追踪特征点 */
    bool Track(const cv::Mat &image, double timeStamp);
    /* 回退图像 ID 的编号 */
    static void AdjustImageIDCounter(size_t offset = 0);
    /* 重置图像 ID 的编号 */
    static void ResetImageIDCounter(size_t id = 1);
    /* 绘制出原始图像的特征点追踪结果 */
    void DrawTrackingResult(int delay);
    /* 绘制出畸变校正后图像的特征点追踪结果 */
    void DrawUndistortedTrackingResult(void);

/* 对内接口方法 */
private:
    /* 特征点追踪器预分配空间 */
    void Reserve(size_t size);
    /* 捆绑追踪次数、像素坐标和特征点 ID，基于追踪次数进行排序 */
    std::vector<std::pair<size_t, std::pair<cv::Point2f, size_t>>> SortFeatures(void);
    /* 基于特征点被追踪到的次数，检测扎堆的特征点 */
    void DetectDenseFeatures(std::vector<uchar> &status, int radius);
    /* 基于特征点被追踪到的次数，构造屏蔽域 */
    cv::Mat CreateMask(int radius);
    /* 基于待删除索引 status，调整 vector 容器 */
    template<typename T>
    void AdjustVector(std::vector<T> &v, const std::vector<uchar> &status);
    /* 基于待删除索引 status，一次性调整所有需要调整的 vector 容器 */
    void AdjustVectors(const std::vector<uchar> &status);
    /* 添加新检测到的特征点 */
    void AddNewFeatures(const std::vector<cv::Point2f> &newFeatures);
    /* 更新特征点的 ID 号 */
    void UpdateFeaturesID(void);
    /* 基于去畸变像素坐标，计算光流速度 */
    void ComputePixelVelocity(void);
    /* 更新追踪结果 */
    void UpdateTrackingResult(void);
};