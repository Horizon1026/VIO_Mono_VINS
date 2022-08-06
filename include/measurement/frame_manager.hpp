#pragma once
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <include/measurement/landmark_manager.hpp>
#include <include/measurement/imu_preintegration.hpp>

/* 定义图像帧类，管理一帧图像的基本信息 */
class Frame {
/* 构造函数与析构函数 */
public:
    Frame(const size_t &setID);
    ~Frame() {}

/* 对外公开成员变量 */
public:
    // 此帧观测到的所有特征点 <特征点 ID，特征点指针>
    // 若要搜索两帧的共视特征点，此项可缩小搜索范围
    std::unordered_map<size_t, std::shared_ptr<Landmark>> landmarks;
    // 此帧对应的相机位姿和速度
    Eigen::Quaternionf q_wc;
    Eigen::Vector3f t_wc;
    Eigen::Vector3f v_wc;
    // 此帧的全局索引
    size_t ID;
    // 此帧保存的图像信息
    cv::Mat image;
    // 此帧的时间戳
    double timeStamp;

/* 对外公开成员函数 */
public:
    // 添加一个特征点
    bool AddLandmark(std::shared_ptr<Landmark> &newLandmark);
    // 找到指定 ID 的一帧与此帧的共视特征点
    std::vector<std::shared_ptr<Landmark>> GetCovisibleLandmarks(const std::shared_ptr<Frame> &target);
    // 计算指定 ID 的一帧与此帧的平均视差
    float GetAverageParallax(const std::shared_ptr<Frame> &target);
    // 确定给定 ID 的目标帧与此帧的关联性，返回平均视差和共视特征点数量
    std::pair<float, size_t> GetCorrespondence(const std::shared_ptr<Frame> &target);
};


/* 定义与预积分块捆绑的图像帧类，本质上就是对 Frame 类的扩充 */
class CombinedFrame : public Frame {
/* 构造函数与析构函数 */
public:
    CombinedFrame(const size_t &setID, const std::shared_ptr<IMUPreintegration> &imu);
    ~CombinedFrame() {}

/* 扩充的对外公开的成员变量 */
public:
    // 指向一个 IMU 预积分块
    std::shared_ptr<IMUPreintegration> imu;
    // 此帧对应的 IMU 坐标系的位姿和速度
    Eigen::Quaternionf q_wb;
    Eigen::Vector3f t_wb;
    Eigen::Vector3f v_wb;
};


/* 定义帧管理器，管理滑动窗口内所有帧 */
class FrameManager {
/* 构造函数与析构函数 */
public:
    FrameManager(const size_t &windowSize);
    ~FrameManager() {}

/* 对外公开成员变量 */
public:
    // 滑动窗口管理所有帧
    std::list<std::shared_ptr<CombinedFrame>> frames;
    // 指定滑动窗口大小
    const size_t windowSize;

/* 对外公开成员函数 */
public:
    /* 增加新一帧图像帧，添加前检查 ID 是否合法 */
    bool AddNewFrame(const std::shared_ptr<CombinedFrame> &newFrame);
    /* 移除一帧图像帧，给定图像帧在滑动窗口内的索引（ 从 0 到 windowSize-1 ） */
    bool RemoveFrame(const size_t offset);
    /* 提取某一帧图像帧，提取前检查 ID 是否合法 */
    std::shared_ptr<CombinedFrame> GetFrame(const size_t id);
    /* 打印出所有图像帧的信息 */
    void PrintAllFrames(void);
    /* 绘制出指定两帧的特征点共视关系 */
    void DrawCovisibleLandmarks(const size_t offset1, const size_t offset2);
    void DrawCovisibleLandmarks(const std::shared_ptr<CombinedFrame> &frame1, const std::shared_ptr<CombinedFrame> &frame2);
};


/* 定义位姿类，用于保存被边缘化的旧位姿形成的轨迹 */
class Pose {
/* 构造函数与析构函数 */
public:
    Pose(const double timeStamp,
        const Eigen::Quaternionf &q_wb,
        const Eigen::Vector3f &t_wb,
        const Eigen::Vector3f &v_wb) : timeStamp(timeStamp), q_wb(q_wb), t_wb(t_wb), v_wb(v_wb) {}
    ~Pose() {}

/* 对外公开成员变量 */
public:
    double timeStamp;
    Eigen::Quaternionf q_wb;
    Eigen::Vector3f t_wb;
    Eigen::Vector3f v_wb;
};