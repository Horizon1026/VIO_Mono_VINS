#pragma once

#include <unordered_map>
#include <memory>
#include <Eigen/Core>

/* 定义某一个特征点在某一帧图像中的观测 */
class Observe {
/* 构造与析构函数 */
public:
    Observe(const Eigen::Vector2f &setNorm,
            const Eigen::Vector2f &setPixel,
            float setDt) :
            norm(setNorm),
            pixel(setPixel),
            dt(setDt) {}
    ~Observe() {}
/* 存储参数 */
public:
    // 归一化平面坐标
    Eigen::Vector2f norm;
    // 畸变的像素坐标
    Eigen::Vector2f pixel;
    // IMU 与相机的相对时间差
    float dt;
};


/* 定义一个特征点 */
class Landmark {
/* 构造函数与析构函数 */
public:
    Landmark(const size_t ID_,
             const size_t firstFrameID_,
             const std::vector<std::shared_ptr<Observe>> &observes_);
    ~Landmark() {}

/* 此特征点存储的信息 */
public:
    // 此特征点的全局索引
    const size_t ID;
    // 首次观测到此特征点的关键帧的全局索引
    size_t firstFrameID;
    // 此特征点在以第 firstFrameID 帧为起始的连续几帧中的观测
    std::vector<std::shared_ptr<Observe>> observes;
    // 此特征点在第 firstFrameID 帧中的逆深度
    float invDep;
    // 此特征点在世界坐标系中的位置
    Eigen::Vector3f p_w;
    // 此特征点的三角化状态
    enum SolveStatus {
        NO = 0,
        YES,
        ERROR
    } isSolved;

/* 某一个特征点对外提供的接口方法 */
public:
    /* 为此特征点添加一个观测 */
    void AddNewObserve(const std::shared_ptr<Observe> &newObserve);
    /* 获取指定 FrameID 的归一化平面坐标观测 */
    Eigen::Vector2f GetNorm(const size_t frameID);
    /* 获取指定 FrameID 的畸变像素坐标 */
    Eigen::Vector2f GetPixel(const size_t frameID);
    /* 获取最后一个观测到此特征点的图像帧的 ID */
    size_t FinalFrameID(void);
};


/* 定义特征点管理器，存储所有滑动窗口内的特征点，并进行管理 */
class LandmarkManager {
/* 构造函数与析构函数 */
public:
    LandmarkManager() {}
    ~LandmarkManager() {}

/* 特征点管理器存储的信息 */
public:
    std::unordered_map<size_t, std::shared_ptr<Landmark>> landmarks;

/* 特征点管理器提供的接口方法 */
public:
    /* 添加特征点信息 */
    bool AddNewLandmarks(const std::vector<size_t> &IDs,
                         const std::vector<Eigen::Vector2f> &norms,
                         const std::vector<Eigen::Vector2f> &pixels,
                         const size_t frameID,
                         const float dt);
    /* 移除指定关键帧所观测到的特征点。但此关键帧之后的观测帧 ID 会依次向前偏移 */
    void RemoveByFrameID(const size_t frameID, bool offset);
    /* 移除指定 ID 的特征点 */
    void RemoveByID(const size_t landmarkID);
    /* 将所有特征点的指针都存放到一个 vector 容器中 */
    std::vector<std::shared_ptr<Landmark>> GetAllLandmarks(void);
    /* 打印出保存的所有特征点的所有信息 */
    void PrintAllLandmarks(void);
};