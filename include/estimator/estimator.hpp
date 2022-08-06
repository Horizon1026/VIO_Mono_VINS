#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <ctime>
#include <pangolin/pangolin.h>

#include <include/interface/message_format.hpp>
#include <include/config_parameters/config.hpp>
#include <include/measurement/feature_tracker.hpp>
#include <include/measurement/landmark_manager.hpp>
#include <include/measurement/imu_preintegration.hpp>
#include <include/measurement/frame_manager.hpp>

/* VIO 主体类定义 */
class VIOMono {
/* 构造函数与析构函数 */
public:
    VIOMono() {}
    ~VIOMono() {}


/* interface : 对外接口部分 */
using CombinedMessage = std::pair<std::vector<std::shared_ptr<IMUMessage>>, std::shared_ptr<ImageMessage>>;
private:
    // 输入数据缓冲区
    std::list<std::shared_ptr<ImageMessage>> imageMessages;
    std::list<std::shared_ptr<IMUMessage>> imuMessages;
    // 初始时间戳，用于整体平移系统时间
    double originTimeStamp = -1;
    // 图像帧时间戳补偿
    double current_imageTimeOffset = 0.0;
    // 历史位姿形成的轨迹
    std::list<std::shared_ptr<Pose>> poses;
    // 历史单步优化耗时
    std::list<double> costTimes;
    // 相机与 IMU 之间的相对位姿
    Eigen::Quaternionf q_bc;
    Eigen::Vector3f t_bc;
    // 参数配置器
    std::shared_ptr<Config> config;
public:
    // 最新时刻的时间戳
    double current_timeStamp = 0.0;
    // 最新时刻的 IMU 位姿和速度
    Eigen::Quaternionf current_q_wb;
    Eigen::Vector3f current_t_wb;
    Eigen::Vector3f current_v_wb;
    // 已优化求解的最新一帧相机位姿和速度
    Eigen::Quaternionf current_q_wc;
    Eigen::Vector3f current_t_wc;
    Eigen::Vector3f current_v_wc;
    // 最新时刻的 IMU 偏差值
    Eigen::Vector3f current_bias_a;
    Eigen::Vector3f current_bias_g;

public:
    /* 输入一帧图像数据 */
    bool PushImageMessage(std::shared_ptr<ImageMessage> newImageMessage);
    /* 输入一帧 IMU 数据 */
    bool PushIMUMessage(std::shared_ptr<IMUMessage> newIMUMessage);
    /* VIO 初始化，要求输入配置参数文件的路径 */
    bool Initialize(std::string &configFilePath);
    /* 单步运行 */
    bool RunOnce(void);
    /* 开始运行，并设定超时时间 */
    bool Run(size_t timeOut);
    /* 将历史位姿写入到 this->config->pathes.savePoses 指定文件中 */
    bool SavePosesAsFile();
    /* 将历史优化耗时写入到 this->config->pathes.saveCostTimes 指定文件中 */
    bool SaveCostTimesAsFile();


/* estimator : 内部通用部分 */
private:
    /* 生效参数配置器 config 中的参数 */
    bool UpdateConfigParams(void);
    /* 输出一帧图像和 IMU 捆绑的数据。如果当前缓存区中并不存在有效量测，则返回 false */
    bool GetCombinedMessage(CombinedMessage &message);
    /* 打印出一帧图像和 IMU 捆绑的数据 */
    void PrintCombinedMesage(CombinedMessage &message);

/* feature_tracker : 特征点追踪器部分 */
private:
    std::shared_ptr<FeatureTracker> featureTracker;
private:
    /* 为特征点追踪器配置参数 */
    bool UpdateConfigParams_FeatureTracker(void);


/* measurements : 图像数据与 IMU 数据管理部分 */
private:
    std::shared_ptr<LandmarkManager> landmarkManager;
    std::shared_ptr<FrameManager> frameManager;
private:
    /* 初始化特征点管理器 */
    bool InitializeLandmarkManager(void);
    /* 初始化帧管理器，为其配置参数 */
    bool UpdateConfigParams_FrameManager(void);
    /* 为 IMU 噪声模型配置参数 */
    bool UpdateConfigParams_IMU(void);
    /* 处理一组捆绑数据 */
    bool ProcessCombinedMessage(const CombinedMessage &message);
    /* 移除最旧观测以及相关信息 */
    bool RemoveOldestFrame(void);
    /* 移除次新观测以及相关信息 */
    bool RemoveSubnewFrame(void);
    /* 移除被标记为 Error 的特征点，同步移除在 frame 中的索引 */
    bool RemoveErrorLandmarks(void);
    /* 检查滑动窗口内的信息是否合法 */
    bool CheckSlidingWindowInfo(void);
    /* 打印出最新状态数据 */
    void PrintCurrentParams(void);
    /* 更新最新时刻状态值 */
    void UpdateCurrentStatus(void);
    /* 将最旧帧添加到历史轨迹中 */
    void SaveOldestFrameToPoses(void);


/* estimator_support : 后端求解器相关子算法模块 */
private:
    // 重投影信息矩阵
    Eigen::Matrix2f projectionInformation;
    // 后端求解器类型
    enum SolverType {
        GraphOptimizor = 0,
        RootVIO
    } solverType;
private:
    /* 给定位姿与观测，三角测量一个特征点，返回测量成功与否 */
    bool TriangulateLandmark(const std::shared_ptr<Landmark> &landmark,
                             const Eigen::Quaternionf &q_cw_1,
                             const Eigen::Vector3f &t_cw_1,
                             const Eigen::Vector2f &norm_1,
                             const Eigen::Quaternionf &q_cw_2,
                             const Eigen::Vector3f &t_cw_2,
                             const Eigen::Vector2f &norm_2);
    /* 指定一个特征点，使用观测到他的所有关键帧共同估计他的位置 */
    bool TriangulateLandmark(const std::shared_ptr<Landmark> &landmark);
    /* 指定两个图像帧，基于他们的已知相机位姿，三角测量他们共视的特征点 */
    bool TriangulateLandmarks(const std::shared_ptr<CombinedFrame> &frame1,
                              const std::shared_ptr<CombinedFrame> &frame2);
    /* 指定一个特征点，使用他的世界坐标系位置，更新他的逆深度 */
    bool UsePosToUpdateInvdep(const std::shared_ptr<Landmark> &landmark);
    /* 指定一个特征点，使用他的逆深度，更新他的世界坐标系位置 */
    bool UseInvdepToUpdatePos(const std::shared_ptr<Landmark> &landmark);
    /* 指定两个图像帧，基于他们的共视特征点，估计出两者相对位姿 */
    bool EstimateRelativePose(const std::shared_ptr<CombinedFrame> &frame1,
                              const std::shared_ptr<CombinedFrame> &frame2,
                              Eigen::Quaternionf &q_c1c2,
                              Eigen::Vector3f &t_c1c2);
    /* 指定一个图像帧，基于他的已三角化特征点的世界坐标系位置，估计此帧位姿 */
    bool EstimatePose(const std::shared_ptr<CombinedFrame> &frame);


/* estimator_visualize : 数据可视化相关 */
public:
    /* 绘制所有特征点和所有关键帧之间的空间位置关系 */
    void Visualize(void);
    /* 构建绘图窗口 */
    void VisualizeStart(pangolin::OpenGlRenderState &s_cam, pangolin::View &d_cam);
    /* 读取数据，进行一次绘制 */
    void VisualizeOnce(pangolin::OpenGlRenderState &s_cam, pangolin::View &d_cam);
    /* 关闭绘图窗口 */
    void VisualizeStop(void);
    /* 判断绘图窗口是否要关闭 */
    bool VisualizeShouldQuit(void);



/* estimator_initialize : 后端求解器初始化相关 */
private:
    // 定义枚举类型，表示系统当前的状态
    enum Status {
        NeedInit = 0,       // 后端求解器需要进行初始化
        MargOldest = 1,     // 运行中，需要边缘化最旧帧
        MargSubnew = 2,     // 运行中，需要边缘化次新帧
        Running = 3         // 运行中
    } status;
    // 后端求解器初始化相关阈值
    float threshold_initMinIMUMotion;
    float threshold_initMinMeanParallax;
    int threshold_initMinCovisibleNum;
    // 初始化第一步寻找到的最佳配对帧
    std::pair<size_t, std::shared_ptr<CombinedFrame>> bestCorrespond;
    // 单目相机尺度因子
    float cameraScale;
    // 重力加速度向量以及预设参考
    Eigen::Vector3f gravity;
    Eigen::Vector3f targetGravity = Eigen::Vector3f(0, 0, 9.8);
    // 滑动窗口内每一帧的速度
    Eigen::VectorXf v_bb;
private:
    /* 为后端求解器初始化过程配置参数 */
    bool UpdateConfigParams_EstimatorInit(void);
    /* 后端求解器初始化，入口函数 */
    bool EstimatorInit(void);
    /* 第一步：检查 IMU 运动激励 */
    bool DetectIMUMotion(void);
    /* 第二步：计算滑动窗口内所有帧的位姿和所有特征点的位置的初值 */
    bool ComputeVisualSFMInitValue(void);
    /* 第三步：优化滑动窗口内所有帧的位姿和所有特征点的位置 */
    bool RefineVisualSFM_go_pos(void);
    bool RefineVisualSFM_go_invdep(void);
    bool RefineVisualSFM_rootvio_pos(void);
    bool RefineVisualSFM_rootvio_invdep(void);
    /* 第四步：估计 IMU 和相机之间的相对位姿 */
    bool EstimateExtrinsicPose(void);
    /* 第五步：基于视觉结果，估计 IMU 的角速度偏差 */
    bool EstimateGyroBias(void);
    /* 第六步：粗略估计每帧速度、重力向量和相机尺度 */
    bool EstimateVelocityScaleGravity3DOF(void);
    /* 获取指定向量的正切平面上的一组基底 */
    Eigen::Matrix<float, 3, 2> GetTangentBasis(Eigen::Vector3f &v);
    /* 第七步：优化每帧速度、重力向量和相机尺度的估计结果 */
    bool EstimateVelocityScaleGravity2DOF(void);
    /* 第八步：基于初始化结果，更新滑动窗口内的所有参数 */
    bool UpdateBasedOnInitialization(void);


/* estimator_optimize : 后端求解器运行部分 */
private:
    // 最新一次的先验信息
    struct PriorInfo {
        Eigen::MatrixXd H;
        Eigen::MatrixXd JTinv;
        Eigen::VectorXd b;
        Eigen::VectorXd r;
        Eigen::MatrixXd J;
    } prior;
    // 后端求解器相关阈值
    float threshold_minMeanParallax;
    int threshold_minCovisibleNum;
    float threshold_maxImuSumTime;

private:
    /* 为后端求解器配置参数 */
    bool UpdateConfigParams_Estimator(void);
    /* 后端求解器迭代优化，入口函数 */
    bool EstimatorOptimize(void);
    /* 第一步：计算最新帧和新特征点的初值 */
    bool ComputeNewsInitValue(void);
    /* 第二步：构造整个滑动窗口的优化问题，迭代优化所有参数 */
    bool EstimateAllInSlidingWindow_go(void);
    bool EstimateAllInSlidingWindow_rootvio(void);
    /* 第三步：判断次新帧是否为关键帧，以此决定边缘化策略 */
    bool DetermineMarginalizeStrategy(void);
    /* 第四步：边缘化指定帧，构造先验信息 */
    bool Marginalize_go(void);
    bool Marginalize_rootvio(void);
    /* 边缘化最旧帧，构造先验信息 */
    bool MarginalizeOldestFrame_go(void);
    bool MarginalizeOldestFrame_rootvio(void);
    /* 边缘化次新帧，构造先验信息 */
    bool MarginalizeSubnewFrame_go(void);
    bool MarginalizeSubnewFrame_rootvio(void);
};