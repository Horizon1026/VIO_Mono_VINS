#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <list>

/* 定义 IMU 预积分块类 */
class IMUPreintegration {
/* IMU 噪声模型参数以静态成员变量保存 */
public:
    static float accel_noise;
    static float gyro_noise;
    static float accel_random_walk;
    static float gyro_random_walk;
    static Eigen::Matrix<float, 18, 18> Q;

// 雅可比矩阵中的状态量索引
public:
    enum Order {
        P = 0,
        R = 3,
        V = 6,
        Ba = 9,
        Bg = 12
    };

/* 构造函数与析构函数 */
public:
    IMUPreintegration(const Eigen::Vector3f &bias_a, const Eigen::Vector3f &bias_g);
    ~IMUPreintegration() {}

/* 私有成员变量 */
private:
    // 量测数据缓冲区
    struct IMUMessure {
        Eigen::Vector3f accel;
        Eigen::Vector3f gyro;
        double timeStamp;
    };
    std::list<IMUMessure> buff;
    // 加速度和角速度偏差
    Eigen::Vector3f bias_a;
    Eigen::Vector3f bias_g;
    // 旋转、位置、速度预积分结果
    Eigen::Quaternionf delta_r;
    Eigen::Vector3f delta_p;
    Eigen::Vector3f delta_v;
    // 线性化点
    Eigen::Vector3f linearized_accel;
    Eigen::Vector3f linearized_gyro;
    // 协方差矩阵
    Eigen::Matrix<float, 15, 15> covariance;
    // 大雅可比矩阵
    Eigen::Matrix<float, 15, 15> jacobian;
    // 积分时间总和
    float sumTime;

/* 对外公开成员方法 */
public:
    /* 新增加一个量测，进行一步积分，更新预积分结果 */
    void Propagate(float timeStamp, const Eigen::Vector3f &accel, const Eigen::Vector3f &gyro);
    /* 基于大雅可比矩阵和偏差值的更新量，近似修正预积分值 */
    void Correct(const Eigen::Vector3f &delta_ba, const Eigen::Vector3f &delta_bg);
    /* 重新遍历 buff，计算预积分值 */
    void Repropagate(void);
    /* 重置预积分块 */
    void Reset(void);
    /* 清空数据缓冲区 */
    void ResetBuff(void);
    /* 获取数据缓冲区的引用 */
    std::list<IMUMessure> &GetBuff(void);
    /* 设置加速度偏差量 */
    void SetBiasA(const Eigen::Vector3f &bias_a);
    /* 设置角速度偏差量 */
    void SetBiasG(const Eigen::Vector3f &bias_g);
    /* 获取加速度偏差量 */
    Eigen::Vector3f GetBiasA(void);
    /* 获取角速度偏差量 */
    Eigen::Vector3f GetBiasG(void);
    /* 获取雅可比矩阵中的部分块 */
    void GetJacobians(Eigen::Matrix3f &dr_dbg,
                      Eigen::Matrix3f &dv_dbg,
                      Eigen::Matrix3f &dv_dba,
                      Eigen::Matrix3f &dp_dbg,
                      Eigen::Matrix3f &dp_dba);
    Eigen::Matrix3f GetDrDbg(void);
    Eigen::Matrix3f GetDvDbg(void);
    Eigen::Matrix3f GetDvDba(void);
    Eigen::Matrix3f GetDpDbg(void);
    Eigen::Matrix3f GetDpDba(void);
    /* 获取大雅可比矩阵 */
    Eigen::Matrix<float, 15, 15> &GetJacobian(void);
    /* 获取协方差矩阵 */
    Eigen::Matrix<float, 15, 15> &GetCovariance(void);
    /* 获取总积分时间 */
    float GetSumTime(void);
    /* 获取预积分结果 */
    void GetDeltaRVP(Eigen::Quaternionf &delta_r,
                     Eigen::Vector3f &delta_v,
                     Eigen::Vector3f &delta_p);
    Eigen::Quaternionf GetDeltaR(void);
    Eigen::Vector3f GetDeltaV(void);
    Eigen::Vector3f GetDeltaP(void);
    /* 打印出此预积分块的信息 */
    void PrintContent(void);

/* 对内私有成员方法 */
private:
    /* 计算反对称矩阵 */
    Eigen::Matrix3f SkewSymmetricMatrix(const Eigen::Vector3f &v);
    /* 中值积分法 */
    void MidPointIntegrate(Eigen::Vector3f &delta_p_0,
                           Eigen::Quaternionf &delta_r_0,
                           Eigen::Vector3f &delta_v_0,
                           Eigen::Matrix<float, 15, 15> &jacobian,
                           Eigen::Matrix<float, 15, 15> &covariance,
                           const Eigen::Vector3f &bias_a,
                           const Eigen::Vector3f &bias_g,
                           const Eigen::Vector3f &accel_0,
                           const Eigen::Vector3f &gyro_0,
                           const Eigen::Vector3f &accel_1,
                           const Eigen::Vector3f &gyro_1,
                           float delta_t);
};