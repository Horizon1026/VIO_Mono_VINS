#include <include/measurement/imu_preintegration.hpp>
#include <3rd_library/sophus/se3.hpp>
#include <iostream>

// 声明静态变量
float IMUPreintegration::accel_noise = 1e-6f;
float IMUPreintegration::gyro_noise = 1e-6f;
float IMUPreintegration::accel_random_walk = 1e-6f;
float IMUPreintegration::gyro_random_walk = 1e-6f;
Eigen::Matrix<float, 18, 18> IMUPreintegration::Q = Eigen::Matrix<float, 18, 18>::Identity() * 1e-3f;


/* 构造函数 */
IMUPreintegration::IMUPreintegration(const Eigen::Vector3f &bias_a, const Eigen::Vector3f &bias_g) {
    this->bias_a = bias_a;
    this->bias_g = bias_g;
    this->Reset();
    this->ResetBuff();
}


/* 新增加一个量测，进行一步积分，更新预积分结果 */
void IMUPreintegration::Propagate(float timeStamp, const Eigen::Vector3f &accel, const Eigen::Vector3f &gyro) {
    if (!this->buff.empty()) {
        // 进行一次中值积分
        float delta_t = static_cast<float>(timeStamp - this->buff.back().timeStamp);
        this->MidPointIntegrate(this->delta_p,
                                this->delta_r,
                                this->delta_v,
                                this->jacobian,
                                this->covariance,
                                this->bias_a,
                                this->bias_g,
                                this->buff.back().accel,
                                this->buff.back().gyro,
                                accel,
                                gyro,
                                delta_t);
        // 更新总积分时间
        this->sumTime += delta_t;
    } else {
        // 如果输入的是第一个数据，则更新线性化点
        this->linearized_accel = accel;
        this->linearized_gyro = gyro;
    }
    // 最新数据压入 buff 区
    struct IMUMessure messure;
    messure.accel = accel;
    messure.gyro = gyro;
    messure.timeStamp = timeStamp;
    this->buff.emplace_back(messure);
}


/* 基于大雅可比矩阵和偏差值的更新量，近似修正预积分值 */
void IMUPreintegration::Correct(const Eigen::Vector3f &delta_ba, const Eigen::Vector3f &delta_bg) {
    this->delta_r = this->delta_r * Sophus::SO3f::exp(this->jacobian.block<3, 3>(Order::R, Order::Bg) * delta_bg).unit_quaternion();
    this->delta_r.normalize();
    this->delta_v += this->jacobian.block<3, 3>(Order::V, Order::Ba) * delta_ba +
                     this->jacobian.block<3, 3>(Order::V, Order::Bg) * delta_bg;
    this->delta_p += this->jacobian.block<3, 3>(Order::P, Order::Ba) * delta_ba +
                     this->jacobian.block<3, 3>(Order::P, Order::Bg) * delta_bg; 
}


/* 重新遍历 buff，计算预积分值 */
void IMUPreintegration::Repropagate(void) {
    // 重置预积分结果
    this->Reset();
    // 全部重新积分
    for (auto i = this->buff.begin(); std::next(i) != this->buff.end(); ++i) {
        auto j = std::next(i);
        this->MidPointIntegrate(this->delta_p,
                                this->delta_r,
                                this->delta_v,
                                this->jacobian,
                                this->covariance,
                                this->bias_a,
                                this->bias_g,
                                i->accel,
                                i->gyro,
                                j->accel,
                                j->gyro,
                                static_cast<float>(j->timeStamp - i->timeStamp));
    }
    // 记录总共积分时间
    this->sumTime = static_cast<float>(this->buff.back().timeStamp - this->buff.front().timeStamp);
}


/* 重置预积分块 */
void IMUPreintegration::Reset(void) {
    this->delta_p.setZero();
    this->delta_r.setIdentity();
    this->delta_v.setZero();
    this->sumTime = 0;
    this->jacobian.setIdentity();
    this->covariance.setZero();
    this->linearized_accel = this->buff.front().accel;
    this->linearized_gyro = this->buff.front().gyro;
}


/* 清空数据缓冲区 */
void IMUPreintegration::ResetBuff(void) {
    this->buff.clear();
}


/* 获取数据缓冲区的引用 */
std::list<IMUPreintegration::IMUMessure> &IMUPreintegration::GetBuff(void) {
    return this->buff;
}


/* 设置加速度偏差量 */
void IMUPreintegration::SetBiasA(const Eigen::Vector3f &bias_a) {
    this->bias_a = bias_a;
}


/* 设置角速度偏差量 */
void IMUPreintegration::SetBiasG(const Eigen::Vector3f &bias_g) {
    this->bias_g = bias_g;
}


/* 获取加速度偏差量 */
Eigen::Vector3f IMUPreintegration::GetBiasA(void) {
    return this->bias_a;
}


/* 获取角速度偏差量 */
Eigen::Vector3f IMUPreintegration::GetBiasG(void) {
    return this->bias_g;
}


/* 获取雅可比矩阵中的部分块 */
void IMUPreintegration::GetJacobians(Eigen::Matrix3f &dr_dbg,
                                     Eigen::Matrix3f &dv_dbg,
                                     Eigen::Matrix3f &dv_dba,
                                     Eigen::Matrix3f &dp_dbg,
                                     Eigen::Matrix3f &dp_dba) {
    dr_dbg = this->jacobian.block<3, 3>(Order::R, Order::Bg);
    dv_dbg = this->jacobian.block<3, 3>(Order::V, Order::Bg);
    dv_dba = this->jacobian.block<3, 3>(Order::V, Order::Ba);
    dp_dbg = this->jacobian.block<3, 3>(Order::P, Order::Bg);
    dp_dba = this->jacobian.block<3, 3>(Order::P, Order::Ba);
}
Eigen::Matrix3f IMUPreintegration::GetDrDbg(void) {
    return this->jacobian.block<3, 3>(Order::R, Order::Bg);
}
Eigen::Matrix3f IMUPreintegration::GetDvDbg(void) {
    return this->jacobian.block<3, 3>(Order::V, Order::Bg);
}
Eigen::Matrix3f IMUPreintegration::GetDvDba(void) {
    return this->jacobian.block<3, 3>(Order::V, Order::Ba);
}  
Eigen::Matrix3f IMUPreintegration::GetDpDbg(void) {
    return this->jacobian.block<3, 3>(Order::P, Order::Bg);
}
Eigen::Matrix3f IMUPreintegration::GetDpDba(void) {
    return this->jacobian.block<3, 3>(Order::P, Order::Ba);
}


/* 获取大雅可比矩阵 */
Eigen::Matrix<float, 15, 15> &IMUPreintegration::GetJacobian(void) {
    return this->jacobian;
}


/* 获取协方差矩阵 */
Eigen::Matrix<float, 15, 15> &IMUPreintegration::GetCovariance(void) {
    return this->covariance;
}


/* 获取总积分时间 */
float IMUPreintegration::GetSumTime(void) {
    return this->sumTime;
}


/* 获取预积分结果 */
void IMUPreintegration::GetDeltaRVP(Eigen::Quaternionf &delta_r,
                                    Eigen::Vector3f &delta_v,
                                    Eigen::Vector3f &delta_p) {
    delta_r = this->delta_r;
    delta_v = this->delta_v;
    delta_p = this->delta_p;
}
Eigen::Quaternionf IMUPreintegration::GetDeltaR(void) {
    return this->delta_r;
}
Eigen::Vector3f IMUPreintegration::GetDeltaV(void) {
    return this->delta_v;
}
Eigen::Vector3f IMUPreintegration::GetDeltaP(void) {
    return this->delta_p;
}


/* 打印出此预积分块的信息 */
void IMUPreintegration::PrintContent(void) {
    std::cout << "<IMU> Print content in Integration block:" << std::endl;
    std::cout << "  delta_r : " << this->delta_r.w() << ", " << this->delta_r.x() << ", " << this->delta_r.y() << ", " <<
        this->delta_r.z() << std::endl;
    std::cout << "  delta_v : " << this->delta_v.transpose() << std::endl;
    std::cout << "  delta_p : " << this->delta_p.transpose() << std::endl;
    std::cout << "  bias_a : " << this->bias_a.transpose() << std::endl;
    std::cout << "  bias_g : " << this->bias_g.transpose() << std::endl;
    std::cout << "  jacobian :" << std::endl << this->jacobian << std::endl;
    std::cout << "  covariance :" << std::endl << this->covariance << std::endl;
    std::cout << "  sum time : " << this->sumTime << std::endl;
    std::cout << "  buff size : " << this->buff.size() << std::endl;
    for (auto it = this->buff.begin(); it != this->buff.end(); ++it) {
        std::cout << "    [t, a, g] = " << it->timeStamp << ", [" << it->accel.transpose() << "], [" << it->gyro.transpose() << "]\n";
    }
}


/* 计算反对称矩阵 */
Eigen::Matrix3f IMUPreintegration::SkewSymmetricMatrix(const Eigen::Vector3f &v) {
    Eigen::Matrix3f M;
    M << 0, - v.z(), v.y(),
         v.z(), 0, - v.x(),
         - v.y(), v.x(), 0;
    return M;
}


/* 中值积分法 */
void IMUPreintegration::MidPointIntegrate(Eigen::Vector3f &delta_p_0,
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
                                          float delta_t) {
    // 计算角速度中值，由此确定姿态的相对变化量
    Eigen::Vector3f midGyro = 0.5 * (gyro_0 + gyro_1) - bias_g;
    Eigen::Quaternionf dq(1.0, delta_t * 0.5 * midGyro.x(), delta_t * 0.5 * midGyro.y(), delta_t * 0.5 * midGyro.z());

    // 计算下一时刻的姿态预积分，进一步计算加速度中值
    Eigen::Quaternionf delta_r_1 = delta_r_0 * dq;
    delta_r_1.normalize();
    Eigen::Matrix3f delta_R_0(delta_r_0);
    Eigen::Matrix3f delta_R_1(delta_r_1);
    Eigen::Vector3f midAccel = 0.5 * (delta_R_0 * (accel_0 - bias_a) + delta_R_1 * (accel_1 - bias_a));

    // 计算下一时刻的位置和速度预积分
    Eigen::Vector3f delta_p_1 = delta_p_0 + delta_v_0 * delta_t + 0.5 * midAccel * delta_t * delta_t;
    Eigen::Vector3f delta_v_1 = delta_v_0 + midAccel * delta_t;

    // 计算一些临时变量
    Eigen::Vector3f a_0_x = accel_0 - bias_a;
    Eigen::Vector3f a_1_x = accel_1 - bias_a;
    Eigen::Matrix3f R_w_x = this->SkewSymmetricMatrix(midGyro);
    Eigen::Matrix3f R_a_0_x = this->SkewSymmetricMatrix(a_0_x);
    Eigen::Matrix3f R_a_1_x = this->SkewSymmetricMatrix(a_1_x);
    Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();

    // 计算 F 矩阵，默认误差项量位置为：位置、姿态、速度、加速度偏差、角速度偏差
    Eigen::MatrixXf F     = Eigen::MatrixXf::Zero(15, 15);
    F.block<3, 3>(0, 0)   = I3;
    F.block<3, 3>(0, 3)   = - 0.25f * delta_R_0 * R_a_0_x * delta_t * delta_t + 
                            - 0.25f * delta_R_1 * R_a_1_x * (I3 - R_w_x * delta_t) * delta_t * delta_t;
    F.block<3, 3>(0, 6)   = I3 * delta_t;
    F.block<3, 3>(0, 9)   = - 0.25f * (delta_R_0 + delta_R_1) * delta_t * delta_t;
    F.block<3, 3>(0, 12)  = - 0.25f * delta_R_1 * R_a_1_x * delta_t * delta_t * -delta_t;
    F.block<3, 3>(3, 3)   = I3 - R_w_x * delta_t;
    F.block<3, 3>(3, 12)  = - 1.0f * I3 * delta_t;
    F.block<3, 3>(6, 3)   = - 0.5f * delta_R_0 * R_a_0_x * delta_t -
                            0.5f * delta_R_1 * R_a_1_x * (I3 - R_w_x * delta_t) * delta_t;
    F.block<3, 3>(6, 6)   = I3;
    F.block<3, 3>(6, 9)   = - 0.5f * (delta_R_0 + delta_R_1) * delta_t;
    F.block<3, 3>(6, 12)  = - 0.5f * delta_R_1 * R_a_1_x * delta_t * -delta_t;
    F.block<3, 3>(9, 9)   = I3;
    F.block<3, 3>(12, 12) = I3;

    // 计算 V 矩阵
    Eigen::MatrixXf V     = Eigen::MatrixXf::Zero(15,18);
    V.block<3, 3>(0, 0)   = 0.25f * delta_R_0 * delta_t * delta_t;
    V.block<3, 3>(0, 3)   = - 0.25f * delta_R_1 * R_a_1_x  * delta_t * delta_t * 0.5f * delta_t;
    V.block<3, 3>(0, 6)   = 0.25f * delta_R_1 * delta_t * delta_t;
    V.block<3, 3>(0, 9)   = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3)   = 0.5f * I3 * delta_t;
    V.block<3, 3>(3, 9)   = 0.5f * I3 * delta_t;
    V.block<3, 3>(6, 0)   = 0.5f * delta_R_0 * delta_t;
    V.block<3, 3>(6, 3)   = - 0.5f * delta_R_1 * R_a_1_x  * delta_t * 0.5f * delta_t;
    V.block<3, 3>(6, 6)   = 0.5f * delta_R_1 * delta_t;
    V.block<3, 3>(6, 9)   = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12)  = I3 * delta_t;
    V.block<3, 3>(12, 15) = I3 * delta_t;

    // 更新大雅可比矩阵和协方差矩阵
    jacobian = F * jacobian;
    covariance = F * covariance * F.transpose() + V * IMUPreintegration::Q * V.transpose();

    // 更新预积分结果
    delta_p_0 = delta_p_1;
    delta_r_0 = delta_r_1;
    delta_v_0 = delta_v_1;
}