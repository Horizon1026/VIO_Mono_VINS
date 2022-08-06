#include <include/estimator/estimator.hpp>


/* 第四步：估计 IMU 和相机之间的相对位姿 */
bool VIOMono::EstimateExtrinsicPose(void) {
    if (this->config == nullptr) {
        return false;
    }

    // 加载配置结果，进行格式转换
    Eigen::Matrix3f R_bc;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            R_bc(i, j) = static_cast<float>(this->config->cameraExtrinsic.R_bc.at<double>(i, j));
        }
        this->t_bc(i) = static_cast<float>(this->config->cameraExtrinsic.t_bc.at<double>(i, 0));
    }
    this->q_bc = Eigen::Quaternionf(R_bc);
    std::cout << "<Estimator> Estimate Expose is [" << this->q_bc.w() << ", " << this->q_bc.x() <<
        ", " << this->q_bc.y() << ", " << this->q_bc.z() << "], [" << this->t_bc.transpose() << "]" << std::endl;
    
    // 补充计算滑动窗口内的位姿
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto &frame = *it;
        frame->q_wb = frame->q_wc * this->q_bc.inverse();
        frame->t_wb = - frame->q_wb.toRotationMatrix() * this->t_bc + frame->t_wc;
    }
    // T_wb = T_wc * T_bc.inverse();
    // [R_wb   t_wb] = [R_wc   t_wc] * [R_bc'   - R_bc' * t_bc]
    // [ 0      1  ]   [ 0       1 ]   [ 0              1     ]
    //               = [R_wc * R_bc'   R_wc * (- R_bc' * t_bc) + t_wc]
    //                 [     0                       1               ]
    //               = [R_wc * R_bc'   - R_wb * t_bc + t_wc]
    //                 [     0                   1         ]
    return true;
}


/* 第五步：基于视觉结果，估计 IMU 的角速度偏差 */
bool VIOMono::EstimateGyroBias(void) {
    for (size_t iter = 0; iter < 5; ++iter) {
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();
        for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
            auto &frame_i = *it;
            auto &frame_j = *std::next(it);
            Eigen::Quaternionf q_bb_ij = frame_j->imu->GetDeltaR();
            Eigen::Matrix<float, 3, 3> J_dr_dbg = frame_j->imu->GetDrDbg();
            Eigen::Vector3f residual = 2.0f * (q_bb_ij.inverse() * frame_i->q_wb.inverse() * frame_j->q_wb).vec();
            H += J_dr_dbg.transpose() * J_dr_dbg;
            b += J_dr_dbg.transpose() * residual;
        }
        Eigen::Vector3f delta_bg = H.ldlt().solve(b);
        if (std::isnan(delta_bg.norm())) {
            return false;
        }
        // 更新角速度偏差
        for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
            (*it)->imu->SetBiasG((*it)->imu->GetBiasG() + delta_bg);
        }
        // 重新预积分
        for (auto it = std::next(this->frameManager->frames.begin()); it != this->frameManager->frames.end(); ++it) {
            (*it)->imu->Repropagate();
        }
        std::cout << "<Estimator> Delta gyro bias is " << delta_bg.transpose() << std::endl;
        if (delta_bg.norm() < 1e-6) {
            break;
        }
    }
    std::cout << "<Estimator> Refined gyro bias is " << this->frameManager->frames.front()->imu->GetBiasG().transpose() << std::endl;
    return true;
}


/* 第六步：粗略估计每帧速度、重力向量和相机尺度 */
bool VIOMono::EstimateVelocityScaleGravity3DOF(void) {
    // 要优化每一帧的速度，共用的重力加速度向量，共用的尺度因子，因此可以得到整体维度
    size_t size = this->frameManager->frames.size() * 3 + 3 + 1;
    Eigen::MatrixXf A;
    A.setZero(size, size);
    Eigen::VectorXf b;
    b.setZero(size, 1);
    // 遍历每一对相邻的关键帧
    for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
        auto &frame_i = *it;
        auto &frame_j = *std::next(it);
        // 误差向量为【位置预积分 - 多项式，速度预积分】，有 6 维
        // 状态向量，对于每一对相邻关键帧而言，有 2 个速度，1 个重力加速度和 1 个尺度因子，有 10 维
        // 误差协方差矩阵的逆，维度和误差向量对应
        Eigen::Matrix<float, 6, 10> H = Eigen::Matrix<float, 6, 10>::Zero();
        Eigen::Matrix<float, 6, 1> z = Eigen::Matrix<float, 6, 1>::Zero();
        Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Identity();
        // 提取整个积分时间
        float dt = frame_j->imu->GetSumTime();
        // 提取前后两帧的位姿，其中提取的位置信息是去除了尺度的信息
        Eigen::Matrix3f R_wc_0(frame_i->q_wc);
        Eigen::Vector3f t_wc_0(frame_i->t_wc);
        Eigen::Matrix3f R_wc_1(frame_j->q_wc);
        Eigen::Vector3f t_wc_1(frame_j->t_wc);
        Eigen::Matrix3f R_bw_0 = this->q_bc.toRotationMatrix() * R_wc_0.transpose();
        Eigen::Matrix3f R_wb_1 = R_wc_1 * this->q_bc.toRotationMatrix().transpose();
        // 计算 H 矩阵
        H.block<3, 3>(0, 0) = - dt * Eigen::Matrix3f::Identity();
        H.block<3, 3>(0, 6) = 0.5 * R_bw_0 * dt * dt;
        H.block<3, 1>(0, 9) = R_bw_0 * (t_wc_1 - t_wc_0);
        H.block<3, 3>(3, 0) = - Eigen::Matrix3f::Identity();
        H.block<3, 3>(3, 3) = R_bw_0 * R_wb_1;
        H.block<3, 3>(3, 6) = R_bw_0 * dt;
        // 计算 z 向量
        z.block<3, 1>(0, 0) = frame_j->imu->GetDeltaP() - this->t_bc + R_bw_0 * R_wb_1 * this->t_bc;
        z.block<3, 1>(3, 0) = frame_j->imu->GetDeltaV();
        // 构建局部方程
        Eigen::MatrixXf sub_A = H.transpose() * Q * H;
        Eigen::VectorXf sub_b = H.transpose() * Q * z;
        // 将局部方程添加到全局方程中
        size_t idx = frame_i->ID - this->frameManager->frames.front()->ID;
        A.block<6, 6>(idx * 3, idx * 3) += sub_A.topLeftCorner<6, 6>();
        b.segment<6>(idx * 3) += sub_b.head<6>();
        A.bottomRightCorner<4, 4>() += sub_A.bottomRightCorner<4, 4>();
        b.tail<4>() += sub_b.tail<4>();
        A.block<6, 4>(idx * 3, size - 4) += sub_A.topRightCorner<6, 4>();
        A.block<4, 6>(size - 4, idx * 3) += sub_A.bottomLeftCorner<4, 6>();
    }
    Eigen::VectorXf x = A.ldlt().solve(b);
    this->cameraScale = x.tail<1>()[0];
    this->gravity = x.segment<3>(size - 4);
    this->v_bb = x.segment(0, this->frameManager->frames.size() * 3);
    std::cout << "<Estimator> Estimated gravity is " << this->gravity.transpose() << ", norm is " <<
        this->gravity.norm() << ", scale is " << this->cameraScale << std::endl;
    // 判断结果是否合法
    if (this->cameraScale < 0 || std::abs(this->gravity.norm() - this->targetGravity.norm()) > 1.0f) {
        return false;
    } else {
        return true;
    }
}


/* 获取指定向量的正切平面上的一组基底 */
Eigen::Matrix<float, 3, 2> VIOMono::GetTangentBasis(Eigen::Vector3f &v) {
    Eigen::Matrix<float, 3, 2> b0b1;
    Eigen::Vector3f a = v.normalized();
    Eigen::Vector3f b(0, 0, 1);
    if (a == b) {
        b << 1, 0, 0;
    }
    b0b1.block<3, 1>(0, 0) = (b - a * (a.transpose() * b)).normalized();
    b0b1.block<3, 1>(0, 1) = a.cross(b0b1.block<3, 1>(0, 0));
    return b0b1;
}


/* 第七步：优化每帧速度、重力向量和相机尺度的估计结果 */
bool VIOMono::EstimateVelocityScaleGravity2DOF(void) {
    // 要优化每一帧的速度，共用的重力加速度向量（自由度固定一维），共用的尺度因子，因此可以得到整体维度
    size_t size = this->frameManager->frames.size() * 3 + 2 + 1;
    Eigen::MatrixXf A;
    A.setZero(size, size);
    Eigen::VectorXf b;
    b.setZero(size, 1);
    // 为加速度估计量提供初值
    Eigen::Vector3f tempGravity = this->gravity.normalized() * this->targetGravity.norm();
    // 提取出加速度向量的正切平面的一组正交基底
    Eigen::Matrix<float, 3, 2> base = this->GetTangentBasis(tempGravity);

    for (size_t iter = 0; iter < 4; ++iter) {
        // 遍历每一对相邻的关键帧
        for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
            auto &frame_i = *it;
            auto &frame_j = *std::next(it);
            // 误差向量为【位置预积分 - 多项式，速度预积分】，有 6 维
            // 状态向量，对于每一对相邻关键帧而言，有 2 个速度，1 个重力加速度和 1 个尺度因子，有 9 维
            // 误差协方差矩阵的逆，维度和误差向量对应
            Eigen::Matrix<float, 6, 9> H = Eigen::Matrix<float, 6, 9>::Zero();
            Eigen::Matrix<float, 6, 1> z = Eigen::Matrix<float, 6, 1>::Zero();
            Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Identity();
            // 提取整个积分时间
            float dt = frame_j->imu->GetSumTime();
            // 提取前后两帧的位姿，其中提取的位置信息是去除了尺度的信息
            Eigen::Matrix3f R_wc_0(frame_i->q_wc);
            Eigen::Vector3f t_wc_0(frame_i->t_wc);
            Eigen::Matrix3f R_wc_1(frame_j->q_wc);
            Eigen::Vector3f t_wc_1(frame_j->t_wc);
            Eigen::Matrix3f R_bw_0 = this->q_bc.toRotationMatrix() * R_wc_0.transpose();
            Eigen::Matrix3f R_wb_1 = R_wc_1 * this->q_bc.toRotationMatrix().transpose();
            // 计算 H 矩阵
            H.block<3, 3>(0, 0) = - dt * Eigen::Matrix3f::Identity();
            H.block<3, 2>(0, 6) = 0.5 * R_bw_0 * dt * dt * base;
            H.block<3, 1>(0, 8) = R_bw_0 * (t_wc_1 - t_wc_0);
            H.block<3, 3>(3, 0) = - Eigen::Matrix3f::Identity();
            H.block<3, 3>(3, 3) = R_bw_0 * R_wb_1;
            H.block<3, 2>(3, 6) = R_bw_0 * base * dt;
            // 计算 z 向量
            z.block<3, 1>(0, 0) = frame_j->imu->GetDeltaP() - this->t_bc + R_bw_0 * R_wb_1 * this->t_bc;
            z.block<3, 1>(3, 0) = frame_j->imu->GetDeltaV() - R_bw_0 * dt * tempGravity;
            // 构建局部方程
            Eigen::MatrixXf sub_A = H.transpose() * Q * H;
            Eigen::VectorXf sub_b = H.transpose() * Q * z;
            // 将局部方程添加到全局方程中
            size_t idx = frame_i->ID - this->frameManager->frames.front()->ID;
            A.block<6, 6>(idx * 3, idx * 3) += sub_A.topLeftCorner<6, 6>();
            b.segment<6>(idx * 3) += sub_b.head<6>();
            A.bottomRightCorner<3, 3>() += sub_A.bottomRightCorner<3, 3>();
            b.tail<3>() += sub_b.tail<3>();
            A.block<6, 3>(idx * 3, size - 3) += sub_A.topRightCorner<6, 3>();
            A.block<3, 6>(size - 3, idx * 3) += sub_A.bottomLeftCorner<3, 6>();
        }
        Eigen::VectorXf x = A.ldlt().solve(b);
        Eigen::Vector2f dg = x.segment<2>(size - 3);
        tempGravity = (tempGravity + base * dg).normalized() * this->targetGravity.norm();
        this->v_bb = x.segment(0, this->frameManager->frames.size() * 3);
        if (dg.norm() < 1e-6) {
            break;
        }
    }
    // 重力加速度向量取多次迭代后的结果
    this->gravity = tempGravity;
    std::cout << "<Estimator> Refined gravity is " << this->gravity.transpose() << ", norm is " <<
        this->gravity.norm() << std::endl;
    return true;
}

