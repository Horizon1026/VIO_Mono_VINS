#include <include/estimator/estimator.hpp>
#include <3rd_library/sophus/se3.hpp>

/* 为后端求解器初始化过程配置参数 */
bool VIOMono::UpdateConfigParams_EstimatorInit(void) {
    if (this->config == nullptr) {
        return false;
    }
    this->threshold_initMinCovisibleNum = this->config->initParams.minCovisibleNum;
    this->threshold_initMinIMUMotion = this->config->initParams.minIMUMotion;
    this->threshold_initMinMeanParallax = this->config->initParams.minMeanParallax;
    this->projectionInformation = this->config->cameraIntrinsic.fx * this->config->cameraIntrinsic.fy / 2.25f *
        Eigen::Matrix2f::Identity();
    return true;
}


/* 后端求解器初始化，入口函数 */
bool VIOMono::EstimatorInit(void) {
    std::cout << "<Estimator> Start initialization." << std::endl;
    // 第一步：检查运动激励
    bool res = this->DetectIMUMotion();
    if (res == false) {
        return false;
    }
    // 第二步：纯视觉三维重构计算初值
    res = this->ComputeVisualSFMInitValue();
    if (res == false) {
        return false;
    }
    // 第三步：优化视觉三维重构的结果
    switch (this->solverType) {
        default:
        case SolverType::GraphOptimizor:
            res = this->RefineVisualSFM_go_invdep();
            // res = this->RefineVisualSFM_go_pos();
            break;
        case SolverType::RootVIO:
            res = this->RefineVisualSFM_rootvio_invdep();
            // res = this->RefineVisualSFM_rootvio_pos();
            break;
    }
    // this->Visualize();
    for (auto &item : this->landmarkManager->landmarks) {
        if (item.second->isSolved != Landmark::SolveStatus::YES) {
            bool res = this->TriangulateLandmark(item.second);
            if (res == true) {
                item.second->isSolved = Landmark::SolveStatus::YES;
                this->UsePosToUpdateInvdep(item.second);
            }
        }
    }
    if (res == false) {
        return false;
    }
    // 第四步：估计 IMU 和相机之间的相对位姿
    res = this->EstimateExtrinsicPose();
    if (res == false) {
        return false;
    }
    // 第五步：基于视觉三维重构结果，估计角速度偏差
    res = this->EstimateGyroBias();
    if (res == false) {
        return false;
    }
    // 第六步：粗略估计重力加速度、单目相机尺度和每一帧的速度
    res = this->EstimateVelocityScaleGravity3DOF();
    if (res == false) {
        return false;
    }
    // 第七步：精细估计重力加速度
    res = this->EstimateVelocityScaleGravity2DOF();
    if (res == false) {
        return false;
    }
    // 第八步：更新滑动窗口内的参数
    res = this->UpdateBasedOnInitialization();
    std::cout << "<Estimator> Initialization succeed." << std::endl;
    return res;
}


/* 第一步：检查 IMU 运动激励 */
bool VIOMono::DetectIMUMotion(void) {
    // 计算滑动窗口内加速度平均值
    Eigen::Vector3f averageAccel = Eigen::Vector3f::Zero();
    long cnt = 0;
    for (auto &frame : this->frameManager->frames) {
        auto &imuBuff = frame->imu->GetBuff();
        cnt += imuBuff.size();
        for (auto it = imuBuff.begin(); it != imuBuff.end(); ++it) {
            averageAccel += it->accel;
        }
    }
    if (cnt == 0) {
        return false;
    }
    averageAccel /= static_cast<float>(cnt);

    // 计算加速度均方差
    float variance = 0.0f;
    for (auto &frame : this->frameManager->frames) {
        auto &imuBuff = frame->imu->GetBuff();
        for (auto it = imuBuff.begin(); it != imuBuff.end(); ++it) {
            Eigen::Vector3f temp = it->accel - averageAccel;
            variance += temp.dot(temp);
        }
    }
    variance /= static_cast<float>(cnt);

    // 当加速度均方差超过阈值才认为有足够的运动激励
    if (variance > this->threshold_initMinIMUMotion) {
        std::cout << "<Estimator> IMU motion is enough : " <<
            variance << " > " << this->threshold_initMinIMUMotion << std::endl;
        return true;
    } else {
        std::cout << "<Estimator> IMU motion is not enough : " <<
            variance << " <= " << this->threshold_initMinIMUMotion << std::endl;
        return false;
    }
}


/* 第八步：基于初始化结果，更新滑动窗口内的所有参数 */
bool VIOMono::UpdateBasedOnInitialization(void) {
    // 定义首帧相机坐标系中的重力加速度向量，以及世界坐标系中的重力加速度向量
    Eigen::Vector3f &g_c = this->gravity;
    Eigen::Vector3f &g_w = this->targetGravity;
    // 计算旋转向量和旋转角，确定从 c0 系旋转到 w 系的旋转矩阵
    float norm = (g_c.cross(g_w)).norm();
    Eigen::Vector3f u = g_c.cross(g_w) / norm;
    float theta = std::atan2(norm, g_c.transpose() * g_w);
    Eigen::Matrix3f R_wc0 = Sophus::SO3f::exp(u * theta).matrix();
    Eigen::Quaternionf q_wc0(R_wc0);
    std::cout << "<Estimator> The rotation from cam0 to word (q_w_c0) is [" << q_wc0.w() <<
        ", " << q_bc.x() << ", " << q_bc.y() << ", " << q_bc.z() << "]\n";
    // 将滑动窗口内的所有图像帧的位姿变换到世界坐标系中，同时恢复米制单位
    size_t idx = 0;
    for (auto &item : this->frameManager->frames) {
        Eigen::Matrix3f R_c0c = item->q_wc.toRotationMatrix();
        Eigen::Vector3f t_c0c = item->t_wc;
        item->q_wc = Eigen::Quaternionf(R_wc0 * R_c0c);
        item->t_wc = R_wc0 * t_c0c * this->cameraScale;
        item->v_wc = item->q_wc.toRotationMatrix() * this->v_bb.segment(idx * 3, 3);
        ++idx;
    }
    // 将滑动窗口内所有特征点位置变换到世界坐标系中，同时恢复米制单位
    for (auto &item : this->landmarkManager->landmarks) {
        auto &lm_ptr = item.second;
        lm_ptr->invDep /= this->cameraScale;
        lm_ptr->p_w = R_wc0 * lm_ptr->p_w * this->cameraScale;
    }
    // 将滑动窗口内所有 IMU 的位姿和速度变换到世界坐标系中
    idx = 0;
    for (auto &item : this->frameManager->frames) {
        item->q_wb = item->q_wc * this->q_bc.inverse();
        item->t_wb = - item->q_wb.toRotationMatrix() * this->t_bc + item->t_wc;
        item->v_wb = item->q_wb.toRotationMatrix() * this->v_bb.segment(idx * 3, 3);
        ++idx;
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