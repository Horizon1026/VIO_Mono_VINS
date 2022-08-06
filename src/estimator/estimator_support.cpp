#include <include/estimator/estimator.hpp>
#include <include/estimator/support/epipolar_geometry.h>
#include <include/estimator/support/perspective_n_point.h>

/* 给定位姿与观测，三角测量一个特征点，返回测量成功与否 */
bool VIOMono::TriangulateLandmark(const std::shared_ptr<Landmark> &landmark,
                                  const Eigen::Quaternionf &q_cw_1,
                                  const Eigen::Vector3f &t_cw_1,
                                  const Eigen::Vector2f &norm_1,
                                  const Eigen::Quaternionf &q_cw_2,
                                  const Eigen::Vector3f &t_cw_2,
                                  const Eigen::Vector2f &norm_2) {
    // 构造 pose 矩阵
    Eigen::Matrix<float, 3, 4> pose1, pose2;
    pose1.block<3, 3>(0, 0) = q_cw_1.toRotationMatrix();
    pose1.block<3, 1>(0, 3) = t_cw_1;
    pose2.block<3, 3>(0, 0) = q_cw_2.toRotationMatrix();
    pose2.block<3, 1>(0, 3) = t_cw_2;
    // 构造待求解方程
    Eigen::Matrix4f designMat = Eigen::Matrix4f::Zero();
	designMat.row(0) = norm_1[0] * pose1.row(2) - pose1.row(0);
	designMat.row(1) = norm_1[1] * pose1.row(2) - pose1.row(1);
	designMat.row(2) = norm_2[0] * pose2.row(2) - pose2.row(0);
	designMat.row(3) = norm_2[1] * pose2.row(2) - pose2.row(1);
    // 通过 SVD 方法求解方程
    Eigen::Vector4f expandPos = designMat.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    if (expandPos(2) < 0 && expandPos(3) < 0) {
        expandPos = - expandPos;
    }
    expandPos = expandPos / expandPos(3);
    landmark->p_w = expandPos.head<3>();
    // 验证结果合理性
    Eigen::Vector3f p_c_1 = q_cw_1 * landmark->p_w + t_cw_1;
    Eigen::Vector3f p_c_2 = q_cw_2 * landmark->p_w + t_cw_2;
    // std::cout << "<Estimator> Landmark triangulate pos_w is " << landmark->p_w.transpose() <<
    //     ", valid z is " << p_c_1.z() << ", " << p_c_2.z() << std::endl;
    if (p_c_1.z() < 0 || p_c_2.z() < 0) {
        return false;
    } else {
        return true;
    }
}


/* 指定一个特征点，使用观测到他的所有关键帧共同估计他的位置 */
bool VIOMono::TriangulateLandmark(const std::shared_ptr<Landmark> &landmark) {
    if (landmark == nullptr) {
        return false;
    }
    if (landmark->observes.size() < 2) {
        return false;
    }
    // 根据观测到此特征点的图像帧数量，确定待求解方程 Ax=0 的维度
    size_t rows = landmark->observes.size() * 2;
    Eigen::MatrixXf A;
    A.resize(rows, 4);
    // 遍历每一个观测帧，填充方程
    for (size_t i = 0; i < landmark->observes.size(); ++i) {
        auto norm = landmark->observes[i]->norm;
        size_t frameID = i + landmark->firstFrameID;
        auto frame = this->frameManager->GetFrame(frameID);
        Eigen::Matrix<float, 3, 4> T_cw;
        T_cw.block<3, 3>(0, 0) = frame->q_wc.inverse().toRotationMatrix();
        T_cw.block<3, 1>(0, 3) = - T_cw.block<3, 3>(0, 0) * frame->t_wc;
        A.block<1, 4>(2 * i    , 0) = norm.x() * T_cw.block<1, 4>(2, 0) - T_cw.block<1, 4>(0, 0);
        A.block<1, 4>(2 * i + 1, 0) = norm.y() * T_cw.block<1, 4>(2, 0) - T_cw.block<1, 4>(1, 0);
    }
    // 通过 SVD 分解求解方程
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f expandPos = svd.matrixV().rightCols<1>();
    if (expandPos(2) < 0 && expandPos(3) < 0) {
        expandPos = - expandPos;
    }
    expandPos = expandPos / expandPos(3);
    landmark->p_w = expandPos.head<3>();
    // 通过最后两个奇异值的比例判断是否三角化成功
    float rate = svd.singularValues()(2, 0) / svd.singularValues()(3, 0);
    if (rate < 1e6) {
        return false;
    }
    // 将特征点投影到每一个观测的相机坐标系中，判断深度正负
    for (size_t i = 0; i < landmark->observes.size(); ++i) {
        size_t frameID = i + landmark->firstFrameID;
        auto frame = this->frameManager->GetFrame(frameID);
        Eigen::Vector3f p_c = frame->q_wc.inverse() * (landmark->p_w - frame->t_wc);
        if (p_c.z() < 0) {
            return false;
        }
    }
    return true;

}


/* 指定两个图像帧，基于他们的已知相机位姿，三角测量他们共视的特征点 */
bool VIOMono::TriangulateLandmarks(const std::shared_ptr<CombinedFrame> &frame1,
                                   const std::shared_ptr<CombinedFrame> &frame2) {
    std::vector<std::shared_ptr<Landmark>> landmarks = frame1->GetCovisibleLandmarks(frame2);
    size_t cnt = 0;
    // 遍历每一个共视的特征点，如果没有三角测量过，则进行三角测量
    for (auto &item : landmarks) {
        if (item->isSolved == Landmark::SolveStatus::YES) {
            continue;
        }
        Eigen::Quaternionf q_cw_1 = frame1->q_wc.inverse();
        Eigen::Vector3f t_cw_1 = - q_cw_1.toRotationMatrix() * frame1->t_wc;
        Eigen::Vector2f norm_1 = item->GetNorm(frame1->ID);
        Eigen::Quaternionf q_cw_2 = frame2->q_wc.inverse();
        Eigen::Vector3f t_cw_2 = - q_cw_2.toRotationMatrix() * frame2->t_wc;
        Eigen::Vector2f norm_2 = item->GetNorm(frame2->ID);
        bool res = this->TriangulateLandmark(item, q_cw_1, t_cw_1, norm_1, q_cw_2, t_cw_2, norm_2);
        if (res == true) {
            item->isSolved = Landmark::SolveStatus::YES;
            ++cnt;
        } else {
            item->isSolved = Landmark::SolveStatus::NO;
        }
    }
    std::cout << "<Estimator> Triangulate " << cnt << "/" << landmarks.size() << " landmarks between frame " << frame1->ID <<
        " and " << frame2->ID << std::endl;
    return true;
}


/* 指定一个特征点，使用他的世界坐标系位置，更新他的逆深度 */
bool VIOMono::UsePosToUpdateInvdep(const std::shared_ptr<Landmark> &landmark) {
    if (landmark == nullptr) {
        return false;
    }
    auto frame = this->frameManager->GetFrame(landmark->firstFrameID);
    if (frame != nullptr) {
        Eigen::Vector3f p_c = frame->q_wc.inverse() * (landmark->p_w - frame->t_wc);
        float invdep = 1.0f / p_c.z();
        landmark->invDep = invdep;
        return true;
    } else {
        landmark->invDep = 0;
        return false;
    }
}


/* 指定一个特征点，使用他的逆深度，更新他的世界坐标系位置 */
bool VIOMono::UseInvdepToUpdatePos(const std::shared_ptr<Landmark> &landmark) {
    if (landmark == nullptr) {
        return false;
    }
    auto frame = this->frameManager->GetFrame(landmark->firstFrameID);
    if (frame != nullptr) {
        Eigen::Vector2f norm = landmark->GetNorm(frame->ID);
        Eigen::Vector3f p_c = Eigen::Vector3f(norm.x(), norm.y(), 1.0f);
        p_c /= landmark->invDep;
        landmark->p_w = frame->q_wc.toRotationMatrix() * p_c + frame->t_wc;
        return true;
    } else {
        return false;
    }
}


/* 指定两个图像帧，基于他们的共视特征点，估计出两者相对位姿 */
bool VIOMono::EstimateRelativePose(const std::shared_ptr<CombinedFrame> &frame1,
                                   const std::shared_ptr<CombinedFrame> &frame2,
                                   Eigen::Quaternionf &q_c1c2,
                                   Eigen::Vector3f &t_c1c2) {
    if (frame1 == frame2) {
        q_c1c2.setIdentity();
        t_c1c2.setZero();
        return true;
    }
    // 提取出两帧共视的点的归一化平面坐标
    std::vector<std::shared_ptr<Landmark>> landmarks = frame1->GetCovisibleLandmarks(frame2);
    if (landmarks.empty()) {
        return false;
    }
    std::vector<Eigen::Vector3d> norms1, norms2;
    norms1.reserve(landmarks.size());
    norms2.reserve(landmarks.size());
    Eigen::Vector2f temp;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        temp = landmarks[i]->GetNorm(frame1->ID);
        norms1.emplace_back(Eigen::Vector3d(temp.x(), temp.y(), 1.0));
        temp = landmarks[i]->GetNorm(frame2->ID);
        norms2.emplace_back(Eigen::Vector3d(temp.x(), temp.y(), 1.0));
    }
    // 基于对极约束模型估计出位姿
    EpipolarGeometryClass tool(0.0005, 30);
    std::vector<uchar> status;
    Eigen::Matrix3d R_c2c1;
    Eigen::Vector3d t_c2c1;
    tool.EstimateRotationAndTranslation(norms1, norms2, status, R_c2c1, t_c2c1);
    // 检查两帧之间的相对旋转是否合理
    double theta = std::acos(R_c2c1.trace() / 2.0 - 0.5);
    if (std::abs(theta) > 1.57) {
        return false;
    }

    q_c1c2 = Eigen::Quaternionf(R_c2c1.cast<float>().transpose());
    t_c1c2 = - q_c1c2.toRotationMatrix() * t_c2c1.cast<float>();

    return true;
}


/* 指定一个图像帧，基于他的已三角化特征点的世界坐标系位置，估计此帧位姿 */
bool VIOMono::EstimatePose(const std::shared_ptr<CombinedFrame> &frame) {
    // 提取出此帧观测到的特征点的 p_w 和 norm 序列
    std::vector<Eigen::Vector3d> pts2;
    std::vector<Eigen::Vector3d> pts3;
    pts2.reserve(frame->landmarks.size());
    pts3.reserve(frame->landmarks.size());
    Eigen::Vector3f pt3;
    Eigen::Vector2f pt2;
    for (auto &item : frame->landmarks) {
        if (item.second->isSolved != Landmark::SolveStatus::YES) {
            continue;
        }
        pt2 = item.second->GetNorm(frame->ID);
        pt3 = item.second->p_w;
        pts2.emplace_back(Eigen::Vector3d(pt2(0), pt2(1), 1.0));
        pts3.emplace_back(Eigen::Vector3d(pt3(0), pt3(1), pt3(2)));
    }
    if (pts2.size() < 10) {
        return false;
    }

    // 设定初值，求解 PnP 问题
    Eigen::Matrix3f R(frame->q_wc.inverse());
    Eigen::Vector3f t(- R * frame->t_wc);
    Eigen::Matrix3d R_ = R.cast<double>();
    Eigen::Vector3d t_ = t.cast<double>();
    PerspectiveNPointClass tool;
    std::vector<uchar> status;
    tool.EstimateRotationAndTranslation(pts3, pts2, R_, t_, status);

    // 校验结果
    Eigen::Matrix3d deltaR = R_.transpose() * R.cast<double>();
    double theta = std::acos(deltaR.trace() / 2.0 - 0.5);
    if (std::fabs(theta) > 0.785) {
        return false;
    } else {
        frame->q_wc = Eigen::Quaternionf(R_.transpose().cast<float>());
        frame->t_wc = - (R_.transpose() * t_).cast<float>();
    }

    std::cout << "<Estimator> Estimate frame " << frame->ID << " pose is [" << frame->q_wc.w() << ", " << frame->q_wc.x() <<
        ", " << frame->q_wc.y() << ", " << frame->q_wc.z() << "], [" << frame->t_wc.transpose() << "]" << std::endl;
    return true;
}

