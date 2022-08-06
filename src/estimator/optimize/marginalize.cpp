#include <include/estimator/estimator.hpp>
#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <include/ba_solver/root_vio/problem_vio.hpp>
#include <include/estimator/vertices/vertex_pose.hpp>
#include <include/estimator/vertices/vertex_landmark_pos.hpp>
#include <include/estimator/vertices/vertex_landmark_invdep.hpp>
#include <include/estimator/vertices/vertex_velocity_bias.hpp>
#include <include/estimator/edges/edge_camera_factor.hpp>
#include <include/estimator/edges/edge_imu_factor.hpp>


/* 第三步：判断次新帧是否为关键帧，以此决定边缘化策略 */
bool VIOMono::DetermineMarginalizeStrategy(void) {
    // 定位到次新帧 subNew 和次次新帧 secondNew
    auto subNew_it = std::next(this->frameManager->frames.rbegin());
    if (subNew_it == this->frameManager->frames.rend()) {
        return false;
    }
    auto secondNew_it = std::next(subNew_it);
    if (secondNew_it == this->frameManager->frames.rend()) {
        return false;
    }
    auto subNew = *subNew_it;
    auto secondNew = *secondNew_it;
    // 计算这两帧的平均视差和匹配点个数 std::pair<float, size_t>
    auto correspond = subNew->GetCorrespondence(secondNew);
    std::cout << "<Marginalizor> Check [parallax, points] is [" << correspond.first << "/" << this->threshold_minMeanParallax <<
            ", " << correspond.second << "/" << this->threshold_minCovisibleNum << "], ";
    // 基于阈值，决定边缘化策略
    if (correspond.first > this->threshold_minMeanParallax || correspond.second < this->threshold_minCovisibleNum) {
        this->status = Status::MargOldest;
        std::cout << "need marginalize oldest frame." << std::endl;
    } else {
        this->status = Status::MargSubnew;
        std::cout << "need marginalize subnew frame." << std::endl;
    }
    return true;
}


/* 第四步：边缘化指定帧，构造先验信息 */
bool VIOMono::Marginalize_go(void) {
    switch (this->status) {
        case Status::MargOldest:
            return this->MarginalizeOldestFrame_go();
        case Status::MargSubnew:
            return this->MarginalizeSubnewFrame_go();
        default:
            return false;
    }
}


/* 第四步：边缘化指定帧，构造先验信息 */
bool VIOMono::Marginalize_rootvio(void) {
    switch (this->status) {
        case Status::MargOldest:
            return this->MarginalizeOldestFrame_rootvio();
        case Status::MargSubnew:
            return this->MarginalizeSubnewFrame_rootvio();
        default:
            return false;
    }
}


/* 边缘化最旧帧，构造先验信息 */
bool VIOMono::MarginalizeOldestFrame_go(void) {
    using Scalar = double;
    using ParamType = GraphOptimizor::VectorX<Scalar>;
    using PoseType = GraphOptimizor::VertexPose<Scalar>;
    using LandmarkType = GraphOptimizor::VertexLandmarkInvDepth<Scalar>;
    using MotionType = GraphOptimizor::VertexVelocityBias<Scalar>;
    using CameraFactor = GraphOptimizor::EdgeCameraFactor<Scalar>;
    using ImuFactor = GraphOptimizor::EdgeIMUFactor<Scalar>;
    using KernelType = GraphOptimizor::HuberKernel<Scalar>;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_ex = 0;
    size_t type_camera = 1;
    size_t type_landmark = 2;
    size_t type_motion = 3;
    size_t sizeof_prior = 0;

    // 添加 IMU 与相机的外参数节点，但因为是构造先验信息，因此不需要固定
    std::shared_ptr<PoseType> exPoseVertex(new PoseType());
    for (size_t i = 0; i < 1; ++i) {
        ParamType param(7);
        param << this->t_bc.cast<Scalar>(), this->q_bc.x(), this->q_bc.y(), this->q_bc.z(), this->q_bc.w();
        exPoseVertex->SetParameters(param);
        exPoseVertex->SetType(type_ex);
        problem.AddVertex(exPoseVertex);
        sizeof_prior += exPoseVertex->GetCalculationDimension();
    }

    // 添加相机 Pose 节点和 Motion(Velocity, Bias) 节点
    std::unordered_map<size_t, std::shared_ptr<PoseType>> cameraVertices;
    std::unordered_map<size_t, std::shared_ptr<MotionType>> motionVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        ParamType paramCamera(7);
        paramCamera << (*it)->t_wb.cast<Scalar>(), (*it)->q_wb.x(), (*it)->q_wb.y(), (*it)->q_wb.z(), (*it)->q_wb.w();
        std::shared_ptr<PoseType> cameraVertex(new PoseType());
        cameraVertex->SetParameters(paramCamera);
        cameraVertex->SetType(type_camera);
        problem.AddVertex(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        sizeof_prior += cameraVertex->GetCalculationDimension();

        ParamType paramMotion(9);
        paramMotion << (*it)->v_wb.cast<Scalar>(), (*it)->imu->GetBiasA().cast<Scalar>(), (*it)->imu->GetBiasG().cast<Scalar>();
        std::shared_ptr<MotionType> motionVertex(new MotionType());
        motionVertex->SetParameters(paramMotion);
        motionVertex->SetType(type_motion);
        problem.AddVertex(motionVertex);
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
        sizeof_prior += motionVertex->GetCalculationDimension();
    }
    std::cout << "<Marginalizor> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera and motion vertices." << std::endl;

    // 添加特征点逆深度节点，同时添加视觉残差边
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->isSolved != Landmark::SolveStatus::YES) {
            continue;
        }
        // 如果此特征点的首次观测帧不是要被边缘化的最旧帧，直接跳过
        if (this->frameManager->frames.front()->ID != lm_ptr->firstFrameID) {
            continue;
        }
        if (lm_ptr->observes.size() > 1 && lm_ptr->firstFrameID < this->frameManager->frames.back()->ID - 2) {
            ParamType paramLandmark(1);
            paramLandmark << lm_ptr->invDep;
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType());
            landmarkVertex->SetParameters(paramLandmark);
            landmarkVertex->SetType(type_landmark);
            problem.AddVertex(landmarkVertex);
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));

            // 构造相机残差边
            for (size_t i = 1; i < lm_ptr->observes.size(); ++i) {
                GraphOptimizor::Vector2<Scalar> norm_0, norm_i;
                norm_0 = lm_ptr->GetNorm(lm_ptr->firstFrameID).cast<Scalar>();
                norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i).cast<Scalar>();
                auto cam_0 = cameraVertices.find(lm_ptr->firstFrameID);
                auto cam_i = cameraVertices.find(lm_ptr->firstFrameID + i);
                std::shared_ptr<CameraFactor> edge(new CameraFactor(norm_0, norm_i));
                edge->AddVertex(landmarkVertex, 0);
                edge->AddVertex(cam_0->second, 1);
                edge->AddVertex(cam_i->second, 2);
                edge->AddVertex(exPoseVertex, 3);
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                edge->SetKernel(kernel);
                edge->SetInformation(this->projectionInformation.cast<Scalar>());
                problem.AddEdge(edge);
                ++cnt;
            }
        }
    }
    std::cout << "<Marginalizor> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark vertices." << std::endl;
    std::cout << "<Marginalizor> Add " << cnt << " camera residual factors." << std::endl;

    // 添加 IMU 残差边
    cnt = 0;
    auto &frame_i = *(this->frameManager->frames.begin());
    auto &frame_j = *std::next(this->frameManager->frames.begin());
    auto imu = frame_j->imu;
    // 忽略积分时间过长的 IMU 预积分块
    if (imu->GetSumTime() < this->threshold_maxImuSumTime) {
        auto cam_vertex_i = cameraVertices.find(frame_i->ID)->second;
        auto cam_vertex_j = cameraVertices.find(frame_j->ID)->second;
        auto motion_vertex_i = motionVertices.find(frame_i->ID)->second;
        auto motion_vertex_j = motionVertices.find(frame_j->ID)->second;
        std::shared_ptr<ImuFactor> edge(new ImuFactor(imu, this->targetGravity.cast<Scalar>()));
        edge->AddVertex(cam_vertex_i, 0);
        edge->AddVertex(motion_vertex_i, 1);
        edge->AddVertex(cam_vertex_j, 2);
        edge->AddVertex(motion_vertex_j, 3);
        problem.AddEdge(edge);
        cnt = 1;
    }
    std::cout << "<Marginalizor> Add " << cnt << " imu residual factors." << std::endl;

    // 添加先验信息
    if (this->prior.H.rows() > 0) {
        problem.SetPrior(this->prior.H.cast<Scalar>(), this->prior.b.cast<Scalar>(), this->prior.JTinv.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Marginalizor> Add prior factors with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;
    }

    // 配置边缘化参数，开始边缘化
    problem.SetMargnedVertexTypesWhenSolving(type_landmark);    // 所有首帧观测是最旧帧的特征点都被边缘化
    std::vector<std::shared_ptr<GraphOptimizor::VertexBase<Scalar>>> needMarg;
    needMarg.emplace_back(cameraVertices.find(this->frameManager->frames.front()->ID)->second);
    needMarg.emplace_back(motionVertices.find(this->frameManager->frames.front()->ID)->second);
    GraphOptimizor::Problem<Scalar>::Options options;
    options.minPriorItem = 1e-10;
    problem.SetOptions(options);
    bool res = problem.Marginalize(needMarg, sizeof_prior - 15);
    if (res == false) {
        return false;
    }
    GraphOptimizor::MatrixX<Scalar> prior_H, prior_JTinv;
    GraphOptimizor::VectorX<Scalar> prior_b, prior_r;
    problem.GetPrior(prior_H, prior_b, prior_JTinv, prior_r);
    this->prior.H = prior_H.cast<double>();
    this->prior.JTinv = prior_JTinv.cast<double>();
    this->prior.b = prior_b.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Marginalizor> Make prior with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;

    return true;
}


/* 边缘化次新帧，构造先验信息 */
bool VIOMono::MarginalizeSubnewFrame_go(void) {
    using Scalar = double;
    using ParamType = GraphOptimizor::VectorX<Scalar>;
    using PoseType = GraphOptimizor::VertexPose<Scalar>;
    using LandmarkType = GraphOptimizor::VertexLandmarkInvDepth<Scalar>;
    using MotionType = GraphOptimizor::VertexVelocityBias<Scalar>;
    using CameraFactor = GraphOptimizor::EdgeCameraFactor<Scalar>;
    using ImuFactor = GraphOptimizor::EdgeIMUFactor<Scalar>;
    using KernelType = GraphOptimizor::HuberKernel<Scalar>;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_ex = 0;
    size_t type_camera = 1;
    size_t type_landmark = 2;
    size_t type_motion = 3;
    size_t sizeof_prior = 0;

    // 添加 IMU 与相机的外参数节点，但因为是构造先验信息，因此不需要固定
    std::shared_ptr<PoseType> exPoseVertex(new PoseType());
    for (size_t i = 0; i < 1; ++i) {
        ParamType param(7);
        param << this->t_bc.cast<Scalar>(), this->q_bc.x(), this->q_bc.y(), this->q_bc.z(), this->q_bc.w();
        exPoseVertex->SetParameters(param);
        exPoseVertex->SetType(type_ex);
        problem.AddVertex(exPoseVertex);
        sizeof_prior += exPoseVertex->GetCalculationDimension();
    }

    // 添加相机 Pose 节点和 Motion(Velocity, Bias) 节点
    std::unordered_map<size_t, std::shared_ptr<PoseType>> cameraVertices;
    std::unordered_map<size_t, std::shared_ptr<MotionType>> motionVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        ParamType paramCamera(7);
        paramCamera << (*it)->t_wb.cast<Scalar>(), (*it)->q_wb.x(), (*it)->q_wb.y(), (*it)->q_wb.z(), (*it)->q_wb.w();
        std::shared_ptr<PoseType> cameraVertex(new PoseType());
        cameraVertex->SetParameters(paramCamera);
        cameraVertex->SetType(type_camera);
        problem.AddVertex(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        sizeof_prior += cameraVertex->GetCalculationDimension();

        ParamType paramMotion(9);
        paramMotion << (*it)->v_wb.cast<Scalar>(), (*it)->imu->GetBiasA().cast<Scalar>(), (*it)->imu->GetBiasG().cast<Scalar>();
        std::shared_ptr<MotionType> motionVertex(new MotionType());
        motionVertex->SetParameters(paramMotion);
        motionVertex->SetType(type_motion);
        problem.AddVertex(motionVertex);
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
        sizeof_prior += motionVertex->GetCalculationDimension();
    }
    std::cout << "<Marginalizor> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera and motion vertices." << std::endl;

    // 添加先验信息
    if (this->prior.H.rows() > 0) {
        problem.SetPrior(this->prior.H.cast<Scalar>(), this->prior.b.cast<Scalar>(), this->prior.JTinv.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Marginalizor> Add prior factors with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;
    }

    // 配置边缘化参数，开始边缘化
    problem.SetMargnedVertexTypesWhenSolving(type_landmark);    // 所有首帧观测是最旧帧的特征点都被边缘化
    std::vector<std::shared_ptr<GraphOptimizor::VertexBase<Scalar>>> needMarg;
    needMarg.emplace_back(cameraVertices.find(this->frameManager->frames.back()->ID - 1)->second);
    needMarg.emplace_back(motionVertices.find(this->frameManager->frames.back()->ID - 1)->second);
    GraphOptimizor::Problem<Scalar>::Options options;
    options.minPriorItem = 1e-10;
    problem.SetOptions(options);
    bool res = problem.Marginalize(needMarg, sizeof_prior - 15);
    if (res == false) {
        return false;
    }
    GraphOptimizor::MatrixX<Scalar> prior_H, prior_JTinv;
    GraphOptimizor::VectorX<Scalar> prior_b, prior_r;
    problem.GetPrior(prior_H, prior_b, prior_JTinv, prior_r);
    this->prior.H = prior_H.cast<double>();
    this->prior.JTinv = prior_JTinv.cast<double>();
    this->prior.b = prior_b.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Marginalizor> Make prior with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;
    return true;
}


/* 边缘化最旧帧，构造先验信息 */
bool VIOMono::MarginalizeOldestFrame_rootvio(void) {
    using Scalar = double;
    using ExPoseType = rootVIO::VertexExPose<Scalar>;
    using PoseType = rootVIO::VertexCameraPose<Scalar>;
    using LandmarkType = rootVIO::VertexLandmark<Scalar>;
    using MotionType = rootVIO::VertexVelocityBias<Scalar>;
    using KernelType = rootVIO::HuberKernel<Scalar>;
    using CameraFactor = rootVIO::CameraObserve<Scalar>;
    using IMUFactor = rootVIO::IMUBlock<Scalar>;
    rootVIO::ProblemVIO<Scalar> problem;
    size_t sizeof_prior = 0;

    // 为 VIO 问题添加相机外参节点
    std::shared_ptr<ExPoseType> exPoseVertex(new ExPoseType(this->q_bc.cast<Scalar>(), this->t_bc.cast<Scalar>()));
    problem.AddExPose(exPoseVertex);
    sizeof_prior += 6;
    std::cout << "<Marginalizor> Add 1 ex pose vertex." << std::endl;

    // 为 VIO 问题添加 camera pose 节点和 velocity bias 节点
    std::unordered_map<size_t, std::shared_ptr<PoseType>> cameraVertices;
    std::unordered_map<size_t, std::shared_ptr<MotionType>> motionVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        std::shared_ptr<PoseType> cameraVertex(new PoseType((*it)->q_wb.cast<Scalar>(),
                                                            (*it)->t_wb.cast<Scalar>()));
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        problem.AddCamera(cameraVertex);
        sizeof_prior += 6;

        std::shared_ptr<MotionType> motionVertex(new MotionType((*it)->v_wb.cast<Scalar>(),
                                                                (*it)->imu->GetBiasA().cast<Scalar>(),
                                                                (*it)->imu->GetBiasG().cast<Scalar>()));
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
        problem.AddVelocityBias(motionVertex);
        sizeof_prior += 9;
    }
    std::cout << "<Marginalizor> Add " << problem.GetCamerasNum() << " camera pose vertices and " << problem.GetVelocityBiasesNum() <<
        " velocity bias vertices." << std::endl;
    
    // 为 VIO 问题添加特征点位置节点，并对应添加视觉约束，同时只考虑首次观测为最旧帧的特征点
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->isSolved != Landmark::SolveStatus::YES) {
            continue;
        }
        // 如果此特征点的首次观测帧不是要被边缘化的最旧帧，直接跳过
        if (this->frameManager->frames.front()->ID != lm_ptr->firstFrameID) {
            continue;
        }
        if (lm_ptr->observes.size() > 1 && lm_ptr->firstFrameID < this->frameManager->frames.back()->ID - 2) {
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType(Scalar(lm_ptr->invDep)));
            std::unordered_map<size_t, std::shared_ptr<CameraFactor>> observes;
            for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
                rootVIO::Vector2<Scalar> norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i).cast<Scalar>();
                auto cam_i = (*cameraVertices.find(lm_ptr->firstFrameID + i)).second;
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                std::shared_ptr<CameraFactor> observe(new CameraFactor(
                    cam_i, norm_i, this->projectionInformation.cast<Scalar>(), kernel));
                observes.insert(std::make_pair(lm_ptr->firstFrameID + i - this->frameManager->frames.front()->ID, observe));
                ++cnt;
            }
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));
            problem.AddLandmark(landmarkVertex, observes, 1);
        }
    }
    std::cout << "<Marginalizor> Add " << problem.GetLandmarksNum() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark blocks." << std::endl;
    std::cout << "<Marginalizor> Add " << cnt << " camera residual factors." << std::endl;
    
    // 为 VIO 问题添加 IMU 观测约束，仅考虑与被边缘化的最旧帧有关的一个约束
    cnt = 0;
    auto &frame_i = *(this->frameManager->frames.begin());
    auto &frame_j = *(std::next(this->frameManager->frames.begin()));
    auto imu = frame_j->imu;
    // 如果积分时间过长，则不考虑
    if (imu->GetSumTime() < this->threshold_maxImuSumTime) {
        std::vector<std::shared_ptr<PoseType>> cameras;
        std::vector<std::shared_ptr<MotionType>> vbs;
        cameras.emplace_back(cameraVertices.find(frame_i->ID)->second);
        cameras.emplace_back(cameraVertices.find(frame_j->ID)->second);
        vbs.emplace_back(motionVertices.find(frame_i->ID)->second);
        vbs.emplace_back(motionVertices.find(frame_j->ID)->second);
        
        rootVIO::IMUBlock<Scalar>::LinearlizedPoint linear;
        linear.bias_a = imu->GetBiasA().cast<Scalar>();
        linear.bias_g = imu->GetBiasG().cast<Scalar>();
        linear.delta_p = imu->GetDeltaP().cast<Scalar>();
        linear.delta_r = imu->GetDeltaR().cast<Scalar>();
        linear.delta_v = imu->GetDeltaV().cast<Scalar>();

        rootVIO::IMUBlock<Scalar>::IMUJacobians jacobians;
        jacobians.dp_dba = imu->GetDpDba().cast<Scalar>();
        jacobians.dp_dbg = imu->GetDpDbg().cast<Scalar>();
        jacobians.dr_dbg = imu->GetDrDbg().cast<Scalar>();
        jacobians.dv_dba = imu->GetDvDba().cast<Scalar>();
        jacobians.dv_dbg = imu->GetDvDbg().cast<Scalar>();

        rootVIO::IMUBlock<Scalar>::Order order;
        order.P = IMUPreintegration::Order::P;
        order.R = IMUPreintegration::Order::R;
        order.V = IMUPreintegration::Order::V;
        order.BA = IMUPreintegration::Order::Ba;
        order.BG = IMUPreintegration::Order::Bg;
        
        std::shared_ptr<IMUFactor> imuFactor(new IMUFactor(
            cameras, vbs, frame_j->ID - this->frameManager->frames.front()->ID, linear, jacobians,
            this->targetGravity.cast<Scalar>(),
            static_cast<Scalar>(imu->GetSumTime()),
            order, imu->GetCovariance().cast<Scalar>()));
        problem.AddIMUFactor(imuFactor);
        ++cnt;
    }
    std::cout << "<Marginalizor> Add " << problem.GetIMUBlocksNum() << " imu factor blocks." << std::endl;

    // 为 VIO 问题添加先验信息
    if (this->prior.J.cols() > 0) {
        problem.SetPrior(this->prior.J.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Marginalizor> Add prior factors with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
            this->prior.r.squaredNorm() << std::endl;
    }

    // 配置参数，执行边缘化最旧帧操作
    rootVIO::ProblemVIO<Scalar>::Options options;
    options.minPriorItem = 1e-10;
    problem.SetOptions(options);
    problem.Marginalize(0, sizeof_prior - 15);

    // 提取边缘化结果
    rootVIO::MatrixX<Scalar> prior_J;
    rootVIO::VectorX<Scalar> prior_r;
    problem.GetPrior(prior_J, prior_r);
    this->prior.J = prior_J.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Marginalizor> Make prior with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;

    return true;
}


/* 边缘化次新帧，构造先验信息 */
bool VIOMono::MarginalizeSubnewFrame_rootvio(void) {
    using Scalar = double;
    using ExPoseType = rootVIO::VertexExPose<Scalar>;
    using PoseType = rootVIO::VertexCameraPose<Scalar>;
    using LandmarkType = rootVIO::VertexLandmark<Scalar>;
    using MotionType = rootVIO::VertexVelocityBias<Scalar>;
    using KernelType = rootVIO::HuberKernel<Scalar>;
    using CameraFactor = rootVIO::CameraObserve<Scalar>;
    using IMUFactor = rootVIO::IMUBlock<Scalar>;
    rootVIO::ProblemVIO<Scalar> problem;
    size_t sizeof_prior = 0;

    // 为 VIO 问题添加相机外参节点
    std::shared_ptr<ExPoseType> exPoseVertex(new ExPoseType(this->q_bc.cast<Scalar>(), this->t_bc.cast<Scalar>()));
    problem.AddExPose(exPoseVertex);
    sizeof_prior += 6;
    std::cout << "<Marginalizor> Add 1 ex pose vertex." << std::endl;

    // 为 VIO 问题添加 camera pose 节点和 velocity bias 节点
    std::unordered_map<size_t, std::shared_ptr<PoseType>> cameraVertices;
    std::unordered_map<size_t, std::shared_ptr<MotionType>> motionVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        std::shared_ptr<PoseType> cameraVertex(new PoseType((*it)->q_wb.cast<Scalar>(),
                                                            (*it)->t_wb.cast<Scalar>()));
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        problem.AddCamera(cameraVertex);
        sizeof_prior += 6;

        std::shared_ptr<MotionType> motionVertex(new MotionType((*it)->v_wb.cast<Scalar>(),
                                                                (*it)->imu->GetBiasA().cast<Scalar>(),
                                                                (*it)->imu->GetBiasG().cast<Scalar>()));
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
        problem.AddVelocityBias(motionVertex);
        sizeof_prior += 9;
    }
    std::cout << "<Marginalizor> Add " << problem.GetCamerasNum() << " camera pose vertices and " << problem.GetVelocityBiasesNum() <<
        " velocity bias vertices." << std::endl;

    // 为 VIO 问题添加先验信息
    if (this->prior.J.cols() > 0) {
        problem.SetPrior(this->prior.J.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Marginalizor> Add prior factors with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
            this->prior.r.squaredNorm() << std::endl;
    }

    // 配置参数，执行边缘化最旧帧操作
    rootVIO::ProblemVIO<Scalar>::Options options;
    options.minPriorItem = 1e-10;
    problem.SetOptions(options);
    problem.Marginalize(0, sizeof_prior - 15);

    // 提取边缘化结果
    rootVIO::MatrixX<Scalar> prior_J;
    rootVIO::VectorX<Scalar> prior_r;
    problem.GetPrior(prior_J, prior_r);
    this->prior.J = prior_J.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Marginalizor> Make prior with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;

    return true;
}