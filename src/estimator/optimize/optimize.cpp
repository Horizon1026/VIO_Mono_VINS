#include <include/estimator/estimator.hpp>
#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <include/ba_solver/root_vio/problem_vio.hpp>
#include <include/estimator/vertices/vertex_pose.hpp>
#include <include/estimator/vertices/vertex_landmark_pos.hpp>
#include <include/estimator/vertices/vertex_landmark_invdep.hpp>
#include <include/estimator/vertices/vertex_velocity_bias.hpp>
#include <include/estimator/edges/edge_camera_factor.hpp>
#include <include/estimator/edges/edge_imu_factor.hpp>

/* 为后端求解器初始化过程配置参数 */
bool VIOMono::UpdateConfigParams_Estimator(void) {
    if (this->config == nullptr) {
        return false;
    }
    this->threshold_minCovisibleNum = this->config->optimizeParams.minCovisibleNum;
    this->threshold_minMeanParallax = this->config->optimizeParams.minMeanParallax;
    this->threshold_maxImuSumTime = this->config->optimizeParams.maxImuSumTime;
    switch (this->config->solverType) {
        case 0:
            this->solverType = SolverType::GraphOptimizor;
            break;
        case 1:
            this->solverType = SolverType::RootVIO;
            break;
        default:
            this->solverType = SolverType::GraphOptimizor;
            break;
    }
    return true;
}


/* 后端求解器迭代优化，入口函数 */
bool VIOMono::EstimatorOptimize(void) {
    std::cout << "\n<Estimator> Start optimization." << std::endl;
    // 第一步：计算最新帧和新特征值的初值
    bool res = true;
    res = this->ComputeNewsInitValue();
    if (res == false) {
        return false;
    }
    // 第二步：构造整个滑动窗口的优化问题，迭代优化所有参数
    GraphOptimizor::Timer timer;
    switch (this->solverType) {
        default:
        case SolverType::GraphOptimizor:
            res = this->EstimateAllInSlidingWindow_go();
            break;
        case SolverType::RootVIO:
            res = this->EstimateAllInSlidingWindow_rootvio();
            break;
    }
    this->costTimes.emplace_back(timer.Stop());
    if (res == false) {
        return false;
    }
    // 第三步：决定边缘化策略
    res = this->DetermineMarginalizeStrategy();
    if (res == false) {
        return false;
    }
    // 第四步：采取对应的边缘化策略，构造先验信息
    switch (this->solverType) {
        default:
        case SolverType::GraphOptimizor:
            res = this->Marginalize_go();
            break;
        case SolverType::RootVIO:
            res = this->Marginalize_rootvio();
            break;
    }
    std::cout << "<Estimator> Optimization succeed." << std::endl;
    return res;
}


/* 第一步：计算最新帧和新特征点的初值 */
bool VIOMono::ComputeNewsInitValue(void) {
    // 定位到最新一帧和上一帧
    auto it = this->frameManager->frames.rbegin();
    auto newest = *it;
    if (std::next(it) == this->frameManager->frames.rend()) {
        return false;
    }
    auto subnew = *std::next(it);
    // 利用最新一帧绑定的 IMU 预积分值来调整初值
    Eigen::Quaternionf delta_r;
    Eigen::Vector3f delta_v, delta_p;
    newest->imu->GetDeltaRVP(delta_r, delta_v, delta_p);
    float dt = newest->imu->GetSumTime();
    // 此处提取的位于 subnew 和 newest 之间的预积分值的参考，是 subnew 的 b 系
    newest->q_wb = subnew->q_wb * delta_r;
    newest->t_wb = subnew->q_wb.toRotationMatrix() * delta_p + subnew->t_wb + subnew->v_wb * dt - 0.5f * this->targetGravity * dt * dt;
    newest->v_wb = subnew->q_wb.toRotationMatrix() * delta_v + subnew->v_wb - this->targetGravity * dt;
    newest->q_wc = newest->q_wb * this->q_bc;
    newest->t_wc = newest->q_wb * this->t_bc + newest->t_wb;
    newest->v_wc = this->q_bc.inverse() * newest->q_wb.inverse() * newest->v_wb;
    // T_wb = T_wc * T_bc.inverse();
    // [R_wb   t_wb] = [R_wc   t_wc] * [R_bc'   - R_bc' * t_bc]
    // [ 0      1  ]   [ 0       1 ]   [ 0              1     ]
    //               = [R_wc * R_bc'   R_wc * (- R_bc' * t_bc) + t_wc]
    //                 [     0                       1               ]
    //               = [R_wc * R_bc'   - R_wb * t_bc + t_wc]
    //                 [     0                   1         ]
    // T_wc = T_wb * T_bc
    // [R_wc   t_wc] = [R_wb   t_wb] * [R_bc  t_bc]
    // [ 0      1  ]   [ 0       1 ]   [ 0      1 ]
    //               = [R_wb * R_bc  R_wb * t_bc + t_wb]
    //                 [     0               1         ]
    // 基于所有图像帧已知的位姿，三角测量未被三角化的特征点
    // 同时利用其在世界坐标系中的位置，来更新其逆深度
    size_t cnt = 0;
    for (auto &item : this->landmarkManager->landmarks) {
        auto &lm_ptr = item.second;
        if (lm_ptr->isSolved != Landmark::SolveStatus::YES && lm_ptr->observes.size() > 1) {
            bool res = this->TriangulateLandmark(lm_ptr);
            if (res == true) {
                lm_ptr->isSolved = Landmark::SolveStatus::YES;
                ++cnt;
            } else {
                lm_ptr->isSolved = Landmark::SolveStatus::NO;
            }
        }
        if (lm_ptr->observes.size() > 1) {
            this->UsePosToUpdateInvdep(lm_ptr);
        }
    }
    std::cout << "<Estimator> Triangulate " << cnt << " new landmarks with all observations." << std::endl;
    return true;
}


/* 第二步：构造整个滑动窗口的优化问题，迭代优化所有参数 */
bool VIOMono::EstimateAllInSlidingWindow_go(void) {
    using Scalar = double;
    using ParamType = GraphOptimizor::VectorX<Scalar>;
    using PoseType = GraphOptimizor::VertexPose<Scalar>;
    using LandmarkType = GraphOptimizor::VertexLandmarkInvDepth<Scalar>;
    using MotionType = GraphOptimizor::VertexVelocityBias<Scalar>;
    using CameraFactor = GraphOptimizor::EdgeCameraFactor<Scalar>;
    using ImuFactor = GraphOptimizor::EdgeIMUFactor<Scalar>;
    using KernelType = GraphOptimizor::HuberKernel<Scalar>;

    // 记录首帧相机位姿
    Eigen::Quaternionf q_wb0 = this->frameManager->frames.front()->q_wb;
    Eigen::Vector3f t_wb0 = this->frameManager->frames.front()->t_wb;
    Eigen::Vector3f v_wb0 = this->frameManager->frames.front()->v_wb;

    // 构造图优化问题
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_ex = 0;
    size_t type_camera = 1;
    size_t type_landmark = 2;
    size_t type_motion = 3;

    // 添加 IMU 与相机的外参数节点
    std::shared_ptr<PoseType> exPoseVertex(new PoseType());
    for (size_t i = 0; i < 1; ++i) {
        ParamType param(7);
        param << this->t_bc.cast<Scalar>(), this->q_bc.x(), this->q_bc.y(), this->q_bc.z(), this->q_bc.w();
        exPoseVertex->SetParameters(param);
        exPoseVertex->SetType(type_ex);
        problem.AddVertex(exPoseVertex);
    }
    exPoseVertex->SetFixed();
    std::cout << "<Problem> Add 1 ex pose vertex and fixed." << std::endl;

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

        ParamType paramMotion(9);
        paramMotion << (*it)->v_wb.cast<Scalar>(), (*it)->imu->GetBiasA().cast<Scalar>(), (*it)->imu->GetBiasG().cast<Scalar>();
        std::shared_ptr<MotionType> motionVertex(new MotionType());
        motionVertex->SetParameters(paramMotion);
        motionVertex->SetType(type_motion);
        problem.AddVertex(motionVertex);
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
    }
    std::cout << "<Problem> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera and motion vertices." << std::endl;

    // 添加特征点逆深度节点，同时添加视觉残差边
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
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
    std::cout << "<Problem> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark vertices." << std::endl;
    std::cout << "<Problem> Add " << cnt << " camera residual factors." << std::endl;

    // 添加 IMU 残差边
    cnt = 0;
    for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
        auto &frame_i = *it;
        auto &frame_j = *std::next(it);
        auto imu = frame_j->imu;
        // 忽略积分时间过长的 IMU 预积分块
        if (imu->GetSumTime() > this->threshold_maxImuSumTime) {
            continue;
        }
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
        ++cnt;
    }
    std::cout << "<Problem> Add " << cnt << " imu residual factors." << std::endl;

    // 添加先验信息
    if (this->prior.H.rows() > 0) {
        problem.SetPrior(this->prior.H.cast<Scalar>(), this->prior.b.cast<Scalar>(), this->prior.JTinv.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Problem> Add prior factors with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
            this->prior.r.squaredNorm() << std::endl;
    }

    // 配置求解器，迭代优化求解
    problem.SetMargnedVertexTypesWhenSolving(type_landmark);
    problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::LM_Auto);
    problem.SetLinearSolver(GraphOptimizor::Problem<Scalar>::LinearSolver::PCG_Solver);
    GraphOptimizor::Problem<Scalar>::Options options;
    options.maxInvalidStep = 3;
    options.maxMinCostHold = 3;
    options.minCostDownRate = 1e-5;
    options.minNormDeltaX = 1e-5;
    options.minPCGCostDownRate = 1e-6;
    problem.SetOptions(options);
    problem.Solve(40);

    // 更新先验信息（优化过程中，先验信息会改变）
    GraphOptimizor::MatrixX<Scalar> prior_H, prior_JTinv;
    GraphOptimizor::VectorX<Scalar> prior_b, prior_r;
    problem.GetPrior(prior_H, prior_b, prior_JTinv, prior_r);
    this->prior.H = prior_H.cast<double>();
    this->prior.JTinv = prior_JTinv.cast<double>();
    this->prior.b = prior_b.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Problem> Update prior factors with size " << this->prior.H.rows() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;

    // 更新相机与 IMU 的相对位姿
    auto paramExPose = exPoseVertex->GetParameters();
    this->t_bc = paramExPose.head<3>().cast<float>();
    this->q_bc = Eigen::Quaternionf(paramExPose[6], paramExPose[3], paramExPose[4], paramExPose[5]);

    // 更新每一个关键帧的位姿，以及其对应的 IMU 预积分块的偏差，并对 IMU 预积分块进行重新积分
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto paramCamera = cameraVertices.find((*it)->ID)->second->GetParameters();
        auto paramMotion = motionVertices.find((*it)->ID)->second->GetParameters();
        // paramCamera << (*it)->t_wb, (*it)->q_wb.x(), (*it)->q_wb.y(), (*it)->q_wb.z(), (*it)->q_wb.w();
        (*it)->t_wb = paramCamera.head<3>().cast<float>();
        (*it)->q_wb = Eigen::Quaternionf(paramCamera[6], paramCamera[3], paramCamera[4], paramCamera[5]);
        // paramMotion << (*it)->v_wb, (*it)->imu->GetBiasA(), (*it)->imu->GetBiasG();
        (*it)->v_wb = paramMotion.head<3>().cast<float>();

        // bias 可能会发散，待解决 TODO
        (*it)->imu->SetBiasA(paramMotion.segment<3>(3).cast<float>());
        (*it)->imu->SetBiasG(paramMotion.tail<3>().cast<float>());
        (*it)->imu->Repropagate();
    }

    // 计算轨迹整体偏移值，对所有帧的位姿和速度进行调整
    // 从 q_wb, t_wb 到 q_wb0, t_wb0 的过程，等于从 T_wb 到 T_wb0 的过程，则 T_wb * ? = T_wb0, ? = T_bb0 = T_wb.inv * T_wb0
    /*  [R_bb0  t_bb0] = [R_wb   t_wb]-1  *  [R_wb0   t_wb0]
        [  0      1  ]   [  0      1 ]       [  0       1  ]
                       = [R_wb'  -R_wb' * t_wb]  *  [R_wb0  t_wb0]
                         [  0           1     ]     [  0      1  ]
                       = [R_wb' * R_wb0   R_wb' * t_wb0 - R_wb' * t_wb]
                         [      0                       1             ] */
    /*  [R_wb0  t_wb0] = [R_wb  t_wb] * [R_bb0  t_bb0]
        [  0       1 ]   [  0     1 ]   [  0      1  ]
                       = [R_wb * R_bb0  R_wb * t_bb0 + t_wb]
                         [     0                 1         ] */
    Eigen::Quaternionf q_diff = q_wb0 * this->frameManager->frames.front()->q_wb.inverse();
    Eigen::Vector3f t_diff = t_wb0;
    Eigen::Vector3f t_wb_ = this->frameManager->frames.front()->t_wb;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        (*it)->q_wb = q_diff * (*it)->q_wb;
        (*it)->t_wb = q_diff * ((*it)->t_wb - t_wb_) + t_diff;
        (*it)->v_wb = q_diff * (*it)->v_wb;
        // T_wc = T_wb * T_bc
        // [R_wc   t_wc] = [R_wb   t_wb] * [R_bc  t_bc]
        // [ 0      1  ]   [ 0       1 ]   [ 0      1 ]
        //               = [R_wb * R_bc  R_wb * t_bc + t_wb]
        //                 [     0               1         ]
        (*it)->v_wc = (*it)->q_wc * this->q_bc.inverse() * (*it)->q_wb.inverse() * (*it)->v_wb;
        (*it)->q_wc = (*it)->q_wb * this->q_bc;
        (*it)->t_wc = (*it)->q_wb * this->t_bc + (*it)->t_wb;
    }

    // 更新每一个特征点
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        auto params = item.second->GetParameters();
        lm_ptr->invDep = params.cast<float>()[0];
        if (lm_ptr->invDep <= 0 || lm_ptr->invDep > 50.0) {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        }
        this->UseInvdepToUpdatePos(lm_ptr);
    }

    return true;
}


/* 第二步：构造整个滑动窗口的优化问题，迭代优化所有参数 */
bool VIOMono::EstimateAllInSlidingWindow_rootvio(void) {
    using Scalar = double;
    using ExPoseType = rootVIO::VertexExPose<Scalar>;
    using PoseType = rootVIO::VertexCameraPose<Scalar>;
    using LandmarkType = rootVIO::VertexLandmark<Scalar>;
    using MotionType = rootVIO::VertexVelocityBias<Scalar>;
    using KernelType = rootVIO::HuberKernel<Scalar>;
    using CameraFactor = rootVIO::CameraObserve<Scalar>;
    using IMUFactor = rootVIO::IMUBlock<Scalar>;

    // 记录首帧相机位姿
    Eigen::Quaternionf q_wb0 = this->frameManager->frames.front()->q_wb;
    Eigen::Vector3f t_wb0 = this->frameManager->frames.front()->t_wb;
    Eigen::Vector3f v_wb0 = this->frameManager->frames.front()->v_wb;

    // 构造 root VIO 问题
    rootVIO::ProblemVIO<Scalar> problem;

    // 为 VIO 问题添加相机外参节点,并设置为固定
    std::shared_ptr<ExPoseType> exPoseVertex(new ExPoseType(this->q_bc.cast<Scalar>(), this->t_bc.cast<Scalar>()));
    problem.AddExPose(exPoseVertex);
    exPoseVertex->SetFixed();
    std::cout << "<Problem> Add 1 ex pose vertex and fixed." << std::endl;

    // 为 VIO 问题添加 camera pose 节点和 velocity bias 节点
    std::unordered_map<size_t, std::shared_ptr<PoseType>> cameraVertices;
    std::unordered_map<size_t, std::shared_ptr<MotionType>> motionVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        std::shared_ptr<PoseType> cameraVertex(new PoseType((*it)->q_wb.cast<Scalar>(),
                                                            (*it)->t_wb.cast<Scalar>()));
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        problem.AddCamera(cameraVertex);

        std::shared_ptr<MotionType> motionVertex(new MotionType((*it)->v_wb.cast<Scalar>(),
                                                                (*it)->imu->GetBiasA().cast<Scalar>(),
                                                                (*it)->imu->GetBiasG().cast<Scalar>()));
        motionVertices.insert(std::make_pair((*it)->ID, motionVertex));
        problem.AddVelocityBias(motionVertex);
    }
    std::cout << "<Problem> Add " << problem.GetCamerasNum() << " camera pose vertices and " << problem.GetVelocityBiasesNum() <<
        " velocity bias vertices." << std::endl;

    // 为 VIO 问题添加特征点逆深度节点，并对应添加视觉约束
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
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
    std::cout << "<Problem> Add " << problem.GetLandmarksNum() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark blocks." << std::endl;
    std::cout << "<Problem> Add " << cnt << " camera residual factors." << std::endl;

    // 为 VIO 问题添加 IMU 观测约束
    cnt = 0;
    for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
        auto &frame_i = *it;
        auto &frame_j = *std::next(it);
        auto imu = frame_j->imu;
        // 忽略积分时间过长的 IMU 预积分块
        if (imu->GetSumTime() > this->threshold_maxImuSumTime) {
            continue;
        }
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
    std::cout << "<Problem> Add " << problem.GetIMUBlocksNum() << " imu factor blocks." << std::endl;

    // 为 VIO 问题添加先验信息
    if (this->prior.J.cols() > 0) {
        problem.SetPrior(this->prior.J.cast<Scalar>(), this->prior.r.cast<Scalar>());
        std::cout << "<Problem> Add prior factors with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
            this->prior.r.squaredNorm() << std::endl;
    }

    // 配置参数，求解 VIO 问题
    problem.SetDampPolicy(rootVIO::ProblemVIO<Scalar>::DampPolicy::Auto);
    problem.SetLinearSolver(rootVIO::ProblemVIO<Scalar>::LinearSolver::PCG_Solver);
    rootVIO::ProblemVIO<Scalar>::Options options;
    options.maxInvalidStep = 5;
    options.maxMinCostHold = 5;
    options.minCostDownRate = 1e-5;
    options.minNormDeltaX = 1e-5;
    options.minPCGCostDownRate = 1e-6;
    options.maxTimeCost = 100;
    options.initLambda = 1e-6;
    problem.SetOptions(options);
    problem.Solve(60);

    // 更新先验信息（优化过程中，先验信息会改变）
    rootVIO::MatrixX<Scalar> prior_J;
    rootVIO::VectorX<Scalar> prior_r;
    problem.GetPrior(prior_J, prior_r);
    this->prior.J = prior_J.cast<double>();
    this->prior.r = prior_r.cast<double>();
    std::cout << "<Problem> Update prior factors with size " << this->prior.J.cols() << " and r.squaredNorm() = " <<
        this->prior.r.squaredNorm() << std::endl;

    // 更新相机与 IMU 的相对位姿
    this->t_bc = exPoseVertex->Get_t_bc().cast<float>();
    this->q_bc = exPoseVertex->Get_q_bc().cast<float>();

    // 更新每一个关键帧的位姿，以及其对应的 IMU 预积分块的偏差，并对 IMU 预积分块进行重新积分
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto cameraPose = cameraVertices.find((*it)->ID)->second;
        auto velocityBias = motionVertices.find((*it)->ID)->second;
        (*it)->t_wb = cameraPose->Get_t_wb().cast<float>();
        (*it)->q_wb = cameraPose->Get_q_wb().cast<float>();
        (*it)->v_wb = velocityBias->Get_v_wb().cast<float>();

        // bias 可能会发散，待解决 TODO
        (*it)->imu->SetBiasA(velocityBias->Get_bias_a().cast<float>());
        (*it)->imu->SetBiasG(velocityBias->Get_bias_g().cast<float>());
        (*it)->imu->Repropagate();
    }

    // 计算轨迹整体偏移值，对所有帧的位姿和速度进行调整
    // 从 q_wb, t_wb 到 q_wb0, t_wb0 的过程，等于从 T_wb 到 T_wb0 的过程，则 T_wb * ? = T_wb0, ? = T_bb0 = T_wb.inv * T_wb0
    /*  [R_bb0  t_bb0] = [R_wb   t_wb]-1  *  [R_wb0   t_wb0]
        [  0      1  ]   [  0      1 ]       [  0       1  ]
                       = [R_wb'  -R_wb' * t_wb]  *  [R_wb0  t_wb0]
                         [  0           1     ]     [  0      1  ]
                       = [R_wb' * R_wb0   R_wb' * t_wb0 - R_wb' * t_wb]
                         [      0                       1             ] */
    /*  [R_wb0  t_wb0] = [R_wb  t_wb] * [R_bb0  t_bb0]
        [  0       1 ]   [  0     1 ]   [  0      1  ]
                       = [R_wb * R_bb0  R_wb * t_bb0 + t_wb]
                         [     0                 1         ] */
    Eigen::Quaternionf q_diff = q_wb0 * this->frameManager->frames.front()->q_wb.inverse();
    Eigen::Vector3f t_diff = t_wb0;
    Eigen::Vector3f t_wb_ = this->frameManager->frames.front()->t_wb;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        (*it)->q_wb = q_diff * (*it)->q_wb;
        (*it)->t_wb = q_diff * ((*it)->t_wb - t_wb_) + t_diff;
        (*it)->v_wb = q_diff * (*it)->v_wb;
        // T_wc = T_wb * T_bc
        // [R_wc   t_wc] = [R_wb   t_wb] * [R_bc  t_bc]
        // [ 0      1  ]   [ 0       1 ]   [ 0      1 ]
        //               = [R_wb * R_bc  R_wb * t_bc + t_wb]
        //                 [     0               1         ]
        (*it)->v_wc = (*it)->q_wc * this->q_bc.inverse() * (*it)->q_wb.inverse() * (*it)->v_wb;
        (*it)->q_wc = (*it)->q_wb * this->q_bc;
        (*it)->t_wc = (*it)->q_wb * this->t_bc + (*it)->t_wb;
    }

    // 更新每一个特征点
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        lm_ptr->invDep = item.second->Get_invdep();
        if (lm_ptr->invDep <= 0 || lm_ptr->invDep > 50.0) {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        }
        this->UseInvdepToUpdatePos(lm_ptr);
    }

    return true;
}