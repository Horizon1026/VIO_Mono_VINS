#include <include/estimator/estimator.hpp>
#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <include/ba_solver/root_vio/problem_vo.hpp>
#include <include/estimator/vertices/vertex_pose.hpp>
#include <include/estimator/vertices/vertex_landmark_pos.hpp>
#include <include/estimator/vertices/vertex_landmark_invdep.hpp>
#include <include/estimator/edges/edge_reprojection_pos.hpp>
#include <include/estimator/edges/edge_reprojection_invdep.hpp>


/* 第二步：计算滑动窗口内所有帧的位姿和所有特征点的位置的初值 */
bool VIOMono::ComputeVisualSFMInitValue(void) {
    auto &allFrames = this->frameManager->frames;
    // 将第一帧设置为原点
    std::shared_ptr<CombinedFrame> originFrame = allFrames.front();
    originFrame->q_wc.setIdentity();
    originFrame->t_wc.setZero();
    // 寻找与第一帧有着较好约束关系的一帧
    this->bestCorrespond = std::make_pair(originFrame->ID, originFrame);
    for (auto it = std::next(allFrames.begin()); it != allFrames.end(); ++it) {
        auto correspond = originFrame->GetCorrespondence(*it);
        std::cout << "<Estimator> Detect correspondence between origin and frame " << (*it)->ID << ", [parallax, nums] = [" <<
            correspond.first << " / " << this->threshold_initMinMeanParallax << ", " << correspond.second << " / " <<
            this->threshold_initMinCovisibleNum << "]" << std::endl;
        // correspond = < mean parallax, covisible features num >
        if (correspond.first > this->threshold_initMinMeanParallax &&
            correspond.second > this->threshold_initMinCovisibleNum) {
            this->bestCorrespond.first = (*it)->ID;
            this->bestCorrespond.second = *it;
            break;
        }
    }
    if (this->bestCorrespond.first == originFrame->ID) {
        std::cout << "<Estimator> Cannot find correspond frame with origin frame." << std::endl;
        return false;
    }
    // 在原点帧 originFrame 和最好约束帧 this->bestCorrespond 之间估计相对位姿，并三角测量他们之间的共视特征点
    Eigen::Quaternionf q_c1c2;
    Eigen::Vector3f t_c1c2;
    bool res = this->EstimateRelativePose(originFrame, this->bestCorrespond.second, q_c1c2, t_c1c2);
    this->bestCorrespond.second->q_wc = q_c1c2;
    this->bestCorrespond.second->t_wc = t_c1c2;
    std::cout << "<Estimator> Relative pose between frame " << originFrame->ID << " and " << this->bestCorrespond.first << " is [" <<
        q_c1c2.w() << ", " << q_c1c2.x() << ", " << q_c1c2.y() << ", " << q_c1c2.z() << "], [" << t_c1c2.transpose() << "]\n";
    if (res == false) {
        return false;
    }
    this->TriangulateLandmarks(originFrame, this->bestCorrespond.second);

    // 从前往后，遍历所有帧，采用 PnP 方法估计帧位姿，基于帧位姿三角测量特征点
    for (auto it = std::next(allFrames.begin()); it != allFrames.end(); ++it) {
        if ((*it)->ID != this->bestCorrespond.first) {
            // 通过 PnP 估计下一帧位姿时，以上一帧的位姿为初值
            (*it)->q_wc = (*std::prev(it))->q_wc;
            (*it)->t_wc = (*std::prev(it))->t_wc;
            res = this->EstimatePose(*it);
            if (res == false) {
                return false;
            }
        }
        this->TriangulateLandmarks(*(std::prev(it)), *it);
    }

    // 根据特征点在世界坐标系中的位置，更新其逆深度
    for (auto &item : this->landmarkManager->landmarks) {
        auto lm = item.second;
        if (lm->isSolved == Landmark::SolveStatus::YES) {
            this->UsePosToUpdateInvdep(lm);
        }
    }

    return true;
}


/* 第三步：优化滑动窗口内所有帧的位姿和所有特征点的位置 */
bool VIOMono::RefineVisualSFM_go_pos(void) {
    using Scalar = float;
    using ParamType = GraphOptimizor::VectorX<Scalar>;
    using CameraType = GraphOptimizor::VertexPose<Scalar>;
    using LandmarkType = GraphOptimizor::VertexLandmarkPosition<Scalar>;
    using CameraFactor = GraphOptimizor::EdgeReprojectionPos<Scalar>;
    using KernelType = GraphOptimizor::HuberKernel<Scalar>;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_camera = 0;
    size_t type_landmark = 1;
    // 为问题添加相机 Pose 节点
    std::unordered_map<size_t, std::shared_ptr<CameraType>> cameraVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        ParamType param(7);
        param << (*it)->t_wc, (*it)->q_wc.x(), (*it)->q_wc.y(), (*it)->q_wc.z(), (*it)->q_wc.w();
        std::shared_ptr<CameraType> cameraVertex(new CameraType());
        cameraVertex->SetParameters(param);
        cameraVertex->SetType(type_camera);
        problem.AddVertex(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        if (it == this->frameManager->frames.begin()) {
            cameraVertex->SetFixed();
        }
    }
    std::cout << "<Problem> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera vertices." << std::endl;
    // 为问题添加特征点 position 节点
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->observes.size() > 1) {
            ParamType param(3);
            param << lm_ptr->p_w;
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType());
            landmarkVertex->SetParameters(param);
            landmarkVertex->SetType(type_landmark);
            problem.AddVertex(landmarkVertex);
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));

            // 为问题添加重投影约束边
            for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
                GraphOptimizor::Vector2<Scalar> norm_i;
                norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i);
                auto cameraVertex = cameraVertices.find(lm_ptr->firstFrameID + i);
                std::shared_ptr<CameraFactor> edge(new CameraFactor(norm_i));
                edge->AddVertex(landmarkVertex, 0);
                edge->AddVertex(cameraVertex->second, 1);
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                edge->SetKernel(kernel);
                // edge->SetInformation(this->projectionInformation);
                edge->SetInformation(GraphOptimizor::Matrix2<Scalar>::Identity());
                problem.AddEdge(edge);
                ++cnt;
            }
        }
    }
    std::cout << "<Problem> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark vertices." << std::endl;
    std::cout << "<Problem> Add " << cnt << " reprojection edges." << std::endl;
    // 配置参数，求解问题
    problem.SetMargnedVertexTypesWhenSolving(type_landmark);
    problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::LM_Auto);
    problem.SetLinearSolver(GraphOptimizor::Problem<Scalar>::LinearSolver::PCG_Solver);
    GraphOptimizor::Problem<Scalar>::Options options;
    options.maxInvalidStep = 3;
    options.maxMinCostHold = 3;
    options.minCostDownRate = 1e-6;
    options.minNormDeltaX = 1e-6;
    options.maxTimeCost = 100;
    options.initLambda = 5e-4;
    problem.SetOptions(options);
    problem.Solve(40);
    // 返回结果
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto params = cameraVertices.find((*it)->ID)->second->GetParameters();
        // param << (*it)->t_wc, (*it)->q_wc.x(), (*it)->q_wc.y(), (*it)->q_wc.z(), (*it)->q_wc.w();
        (*it)->t_wc = params.head<3>().cast<float>();
        (*it)->q_wc = Eigen::Quaternionf(params[6], params[3], params[4], params[5]);
    }
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        auto params = item.second->GetParameters();
        lm_ptr->p_w = params.cast<float>();
        this->UsePosToUpdateInvdep(lm_ptr);
        if (lm_ptr->invDep > 0) {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        }
    }
    return true;
}


/* 第三步：优化滑动窗口内所有帧的位姿和所有特征点的位置 */
bool VIOMono::RefineVisualSFM_go_invdep(void) {
    using Scalar = float;
    using ParamType = GraphOptimizor::VectorX<Scalar>;
    using CameraType = GraphOptimizor::VertexPose<Scalar>;
    using LandmarkType = GraphOptimizor::VertexLandmarkInvDepth<Scalar>;
    using CameraFactor = GraphOptimizor::EdgeReprojectionInvdep<Scalar>;
    using KernelType = GraphOptimizor::HuberKernel<Scalar>;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_camera = 0;
    size_t type_landmark = 1;
    // 为问题添加相机 Pose 节点
    std::unordered_map<size_t, std::shared_ptr<CameraType>> cameraVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        ParamType param(7);
        param << (*it)->t_wc, (*it)->q_wc.x(), (*it)->q_wc.y(), (*it)->q_wc.z(), (*it)->q_wc.w();
        std::shared_ptr<CameraType> cameraVertex(new CameraType());
        cameraVertex->SetParameters(param);
        cameraVertex->SetType(type_camera);
        problem.AddVertex(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        if (it == this->frameManager->frames.begin()) {
            cameraVertex->SetFixed();
        }
    }
    std::cout << "<Problem> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera vertices." << std::endl;
    // 为问题添加特征点 invdep 节点
    size_t cnt = 0;
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->observes.size() > 1) {
            ParamType param(1);
            param << lm_ptr->invDep;
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType());
            landmarkVertex->SetParameters(param);
            landmarkVertex->SetType(type_landmark);
            problem.AddVertex(landmarkVertex);
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));

            // 为问题添加重投影约束边
            for (size_t i = 1; i < lm_ptr->observes.size(); ++i) {
                GraphOptimizor::Vector2<Scalar> norm_0, norm_i;
                norm_0 = lm_ptr->GetNorm(lm_ptr->firstFrameID);
                norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i);
                auto cam_0 = cameraVertices.find(lm_ptr->firstFrameID);
                auto cam_i = cameraVertices.find(lm_ptr->firstFrameID + i);
                std::shared_ptr<CameraFactor> edge(new CameraFactor(norm_0, norm_i));
                edge->AddVertex(landmarkVertex, 0);
                edge->AddVertex(cam_0->second, 1);
                edge->AddVertex(cam_i->second, 2);
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                edge->SetKernel(kernel);
                edge->SetInformation(this->projectionInformation);
                problem.AddEdge(edge);
                ++cnt;
            }
        }
    }
    std::cout << "<Problem> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark vertices." << std::endl;
    std::cout << "<Problem> Add " << cnt << " reprojection edges." << std::endl;
    // 配置参数，求解问题
    problem.SetMargnedVertexTypesWhenSolving(type_landmark);
    problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::LM_Auto);
    problem.SetLinearSolver(GraphOptimizor::Problem<Scalar>::LinearSolver::PCG_Solver);
    GraphOptimizor::Problem<Scalar>::Options options;
    options.maxInvalidStep = 3;
    options.maxMinCostHold = 3;
    options.minCostDownRate = 1e-6;
    options.minNormDeltaX = 1e-6;
    options.minPCGCostDownRate = 1e-6;
    options.maxTimeCost = 100;
    options.initLambda = 1e-4;
    problem.SetOptions(options);
    problem.Solve(40);
    // 返回结果
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto params = cameraVertices.find((*it)->ID)->second->GetParameters();
        // param << (*it)->t_wc, (*it)->q_wc.x(), (*it)->q_wc.y(), (*it)->q_wc.z(), (*it)->q_wc.w();
        (*it)->t_wc = params.head<3>().cast<float>();
        (*it)->q_wc = Eigen::Quaternionf(params[6], params[3], params[4], params[5]);
    }
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        auto params = item.second->GetParameters();
        lm_ptr->invDep = params.cast<float>()[0];
        this->UseInvdepToUpdatePos(lm_ptr);
        if (lm_ptr->invDep > 0) {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        }
    }
    return true;
}


/* 第三步：优化滑动窗口内所有帧的位姿和所有特征点的位置 */
bool VIOMono::RefineVisualSFM_rootvio_pos(void) {
    using Scalar = float;
    using ExPoseType = rootVIO::VertexExPose<Scalar>;
    using CameraType = rootVIO::VertexCameraPose<Scalar>;
    using LandmarkType = rootVIO::VertexLandmark<Scalar>;
    using ObserveType = rootVIO::CameraObserve<Scalar>;
    using KernelType = rootVIO::HuberKernel<Scalar>;
    rootVIO::ProblemVO<Scalar> problem;
    problem.Reset();

    // 为问题添加相机与 IMU 外参节点（将 T_wb 等价为 T_wc）
    rootVIO::Quaternion<Scalar> q;
    rootVIO::Vector3<Scalar> t;
    q.setIdentity();
    t.setZero();
    std::shared_ptr<ExPoseType> exPoseVertex(new ExPoseType(q, t));
    exPoseVertex->SetFixed(true);
    problem.AddExPose(exPoseVertex);
    std::cout << "<Problem> Add 1 ex pose vertex." << std::endl;

    // 为问题添加相机 Pose 节点，输入到 problem 中的 camera 节点的 ID 必须从 0 开始
    std::unordered_map<size_t, std::shared_ptr<CameraType>> cameraVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        std::shared_ptr<CameraType> cameraVertex(new CameraType((*it)->q_wc.cast<Scalar>(), (*it)->t_wc.cast<Scalar>()));
        problem.AddCamera(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        if (it == this->frameManager->frames.begin()) {
            cameraVertex->SetFixed(true);
        }
    }
    std::cout << "<Problem> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera vertices." << std::endl;

    // 为问题添加特征点 position 节点，特征点对应观测帧的 ID 必须是在 0-N 之间，而不是帧管理器中的 ID
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->observes.size() > 1) {
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType(lm_ptr->p_w.cast<Scalar>()));
            std::unordered_map<size_t, std::shared_ptr<ObserveType>> observes;
            for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
                rootVIO::Vector2<Scalar> norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i).cast<Scalar>();
                auto cam_i = (*cameraVertices.find(lm_ptr->firstFrameID + i)).second;
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                std::shared_ptr<ObserveType> observe(new ObserveType(cam_i, norm_i, this->projectionInformation.cast<Scalar>(), kernel));
                observes.insert(std::make_pair(lm_ptr->firstFrameID + i - this->frameManager->frames.front()->ID, observe));
            }
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));
            problem.AddLandmark(landmarkVertex, observes, 3);
        }
    }
    std::cout << "<Problem> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark blocks." << std::endl;

    // 配置参数，求解问题
    problem.SetDampPolicy(rootVIO::ProblemVO<Scalar>::DampPolicy::Auto);
    problem.SetLinearSolver(rootVIO::ProblemVO<Scalar>::LinearSolver::PCG_Solver);
    rootVIO::ProblemVO<Scalar>::Options options;
    options.maxInvalidStep = 8;
    options.maxMinCostHold = 6;
    options.minCostDownRate = 1e-6;
    options.minNormDeltaX = 1e-6;
    options.minPCGCostDownRate = 1e-6;
    options.maxTimeCost = 100;
    options.initLambda = 1e-3;
    problem.SetOptions(options);
    problem.Solve(50);

    // 返回结果
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto vertex = cameraVertices.find((*it)->ID)->second;
        (*it)->t_wc = vertex->Get_t_wb().cast<float>();
        (*it)->q_wc = vertex->Get_q_wb().cast<float>();
    }
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        auto params = item.second->Get_p_w();
        lm_ptr->p_w = params.cast<float>();
        this->UsePosToUpdateInvdep(lm_ptr);
        if (lm_ptr->invDep > 0) {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        }
    }
    return true;
}


/* 第三步：优化滑动窗口内所有帧的位姿和所有特征点的逆深度 */
bool VIOMono::RefineVisualSFM_rootvio_invdep(void) {
    using Scalar = float;
    using ExPoseType = rootVIO::VertexExPose<Scalar>;
    using CameraType = rootVIO::VertexCameraPose<Scalar>;
    using LandmarkType = rootVIO::VertexLandmark<Scalar>;
    using ObserveType = rootVIO::CameraObserve<Scalar>;
    using KernelType = rootVIO::HuberKernel<Scalar>;
    rootVIO::ProblemVO<Scalar> problem;
    problem.Reset();

    // 为问题添加相机与 IMU 外参节点（将 T_wb 等价为 T_wc）
    rootVIO::Quaternion<Scalar> q;
    rootVIO::Vector3<Scalar> t;
    q.setIdentity();
    t.setZero();
    std::shared_ptr<ExPoseType> exPoseVertex(new ExPoseType(q, t));
    exPoseVertex->SetFixed(true);
    problem.AddExPose(exPoseVertex);
    std::cout << "<Problem> Add 1 ex pose vertex." << std::endl;

    // 为问题添加相机 Pose 节点，输入到 problem 中的 camera 节点的 ID 必须从 0 开始
    std::unordered_map<size_t, std::shared_ptr<CameraType>> cameraVertices;
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        std::shared_ptr<CameraType> cameraVertex(new CameraType((*it)->q_wc.cast<Scalar>(), (*it)->t_wc.cast<Scalar>()));
        problem.AddCamera(cameraVertex);
        cameraVertices.insert(std::make_pair((*it)->ID, cameraVertex));
        if (it == this->frameManager->frames.begin()) {
            cameraVertex->SetFixed(true);
        }
    }
    std::cout << "<Problem> Add " << cameraVertices.size() << "/" << this->frameManager->frames.size() <<
        " camera vertices." << std::endl;

    // 为问题添加特征点 position 节点，特征点对应观测帧的 ID 必须是在 0-N 之间，而不是帧管理器中的 ID
    std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<LandmarkType>> landmarkVertices;
    for (auto it = this->landmarkManager->landmarks.begin(); it != this->landmarkManager->landmarks.end(); ++it) {
        auto &lm_ptr = (*it).second;
        if (lm_ptr->observes.size() > 1) {
            std::shared_ptr<LandmarkType> landmarkVertex(new LandmarkType(Scalar(lm_ptr->invDep)));
            std::unordered_map<size_t, std::shared_ptr<ObserveType>> observes;
            for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
                rootVIO::Vector2<Scalar> norm_i = lm_ptr->GetNorm(lm_ptr->firstFrameID + i).cast<Scalar>();
                auto cam_i = (*cameraVertices.find(lm_ptr->firstFrameID + i)).second;
                std::shared_ptr<KernelType> kernel(new KernelType(0.5));
                std::shared_ptr<ObserveType> observe(new ObserveType(cam_i, norm_i, this->projectionInformation.cast<Scalar>(), kernel));
                observes.insert(std::make_pair(lm_ptr->firstFrameID + i - this->frameManager->frames.front()->ID, observe));
            }
            landmarkVertices.insert(std::make_pair(lm_ptr, landmarkVertex));
            problem.AddLandmark(landmarkVertex, observes, 1);
        }
    }
    std::cout << "<Problem> Add " << landmarkVertices.size() << "/" << this->landmarkManager->landmarks.size() <<
        " landmark blocks." << std::endl;

    // 配置参数，求解问题
    problem.SetDampPolicy(rootVIO::ProblemVO<Scalar>::DampPolicy::Auto);
    problem.SetLinearSolver(rootVIO::ProblemVO<Scalar>::LinearSolver::PCG_Solver);
    rootVIO::ProblemVO<Scalar>::Options options;
    options.maxInvalidStep = 4;
    options.maxMinCostHold = 4;
    options.minCostDownRate = 1e-6;
    options.minNormDeltaX = 1e-6;
    options.minPCGCostDownRate = 1e-6;
    options.maxTimeCost = 100;
    options.initLambda = 1e-6;
    problem.SetOptions(options);
    problem.Solve(100);

    // 返回结果
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        auto vertex = cameraVertices.find((*it)->ID)->second;
        (*it)->t_wc = vertex->Get_t_wb().cast<float>();
        (*it)->q_wc = vertex->Get_q_wb().cast<float>();
    }
    for (auto &item : landmarkVertices) {
        auto &lm_ptr = item.first;
        lm_ptr->invDep = static_cast<float>(item.second->Get_invdep());
        this->UseInvdepToUpdatePos(lm_ptr);
        if (lm_ptr->invDep > 0) {
            lm_ptr->isSolved = Landmark::SolveStatus::YES;
        } else {
            lm_ptr->isSolved = Landmark::SolveStatus::ERROR;
        }
    }
    return true;
}