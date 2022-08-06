#pragma once

#include <include/ba_solver/root_vio/landmark_block_pos.hpp>
#include <include/ba_solver/root_vio/landmark_block_invdep.hpp>
#include <include/ba_solver/root_vio/imu_block.hpp>
#include <include/ba_solver/root_vio/problem_base.hpp>
#include <tbb/tbb.h>

// 全局命名空间定义为 rootVIO
// 此处为模板类的声明
namespace rootVIO {

    // 采用 rootVIO 方法求解的待求解问题定义
    template<typename Scalar>
    class ProblemVIO : public ProblemBase<Scalar> {
        using LandmarkPtr = std::shared_ptr<LandmarkBlockBase<Scalar>>;
        using CameraPtr = std::shared_ptr<VertexCameraPose<Scalar>>;
        using ExPosePtr = std::shared_ptr<VertexExPose<Scalar>>;
        using VelBiasPtr = std::shared_ptr<VertexVelocityBias<Scalar>>;
        using IMUPtr = std::shared_ptr<IMUBlock<Scalar>>;

    /* 问题相关存储变量定义 */
    private:
        // 与此问题相关的所有 camera 节点，对应的 camera id 即 std::vector<> 的索引
        std::vector<CameraPtr> cameras;
        // 与此问题相关的所有 velocity bias 节点，对应的 id 和 this->cameras 完全一致
        std::vector<VelBiasPtr> velocityBiases;
        // 整个 VIO 的相机与 IMU 之间的相对位姿（默认为单位阵）
        ExPosePtr exPose;
        // 与此问题相关的所有 landmark block
        std::vector<LandmarkPtr> landmarkBlocks;
        std::vector<size_t> rowIndices;
        size_t landmarkSize = 1;
        // 与此问题相关的所有 imu block
        std::vector<IMUPtr> imuBlocks;

    /* 构造函数与析构函数定义 */
    public:
        /* 构造函数与析构函数 */
        ProblemVIO<Scalar>();
        ~ProblemVIO<Scalar>();

    /* 参数设置相关对外接口定义 */
    public:
        /* 重置待求解问题，实际行为是清空所有保存的信息 */
        bool Reset(void);

    /* 构造问题内容相关对外接口定义 */
    public:
        /* 为问题添加相机 pose 节点，分为添加一个节点和一次性添加多个节点 */
        bool AddCamera(const CameraPtr &newCamera);
        bool AddCameras(const std::vector<CameraPtr> &newCameras);
        /* 为问题添加 velocity bias 节点，分为添加一个节点和一次性添加多个节点 */
        bool AddVelocityBias(const VelBiasPtr &newVelocityBias);
        bool AddVelocityBiases(const std::vector<VelBiasPtr> &newVelocityBiases);
        /* 添加相机与 IMU 之间的相对位姿 */
        bool AddExPose(const ExPosePtr &exPose);
        /* 为问题添加 landmark 节点，同时添加其对应观测 std::unordered_map< (camera id), (observe on normal plane) > */
        /* 在添加了 landmark 节点之后，自动构建 landmark block 实例 */
        /* 必须在所有 camera 节点和 expose 节点都添加之后才可以添加 landmark */
        bool AddLandmark(const std::shared_ptr<VertexLandmark<Scalar>> &newLandmark,
                         const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
                         const size_t landmarkParamSize);
        /* 为问题添加 IMU 预积分块，构造其对应的约束关系 */
        bool AddIMUFactor(const IMUPtr &imuBlock);
        /* 获取当前 problem 中有多少相机 pose 节点 */
        size_t GetCamerasNum(void);
        /* 获取当前 problem 中有多少特征点 position 节点 */
        size_t GetLandmarksNum(void);
        /* 获取当前 problem 中有多少 velocity bias 节点 */
        size_t GetVelocityBiasesNum(void);
        /* 获取当前 problem 中有多少 IMU 约束因子 */
        size_t GetIMUBlocksNum(void);

    /* 求解 VO 问题的对外接口定义 */
    public:
        /* 采用 LM 算法迭代求解此问题 */
        bool Solve(size_t maxIteration);
        /* 用于测试功能 */
        bool Test(void);

    /* 求解 VO 问题的内部调用方法定义 */
    private:
        /* 遍历每一个 landmark block，基于当前相机 pose 和特征点的 position 来预计算误差 r */
        /* 遍历每一个 imu block，基于当前相机 pose、 velocity bias 和 IMU 预积分观测来预计算误差 r */
        /* 同时，计算 vo problem 的整体误差 */
        bool PrecomputeResidual(void);
        /* 遍历每一个 landmark block，基于当前相机 pose 和特征点的 position 来计算雅可比 J，
        /* 计算雅可比矩阵 Jp 和 Jl，对 Jl 进行 QR 分解，构造出完整的 storage 矩阵 */
        bool LinearlizeLandmarkBlocks(void);
        /* 遍历每一个 imu block，基于当前相机 pose、velocity bias 和 IMU 预积分观测来计算雅可比 J */
        bool LinearlizeIMUBlocks(void);
        /* 遍历每一个 landmark block，为其增加阻尼因子，并计算 Q_lambda 矩阵来调整 storage 矩阵的结构 */
        bool DampLandmarks(void);
        /* 基于每一个 landmark block 中的 storage 矩阵，以及每一个 imu block 中的雅可比矩阵 */
        /* 构造整个问题的大雅可比矩阵 J 和误差向量 r，并进一步构造增量方程 Hx=b */
        bool ConstructIncrementalFunction(void);
        /* 求解增量方程，求解出相机与 IMU 相对位姿、相机 pose、velocity bias 的增量 */
        /* 遍历每一个 landmark block，分别计算出对应的特征点 position 的增量，保存在 delta_X */
        bool SolveIncrementalFunction(void);
        /* 对所有的相机 pose 进行更新，更新后的状态变量保存在 VertexCameraPose 实例中 */
        /* 对每一个特征点 position 进行更新，更新后的状态变量保存在 VertexLandmark 实例中 */
        /* 如果有先验信息，则更新先验残差 */
        bool Update(void);
        /* 判断此步是否有效，并根据结果更新 LM 算法的阻尼因子 */
        bool IsValidStep(Scalar threshold);
        /* 在某一次迭代无效的情况下，需要对更新过的变量进行回退 */
        bool RollBack(void);

    /* 边缘化过程对外接口定义 */
    public:
        /* 在输入所有参与边缘化的 cameraPose、exPose、velocityBias 和 landmark 节点之后，指定被边缘化的帧的 ID 和输出的先验信息尺寸 */
        /* 进行边缘化，输出先验信息 */
        bool Marginalize(const size_t margedID, const size_t priorSize);
    
    /* 边缘化过程内部调用接口方法定义 */
    private:
        /* 根据参与边缘化的所有参数节点，确定雅可比矩阵的尺寸和先验信息的尺寸，并根据输入进行校验 */
        bool CheckSize(const size_t priorSize);
        /* 在完成 landmark block 的线性化之后，构造整体的雅可比矩阵和残差向量 */
        /* 并将之前的先验信息拼接在下面，共同构造出 this->matrix_J 和 this->vector_r */
        bool ConstructFullJacobianResidual(void);
        /* 将被边缘化的节点对应的矩阵块移动到雅可比矩阵的最左侧 */
        bool MoveMargedBlockToLeft(const size_t margedID);
    };
}


/* 此处为模板类成员的定义 */
/* 基本对外方法接口 */
namespace rootVIO {
    /* 构造函数与析构函数 */
    template<typename Scalar>
    ProblemVIO<Scalar>::ProblemVIO() {
        this->Reset();
    }
    template<typename Scalar>
    ProblemVIO<Scalar>::~ProblemVIO() {}


    /* 重置待求解问题，实际行为是清空所有保存的信息 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::Reset(void) {
        this->ResetBase();
        this->cameras.clear();
        this->cameras.reserve(20);
        this->landmarkBlocks.clear();
        this->landmarkBlocks.reserve(300);
        this->velocityBiases.clear();
        this->velocityBiases.reserve(20);
        this->imuBlocks.clear();
        this->imuBlocks.reserve(20);
        this->AddExPose(nullptr);
        return true;
    }


    /* 为问题添加相机 pose 节点，分为添加一个节点和一次性添加多个节点 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddCamera(const CameraPtr &newCamera) {
        this->cameras.emplace_back(newCamera);
        return true;
    }
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddCameras(const std::vector<CameraPtr> &newCameras) {
        for (auto it = newCameras.begin(); it != newCameras.end(); ++it) {
            this->cameras.emplace_back(*it);
        }
        return true;
    }


    /* 为问题添加 velocity bias 节点，分为添加一个节点和一次性添加多个节点 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddVelocityBias(const VelBiasPtr &newVelocityBias) {
        this->velocityBiases.emplace_back(newVelocityBias);
        return true;
    }
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddVelocityBiases(const std::vector<VelBiasPtr> &newVelocityBiases) {
        for (auto it = newVelocityBiases.begin(); it != newVelocityBiases.end(); ++it) {
            this->velocityBiases.emplace_back(*it);
        }
        return true;
    }


    /* 添加相机与 IMU 之间的相对位姿 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddExPose(const ExPosePtr &exPose) {
        if (exPose == nullptr) {
            return false;
        }
        this->exPose = exPose;
        return true;
    }


    /* 为问题添加 landmark 节点，同时添加其对应观测 std::unordered_map< (camera id), (observe on normal plane) > */
    /* 在添加了 landmark 节点之后，自动构建 landmark block 实例 */
    /* 必须在所有 camera 节点和 expose 节点都添加之后才可以添加 landmark */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddLandmark(const std::shared_ptr<VertexLandmark<Scalar>> &newLandmark,
        const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
        const size_t landmarkParamSize) {
        // 检查 observe 的索引是否存在越界情况
        for (auto it = observes.begin(); it != observes.end(); ++it) {
            if (it->first < 0 || it->first > this->cameras.size() - 1) {
                return false;
            }
        }
        // 构造新的 landmark block 实例，填充对应的 landmark 以及其观测
        if (landmarkParamSize == 3) {
            LandmarkPtr landmarkBlock(new LandmarkBlockPos<Scalar>(newLandmark, observes, this->exPose));
            this->landmarkBlocks.emplace_back(landmarkBlock);
            this->landmarkSize = 3;
        } else {
            LandmarkPtr landmarkBlock(new LandmarkBlockInvdep<Scalar>(newLandmark, observes, this->exPose));
            this->landmarkBlocks.emplace_back(landmarkBlock);
            this->landmarkSize = 1;
        }
        return true;
    }
                    

    /* 为问题添加 IMU 预积分块，构造其对应的约束关系 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::AddIMUFactor(const IMUPtr &imuBlock) {
        // 检查 imu block 的索引是否存在越界情况
        if (imuBlock->result.cameraID_j < 1 ||
            imuBlock->result.cameraID_j > this->cameras.size() - 1 ||
            imuBlock->result.cameraID_j > this->velocityBiases.size() - 1) {
            return false;
        }
        // 保存此 imu block
        this->imuBlocks.emplace_back(imuBlock);
        return true;
    }


    /* 获取当前 problem 中有多少相机 pose 节点 */
    template<typename Scalar>
    size_t ProblemVIO<Scalar>::GetCamerasNum(void) {
        return static_cast<size_t>(this->cameras.size());
    }


    /* 获取当前 problem 中有多少特征点 position 节点 */
    template<typename Scalar>
    size_t ProblemVIO<Scalar>::GetLandmarksNum(void) {
        return static_cast<size_t>(this->landmarkBlocks.size());
    }


    /* 获取当前 problem 中有多少 velocity bias 节点 */
    template<typename Scalar>
    size_t ProblemVIO<Scalar>::GetVelocityBiasesNum(void) {
        return static_cast<size_t>(this->velocityBiases.size());
    }


    /* 获取当前 problem 中有多少 IMU 约束因子 */
    template<typename Scalar>
    size_t ProblemVIO<Scalar>::GetIMUBlocksNum(void) {
        return static_cast<size_t>(this->imuBlocks.size());
    }
}


/* 此处为模板类成员的定义 */
/* 求解问题相关 */
namespace rootVIO {
    /* 采用 LM 算法迭代求解此问题 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::Solve(size_t maxIteration) {
        Timer timer, sumTimer;
        // 计算误差，对所有的 landmark blocks 和 imu blocks 进行线性化
        this->PrecomputeResidual();
        std::cout << "<Begin> origin sum cost r.T * S * r is " << this->sumCost << std::endl;
        this->rowsof_matrix_J = 0;
        this->LinearlizeLandmarkBlocks();
        this->LinearlizeIMUBlocks();
        // 初始化 LM 求解器
        this->InitializeLM();
        // 迭代求解
        for (size_t iter = 0; iter < maxIteration; ++iter) {
            // 给所有 landmark 添加阻尼因子
            this->DampLandmarks();
            // 拼接大雅可比矩阵 J 和残差向量 r，进一步构造增量方程 Hx=b
            this->ConstructIncrementalFunction();
            // 求解增量方程 Hx=b
            this->SolveIncrementalFunction();
            // 更新优化参数以及先验信息
            this->Update();
            // 计算参数更新之后的误差
            this->PrecomputeResidual();
            // 判断是否收敛，如果已经收敛，则可以提前结束
            if (this->IsConverged(iter)) {
                break;
            }
            // 判断本次迭代是否有效，并打印出本次迭代结果
            bool res = this->IsValidStep(0);
            Scalar stepTimeCost = timer.Stop();
            this->timeCost += stepTimeCost;
            std::cout << "<LM Iter " << iter << " / " << maxIteration <<"> cost " << this->sumCost << ", dx " <<
                this->normDeltaX << ", lambda " << this->lambda << ", rho " << this->rho << ", time cost " <<
                stepTimeCost << " ms" << std::endl;
            timer.Start();
            if (res) {
                // 如果本次迭代有效，则为下次迭代作准备
                this->invalidCount = 0;
                this->rowsof_matrix_J = 0;
                this->LinearlizeLandmarkBlocks();
                this->LinearlizeIMUBlocks();
            } else {
                // 如果本次迭代无效，则回退状态
                ++this->invalidCount;
                this->RollBack();
            }
            // 判断是否无法收敛，是的话则提前结束
            if (this->IsUnconverged(iter)) {
                break;
            }
        }
        std::cout << "<End> final sum cost r.T * S * r is " << this->sumCost << std::endl;
        std::cout << "<Finish> Problem solve totally cost " << sumTimer.Stop() << " ms\n";
        return true;
    }


    /* 用于测试功能 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::Test(void) {

        return true;
    }


    /* 遍历每一个 landmark block，基于当前相机 pose 和特征点的 position 来预计算误差 r */
    /* 遍历每一个 imu block，基于当前相机 pose、 velocity bias 和 IMU 预积分观测来预计算误差 r */
    /* 同时，计算 vo problem 的整体误差 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::PrecomputeResidual(void) {
        this->sumCost = 0;
        // 计算每一个 landmark block 的误差，并作累加
        this->sumCost = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, Scalar localSum) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->landmarkBlocks[i]->PrecomputeResidual();
                    auto observes = this->landmarkBlocks[i]->GetObserves();
                    for (auto &ob : observes) {
                        // 逆深度模式下，跳过首帧的核函数值(首帧应该是 0，无需做额外判断)
                        // 这里计算的是每一个观测误差的带核函数的马氏距离，即 rho(r.T * S * r)
                        localSum += ob.second->kernel->y;
                    }
                }
                return localSum;
            }, std::plus<Scalar>()
        );
        // 计算每一个 imu block 的误差，并作累加
        this->sumCost += tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->imuBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, Scalar localSum) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->imuBlocks[i]->ComputeResidual();
                    // 这里计算的是不带核函数的马氏距离，即 r.T * S * r
                    localSum += this->imuBlocks[i]->Get_rT_S_r();
                }
                return localSum;
            }, std::plus<Scalar>()
        );
        // 如果存在先验信息，则也算上他的误差
        if (this->prior_r.rows() > 0) {
            this->sumCost += this->prior_r.squaredNorm();
        }
        return true;
    }


    /* 遍历每一个 landmark block，基于当前相机 pose 和特征点的 position 来计算雅可比 J，
    /* 计算雅可比矩阵 Jp 和 Jl，对 Jl 进行 QR 分解，构造出完整的 storage 矩阵 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::LinearlizeLandmarkBlocks(void) {
        this->rowIndices.resize(this->landmarkBlocks.size() + 1);
        this->rowsof_matrix_J += tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, Scalar rows) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    // 清空 storage 矩阵
                    this->rowIndices[i + 1] = this->landmarkBlocks[i]->ResizeStorage();
                    rows += this->rowIndices[i + 1];
                    // 填充 storage 矩阵中的 r 部分
                    this->landmarkBlocks[i]->ComputeResidual();
                    // 填充 storage 矩阵中的 Jex、Jp、Jl 部分
                    this->landmarkBlocks[i]->ComputeJacobian();
                    // 对 Jl 进行 QR 分解，并对 storage 矩阵左乘 Q.T
                    this->landmarkBlocks[i]->PerformQR();
                }
                return rows;
            }, std::plus<Scalar>()
        );
        this->rowIndices[0] = 0;
        for (size_t i = 1; i < this->rowIndices.size(); ++i) {
            this->rowIndices[i] += this->rowIndices[i - 1];
        }
        // std::cout << "this->matrix_J size is [rows / cols] = [" << this->rowsof_matrix_J << " / " << this->colsof_matrix_J << "]\n";
        return true;
    }


    /* 遍历每一个 imu block，基于当前相机 pose、velocity bias 和 IMU 预积分观测来计算雅可比 J */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::LinearlizeIMUBlocks(void) {
        this->rowsof_matrix_J += tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->imuBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, Scalar rows) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    // 计算每一个 imu block 的雅可比矩阵，并根据其误差向量的维度，扩展大雅可比矩阵 J 的行数
                    this->imuBlocks[i]->ComputeJacobians();
                    rows += 15;
                }
                return rows;
            }, std::plus<Scalar>()
        );
        return true;
    }


    /* 遍历每一个 landmark block，为其增加阻尼因子，并计算 Q_lambda 矩阵来调整 storage 矩阵的结构 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::DampLandmarks(void) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    // 给 landmark block 的 storage 矩阵添加 landmark 的阻尼因子
                    this->landmarkBlocks[i]->Damp(this->lambda);
                }
            }
        );
        return true;
    }


    /* 基于每一个 landmark block 中的 storage 矩阵，以及每一个 imu block 中的雅可比矩阵 */
    /* 构造整个问题的大雅可比矩阵 J 和误差向量 r，并进一步构造增量方程 Hx=b */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::ConstructIncrementalFunction(void) {
        if (this->velocityBiases.size() != this->cameras.size()) {
            return false;
        }
        // 雅可比矩阵的每一列对应的变量定义如下
        /*
            Tbc   Twb1   vb1   Twb2   wb2   ...   Twbn   wbn
        */
        // 考虑先验信息的尺寸，为大雅可比矩阵 J 和大误差向量 r 分配空间
        this->colsof_matrix_J = this->cameras.size() * 6 + this->velocityBiases.size() * 9 + 6;
        this->rowsof_matrix_J += this->prior_J.rows();
        this->matrix_J.setZero(this->rowsof_matrix_J, this->colsof_matrix_J);
        this->vector_r.setZero(this->rowsof_matrix_J, 1);
        
        // 遍历每一个 landmark block，从中提取 Q2.T * J 和 Q2.T * r，填充到大雅可比矩阵和大误差向量的上面，依次往下
        size_t rowIndex = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, size_t rows) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    auto res = this->landmarkBlocks[i]->GetDampedDividedStorage();
                    this->matrix_J.block(this->rowIndices[i], 0, res.Q2T_Jex.rows(), res.Q2T_Jex.cols()) = res.Q2T_Jex;
                    this->vector_r.segment(this->rowIndices[i], res.Q2T_r.rows()) = res.Q2T_r;
                    for (auto &item : res.Q2T_Jps) {
                        this->matrix_J.block(this->rowIndices[i], item.first * 15 + 6,
                            item.second.rows(), item.second.cols()) = item.second;
                    }
                    rows += res.Q2T_r.rows();
                }
                return rows;
            }, std::plus<size_t>()
        );
        // 遍历每一个 imu block，从中提取出四个雅可比矩阵，填充到大雅可比矩阵和大误差向量的中间，依次往下
        size_t imuRowIndex = rowIndex;
        for (size_t i = 0; i < this->imuBlocks.size(); ++i) {
            auto &res = this->imuBlocks[i]->GetResidualJacobians();
            size_t idx = res.cameraID_j * 15 + 6;
            this->matrix_J.block(rowIndex, idx - 15, 15, 6) = res.dr_dpose_i;
            this->matrix_J.block(rowIndex, idx - 9,  15, 9) = res.dr_dvb_i;
            this->matrix_J.block(rowIndex, idx,      15, 6) = res.dr_dpose_j;
            this->matrix_J.block(rowIndex, idx + 6,  15, 9) = res.dr_dvb_j;
            this->vector_r.segment(rowIndex, 15) = res.residual;
            rowIndex += 15;
        }
        // 将先验信息填充到最下方
        if (this->prior_J.rows() > 0) {
            MatrixX<Scalar> tempPrior_J = this->prior_J;
            VectorX<Scalar> tempPrior_r = this->prior_r;
            if (this->exPose->IsFixed()) {
                tempPrior_J.block(0, 0, tempPrior_J.rows(), 6).setZero();
            }
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->cameras.size()),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        if (this->cameras[i]->IsFixed()) {
                            tempPrior_J.block(0, i * 6 + 6, tempPrior_J.rows(), 6).setZero();
                        }
                        if (this->velocityBiases[i]->IsFixed()) {
                            tempPrior_J.block(0, i * 15 + 12, tempPrior_J.rows(), 9).setZero();
                        }
                    }
                }
            );
            this->matrix_J.block(this->matrix_J.rows() - this->prior_J.rows(), 0, this->prior_J.rows(), this->prior_J.cols()) = tempPrior_J;
            this->vector_r.block(this->vector_r.rows() - this->prior_r.rows(), 0, this->prior_r.rows(), 1) = tempPrior_r;
        }
        // 计算增量方程中的 H 矩阵和 b 向量
        MatrixX<Scalar> &&Jt = this->matrix_J.transpose();
        this->matrix_H.resize(this->matrix_J.cols(), this->matrix_J.cols());
        this->vector_b.resize(this->matrix_J.cols());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->matrix_H.rows()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->matrix_H.block(i, 0, 1, this->matrix_H.cols()) = Jt.block(i, 0, 1, Jt.cols()) * this->matrix_J;
                    this->vector_b.segment(i, 1) = - Jt.block(i, 0, 1, Jt.cols()) * this->vector_r;
                }
            }
        );
        // this->matrix_H = this->matrix_J.transpose() * this->matrix_J;
        // this->vector_b = - this->matrix_J.transpose() * this->vector_r;
        // std::cout << "ProblemVIO<Scalar>::ConstructIncrementalFunction :\n  matrix J is\n" << this->matrix_J << "\n  vector r is\n" <<
        //     this->vector_r << std::endl;
        return true;
    }


    /* 求解增量方程，求解出相机与 IMU 相对位姿、相机 pose、velocity bias 的增量 */
    /* 遍历每一个 landmark block，分别计算出对应的特征点 position 的增量，保存在 delta_X */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::SolveIncrementalFunction(void) {
        // 备份未添加阻尼因子的 H 矩阵的对角线元素，为 H 矩阵添加阻尼因子
        this->diag_H = this->matrix_H.diagonal();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->matrix_H.rows()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->matrix_H(i, i) = this->diag_H(i) +
                        std::min(this->options.maxLambda, std::max(this->options.minLambda, this->matrix_H(i, i))) * this->lambda;
                }
            }
        );
        // 求解增量方程，得到 delta_X 的前半部分
        this->delta_X.resize(this->matrix_H.cols() + this->landmarkBlocks.size() * this->landmarkSize, 1);
        this->delta_X.head(this->matrix_H.cols()) = this->SolveLinearFunction(this->matrix_H, this->vector_b);
        // 将求解结果中的 expose 和 camera pose 分量提取到 delta_Xps 中
        VectorX<Scalar> delta_Xps;
        delta_Xps.resize(this->cameras.size() * 6 + 6, 1);
        delta_Xps.template head<6>() = this->delta_X.template head<6>();
        for (size_t i = 0; i < this->cameras.size(); ++i) {
            delta_Xps.segment(i * 6 + 6, 6) = this->delta_X.segment(i * 15 + 6, 6);
        }
        // 遍历每一个 landmark block，求解 delta_Xl 部分
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->delta_X.segment(this->matrix_H.rows() + i * this->landmarkSize, this->landmarkSize) =
                        this->landmarkBlocks[i]->ComputeDeltaXl(delta_Xps);
                }
            }
        );
        // 完成 delta_X 的求解，进一步计算其模长
        this->normDeltaX = this->delta_X.norm();
        // std::cout << "ProblemVIO<Scalar>::SolveIncrementalFunction delta_Xps is\n" << delta_Xps <<
        //     "\n delta_X is\n" << this->delta_X << "\n, norm of delta_X is\n" << this->normDeltaX << std::endl;
        return true;
    }


    /* 对所有的相机 pose 进行更新，更新后的状态变量保存在 VertexCameraPose 实例中 */
    /* 对每一个特征点 position 进行更新，更新后的状态变量保存在 VertexLandmark 实例中 */
    /* 如果有先验信息，则更新先验残差 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::Update(void) {
        // 备份并更新节点参数
        this->exPose->BackUp();
        this->exPose->UpdateParameters(this->delta_X.head(6));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->cameras.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->cameras[i]->BackUp();
                    this->cameras[i]->UpdateParameters(this->delta_X.segment(i * 15 + 6, 6));
                    this->velocityBiases[i]->BackUp();
                    this->velocityBiases[i]->UpdateParameters(this->delta_X.segment(i * 15 + 12, 9));
                }
            }
        );
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->landmarkBlocks[i]->GetLandmark()->BackUp();
                    this->landmarkBlocks[i]->GetLandmark()->UpdateParameters(
                        this->delta_X.segment(this->matrix_H.rows() + i * this->landmarkSize, this->landmarkSize)
                    );
                }
            }
        );
        // 备份并更新先验残差
        if (this->prior_J.cols() > 0) {
            this->stored_prior_r = this->prior_r;
            // - J.T * r => - J.T * r - J.T * J * dx
            // r => r + J * dx
            this->prior_r += this->prior_J * this->delta_X.head(this->prior_J.cols());
        }
        return true;
    }


    /* 判断此步是否有效，并根据结果更新 LM 算法的阻尼因子 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::IsValidStep(Scalar threshold) {
        // 扩展 this->vector_b 的维度，使其与 this->delta_X 一致
        VectorX<Scalar> b;
        b.setZero(this->delta_X.rows());
        b.head(this->vector_b.rows()) = this->vector_b;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    b.segment(this->vector_b.rows() + this->landmarkSize * i, this->landmarkSize) = - this->landmarkBlocks[i]->Get_Q1T_r();
                }
            }
        );
        /* 参考文献：The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems */
        if (this->policy == ProblemBase<Scalar>::DampPolicy::Manual) {
            // 扩展 this->diag_H 的维度, 使其与 this->delta_X 一致
            VectorX<Scalar> diag;
            diag.setZero(this->delta_X.rows());
            diag.head(this->diag_H.rows()) = this->diag_H;
            for (size_t i = 0; i < this->landmarkBlocks.size(); ++i) {
                diag.segment(this->vector_b.rows() + this->landmarkSize * i, this->landmarkSize) = this->landmarkBlocks[i]->Get_R1().diagonal();
            }
            Scalar scale = this->delta_X.transpose() * (this->lambda * diag.asDiagonal() * this->delta_X + b);
            this->rho = Scalar(0.5) * (this->linearizedCost - this->sumCost) / (scale + 1e-6);
            if (this->rho > threshold && std::isfinite(this->sumCost)) {
                this->lambda = std::max(this->lambda / this->Ldown, Scalar(1e-7));
                this->linearizedCost = this->sumCost;
                return true;
            } else {
                this->lambda = std::min(this->lambda * this->Lup, Scalar(1e7));
                return false;
            }
        } else if (this->policy == ProblemBase<Scalar>::DampPolicy::Auto) {
            Scalar scale = this->delta_X.transpose() * (this->lambda * this->delta_X + b);
            this->rho = Scalar(0.5) * (this->linearizedCost - this->sumCost) / (scale + 1e-6);
            if (this->rho > threshold && std::isfinite(this->sumCost)) {
                this->lambda *= std::max(Scalar(1.0 / 3.0), Scalar(1.0 - std::pow(2.0 * this->rho - 1.0, 3)));
                this->v = 2.0;
                this->linearizedCost = this->sumCost;
                return true;
            } else {
                this->lambda *= this->v;
                this->v *= 2.0;
                return false;
            }
        }
        return false;
    }


    /* 在某一次迭代无效的情况下，需要对更新过的变量进行回退 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::RollBack(void) {
        this->exPose->RollBack();
        for (auto &camera : this->cameras) {
            camera->RollBack();
        }
        for (auto &vb : this->velocityBiases) {
            vb->RollBack();
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->landmarkBlocks[i]->RollBack();
                    this->landmarkBlocks[i]->GetLandmark()->RollBack();
                }
            }
        );
        this->prior_r = this->stored_prior_r;
        return true;
    }
}


/* 边缘化相关过程的定义 */
namespace rootVIO {
    /* 在输入所有参与边缘化的 cameraPose、exPose、velocityBias 和 landmark 节点之后，指定被边缘化的帧的 ID 和输出的先验信息尺寸 */
    /* 进行边缘化，输出先验信息 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::Marginalize(const size_t margedID, const size_t priorSize) {
        // 计算误差，线性化所有 landmark blocks 和 imu blocks
        this->PrecomputeResidual();
        this->rowsof_matrix_J = 0;
        this->LinearlizeLandmarkBlocks();
        this->LinearlizeIMUBlocks();
        // 根据输入的所有节点，以及被边缘化的 camera ID 和最后得到的先验信息的尺寸 priorSize
        // 计算出需要构造的雅可比矩阵和残差向量的维度，并校验设置的先验信息尺寸是否合理
        bool res = this->CheckSize(priorSize);
        if (res == false) {
            return false;
        }
        // 在不添加阻尼因子的情况下，构造大雅可比矩阵和大残差向量，并在其最下方添加先验信息
        this->ConstructFullJacobianResidual();
        // 将待边缘化的节点移动到最左边
        this->MoveMargedBlockToLeft(margedID);
        // 对大雅可比矩阵和大残差向量进行 Householder 变换，构造出先验信息
        this->ConstructPrior(15);
        return true;
    }


    /* 根据参与边缘化的所有参数节点，确定雅可比矩阵的尺寸和先验信息的尺寸，并根据输入进行校验 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::CheckSize(const size_t priorSize) {
        this->colsof_matrix_J = this->cameras.size() * 6 + this->velocityBiases.size() * 9 + 6;
        // this->rowsof_matrix_J 已经在 this->LinearlizeLandmarkBlocks() 中做了统计
        // 但是行数本身算上了 damp 的固定几行，因此需要去掉
        this->rowsof_matrix_J -= this->landmarkSize * this->landmarkBlocks.size();
        this->rowsof_matrix_J += this->prior_J.rows();
        if (this->rowsof_matrix_J == 0 || this->colsof_matrix_J < this->prior_J.cols()) {
            return false;
        }
        if (this->colsof_matrix_J - 15 != priorSize) {
            return false;
        }
        return true;
    }


    /* 在完成 landmark block 的线性化之后，构造整体的雅可比矩阵和残差向量 */
    /* 并将之前的先验信息拼接在下面，共同构造出 this->matrix_J 和 this->vector_r */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::ConstructFullJacobianResidual(void) {
        if (this->velocityBiases.size() != this->cameras.size()) {
            return false;
        }
        // 雅可比矩阵的每一列对应的变量定义如下
        /*
            Tbc   Twb1   vb1   Twb2   wb2   ...   Twbn   wbn
        */
        // 考虑先验信息的尺寸，为大雅可比矩阵 J 和大误差向量 r 分配空间
        this->matrix_J.setZero(this->rowsof_matrix_J, this->colsof_matrix_J);
        this->vector_r.setZero(this->rowsof_matrix_J, 1);
        // 遍历每一个 landmark block，从中提取 Q2.T * J 和 Q2.T * r，填充到大雅可比矩阵和大误差向量的上面，依次往下
        size_t rowIndex = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->landmarkBlocks.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, size_t rows) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    auto res = this->landmarkBlocks[i]->GetUndampedDividedStorage();
                    this->matrix_J.block(this->rowIndices[i] - this->landmarkSize * i, 0, res.Q2T_Jex.rows(), res.Q2T_Jex.cols()) = res.Q2T_Jex;
                    this->vector_r.segment(this->rowIndices[i] - this->landmarkSize * i, res.Q2T_r.rows()) = res.Q2T_r;
                    for (auto &item : res.Q2T_Jps) {
                        this->matrix_J.block(this->rowIndices[i] - this->landmarkSize * i,
                            item.first * 15 + 6,
                            item.second.rows(), item.second.cols()) = item.second;
                    }
                    rows += res.Q2T_r.rows();
                }
                return rows;
            }, std::plus<size_t>()
        );
        // 遍历每一个 imu block，从中提取出四个雅可比矩阵，填充到大雅可比矩阵和大误差向量的中间，依次往下
        for (size_t i = 0; i < this->imuBlocks.size(); ++i) {
            auto &res = this->imuBlocks[i]->GetResidualJacobians();
            size_t idx = res.cameraID_j * 15 + 6;
            this->matrix_J.block(rowIndex, idx - 15, 15, 6) = res.dr_dpose_i;
            this->matrix_J.block(rowIndex, idx - 9,  15, 9) = res.dr_dvb_i;
            this->matrix_J.block(rowIndex, idx,      15, 6) = res.dr_dpose_j;
            this->matrix_J.block(rowIndex, idx + 6,  15, 9) = res.dr_dvb_j;
            this->vector_r.segment(rowIndex, 15) = res.residual;
            rowIndex += 15;
        }
        // 将先验信息填充到最下方
        if (this->prior_J.rows() > 0) {
            this->matrix_J.block(this->matrix_J.rows() - this->prior_J.rows(), 0, this->prior_J.rows(), this->prior_J.cols()) = this->prior_J;
            this->vector_r.block(this->vector_r.rows() - this->prior_r.rows(), 0, this->prior_r.rows(), 1) = this->prior_r;
        }
        // std::cout << "ProblemVIO<Scalar>::ConstructFullJacobianResidual :\n  matrix J is\n" << this->matrix_J << "\n  vector r is\n" <<
        //     this->vector_r << std::endl;
        return true;
    }


    /* 将被边缘化的节点对应的矩阵块移动到雅可比矩阵的最左侧 */
    template<typename Scalar>
    bool ProblemVIO<Scalar>::MoveMargedBlockToLeft(const size_t margedID) {
        size_t rows = this->matrix_J.rows();
        size_t cols = this->matrix_J.cols();
        size_t idx = margedID * 15 + 6;
        MatrixX<Scalar> tempCols = this->matrix_J.block(0, idx, rows, 15);
        MatrixX<Scalar> tempLeftCols = this->matrix_J.block(0, 0, rows, idx);
        this->matrix_J.block(0, 0, rows, 15) = tempCols;
        this->matrix_J.block(0, 15, rows, idx) = tempLeftCols;
        // std::cout << "ProblemVIO<Scalar>::MoveMargedBlockToLeft :\n  matrix J is\n" << this->matrix_J << "\n  vector r is\n" <<
        //     this->vector_r << std::endl;
        return true;
    }
}