#pragma once

#include <include/ba_solver/root_vio/landmark_block_base.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {

    // 定义 landmark block pos 类，管理数据存储与相关运算方法
    template<typename Scalar>
    class LandmarkBlockInvdep : public LandmarkBlockBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        LandmarkBlockInvdep() : LandmarkBlockBase<Scalar>() {}
        LandmarkBlockInvdep(const std::shared_ptr<VertexLandmark<Scalar>> &landmark,
                            const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
                            const std::shared_ptr<VertexExPose<Scalar>> &exPose) :
            LandmarkBlockBase<Scalar>(landmark, observes, exPose) {}
        ~LandmarkBlockInvdep() {}
    
    public:
        /* 重新初始化 storage 的尺寸，并清空其中的数据，返回作用于 problem 中大雅可比矩阵中的行数 */
        /* 此时得到一个确定尺寸的 0 矩阵：
            [ 0 | 0 | 0 ] */
        virtual size_t ResizeStorage(void) override;
        /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，暂时不填入到 storage 中 */
        /* 此时 storate 矩阵保持不变 */
        virtual bool PrecomputeResidual(void) override;
        /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，填入到 storage 中 */
        /* 此时填充了 storage 的一部分：
            [ 0 | 0 | r ]
            [ 0 | 0 | 0 ] */
        virtual bool ComputeResidual(void) override;
        /* 计算雅可比矩阵，填充 storage，完成线性化操作 */
        /* 此时填充完成 storage 矩阵:
            [ Jp(format diag block matrix) | Jl | r ]
            [     0                        | 0  | 0 ] */
        virtual bool ComputeJacobian(void) override;
        /* 对 storage 中的 Jl 进行 QR 分解，基于分解结果调整 storage 矩阵 */
        /* 此时基于 QR 分解调整后的 storage 矩阵：
            [ Q1.T * Jp |   R1   | Q1.T * r ]
            [ Q2.T * Jp | R2(=0) | Q2.T * r ]
            [     0     |    0   |     0    ] */
        virtual bool PerformQR(void) override;
        /* 给 storage 矩阵添加阻尼因子，并计算 givens 矩阵使得 storage 矩阵形式上保持不变，记录下 givens 矩阵在 this->Q_lambda */
        /* 添加阻尼因子后的 storage 矩阵：
            [ Q1.T * Jp |        R1        | Q1.T * r ]
            [ Q2.T * Jp |      R2(=0)      | Q2.T * r ]
            [     0     | sqrt(lambda) * I |     0    ] */
        /* 经过 Givens 矩阵调整后的 storage 矩阵，维度和上面的矩阵保持一致：
            [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
            [ Q2_.T * Jp |   0   | Q2_.T * r ]
            [     0     |    0   |     0    ] */
        virtual bool Damp(Scalar lambda) override;
        /* 在 storage 矩阵被 damp 的前提下 */
        /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
        /* 用于 problem 填充大雅可比矩阵 */
        virtual typename LandmarkBlockBase<Scalar>::DividedResult GetDampedDividedStorage(void) override;
        /* 返回 damp 之后的 storage 矩阵有效行数 */
        virtual size_t GetDampedDividedStorageRows(void) override;
        /* 在 storage 矩阵没有被 damp 的前提下 */
        /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
        /* 用于 problem 填充大雅可比矩阵 */
        virtual typename LandmarkBlockBase<Scalar>::DividedResult GetUndampedDividedStorage(void) override;
        /* 返回 damp 之后的 storage 矩阵有效行数 */
        virtual size_t GetUndampedDividedStorageRows(void) override;
        /* 返回 delta_Xl 增量方程的 H 矩阵 */
        /* 本质上就是从 storage 矩阵中提取出 R1 */
        virtual MatrixX<Scalar> Get_R1(void) override;
        /* 返回 delta_Xl 增量方程的 b 向量 */
        /* 本质上就是从 storage 矩阵中提取出 Q1_.T * r */
        virtual VectorX<Scalar> Get_Q1T_r(void) override;
        /* 基于已知的相机 pose 增量的结果 delta_Xp，求解此特征点 position 对应的增量 */
        /* 就是计算 delta_Xl = - R1.inv * (Q1T * r + Q1T * Jp * delta_Xp) */
        virtual VectorX<Scalar> ComputeDeltaXl(const VectorX<Scalar> &delta_Xp) override;
        /* 基于保存的 Q_lambda 矩阵，回退 storage 矩阵至原来的状态 */
        /* 同时，为了方便再次调用 Damp 方法，此处在回退后需要去除因 damp 而增加的空白行 */
        /* 回退之前的 storage 矩阵：
            [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
            [ Q2_.T * Jp |   0   | Q2_.T * r ] */
        /* 回退之后的 storage 矩阵：
            [ Q1.T * Jp |  R1  | Q1.T * r ]
            [ Q2.T * Jp |   0  | Q2.T * r ]
            [     0     |   0  |     0    ] */
        virtual bool RollBack(void) override;

    private:
        /* 根据 camera ID 查找其在 storage 矩阵中的行索引 */
        size_t FindRowIndex(size_t cameraID);
    };
}


// 此处为模板类的成员的定义
namespace rootVIO {
    /* 重新初始化 storage 的尺寸，并清空其中的数据 */
    /* 此时得到一个确定尺寸的 0 矩阵：
        [ 0 | 0 | 0 ] */
    template<typename Scalar>
    size_t LandmarkBlockInvdep<Scalar>::ResizeStorage(void) {
        // T_bc, T_wb_1, T_wb_2, ... , T_wb_n, invdep, r
        // observe2, ... , observe_n, damp
        /*
        [  Jex   Jp1   Jp2    0     0   |  Jl2  |  r2   ]
        [  Jex   Jp1    0    Jp3    0   |  Jl3  |  r3   ]
        [  Jex   Jp1    0     0    Jp4  |  Jl4  |  r4   ]
        [   0     0     0     0     0   |   0   |   0   ] * 1
        */
        // 但是首帧不一定在第二列
        size_t cameraNums = static_cast<size_t>(this->observes.size());
        size_t cols = 6 + cameraNums * 6 + 1 + 1;
        size_t rows = (cameraNums - 1) * 2 + 1;
        this->storage.setZero(rows, cols);
        // std::cout << "LandmarkBlockInvdep<Scalar>::ResizeStorage :\n" << this->storage << std::endl;
        return rows - 1;
    }


    /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，暂时不填入到 storage 中 */
    /* 此时 storate 矩阵保持不变 */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::PrecomputeResidual(void) {
        this->residuals.clear();

        // 提取首帧（第 i 帧）观测帧相关信息
        auto observe_i = *this->observes.find(this->cameraID_i);
        auto camera_i = observe_i.second->camera;
        Vector2<Scalar> measure_i = observe_i.second->norm;
        // 首帧的核函数和信息矩阵都不需要，在 problem 中需要对应剔除
        observe_i.second->kernel->Compute(0);

        for (auto &observe_j : this->observes) {
            // 提取第 j 帧相关信息
            size_t cameraID_j = observe_j.first;
            if (cameraID_j == this->cameraID_i) {
                continue;
            }
            auto &camera_j = observe_j.second->camera;
            Vector2<Scalar> &measure_j = observe_j.second->norm;
            auto &kernel_j = observe_j.second->kernel;
            Matrix2<Scalar> &sqrtInfo = observe_j.second->sqrtInfo;

            // 提取位姿参数
            Quaternion<Scalar> &q_wb_i = camera_i->Get_q_wb();
            Vector3<Scalar> &t_wb_i = camera_i->Get_t_wb();
            Quaternion<Scalar> &q_wb_j = camera_j->Get_q_wb();
            Vector3<Scalar> &t_wb_j = camera_j->Get_t_wb();
            Quaternion<Scalar> &q_bc = this->exPose->Get_q_bc();
            Vector3<Scalar> &t_bc = this->exPose->Get_t_bc();
            Scalar &invdep_i = this->landmark->Get_invdep();

            // 计算重投影的归一化平面坐标 predict = [ux, uy, 1].T
            Vector3<Scalar> p_c_i = Vector3<Scalar>(measure_i(0), measure_i(1), Scalar(1)) / invdep_i;
            Vector3<Scalar> p_b_i = q_bc * p_c_i + t_bc;
            Vector3<Scalar> p_w = q_wb_i * p_b_i + t_wb_i;
            Vector3<Scalar> p_b_j = q_wb_j.inverse() * (p_w - t_wb_j);
            Vector3<Scalar> p_c_j = q_bc.inverse() * (p_b_j - t_bc);
            Scalar invdep_j = Scalar(1) / p_c_j.z();
            Vector2<Scalar> predict = p_c_j.template head<2>() * invdep_j;

            // 计算重投影误差
            Vector2<Scalar> residual;
            if (std::isnan(invdep_i) || std::isinf(invdep_i) ||
                std::isnan(invdep_j) || std::isinf(invdep_j)) {
                residual.setZero();
            } else {
                residual = predict - measure_j;
            }

            // 计算 rho(r.T * S * r) 即 y，以及鲁棒核权重 rho'(r.T * S * r) 即 y_
            Scalar x = residual.transpose() * sqrtInfo * sqrtInfo * residual;
            kernel_j->Compute(x);

            // 计算 sqrt(S) * r，作为误差输出结果
            Vector2<Scalar> weightedResidual = std::sqrt(kernel_j->y_) * sqrtInfo * residual;
            this->residuals.insert(std::make_pair(cameraID_j, weightedResidual));
            // std::cout << "this->residuals.insert [ " << cameraID_j << ", " << weightedResidual.transpose() << " ]\n";
        }
        return true;
    }


    /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，填入到 storage 中 */
    /* 此时填充了 storage 的一部分：
        [ 0 | 0 | r ]
        [ 0 | 0 | 0 ] */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::ComputeResidual(void) {
        // 如果还没有预计算过误差，则需要预计算，然后才能进行填充
        if (this->residuals.empty()) {
            this->PrecomputeResidual();
        }
        for (auto &residual : this->residuals) {
            size_t idx = this->FindRowIndex(residual.first);
            this->storage.template block(idx * 2, this->storage.cols() - 1, 2, 1) = residual.second;
        }
        this->residuals.clear();
        // std::cout << "LandmarkBlockInvdep<Scalar>::ComputeResidual :\n" << this->storage << std::endl;
        return true;
    }


    /* 计算雅可比矩阵，填充 storage，完成线性化操作 */
    /* 此时填充完成 storage 矩阵:
        [ Jp(format diag block matrix) | Jl | r ]
        [     0                        | 0  | 0 ] */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::ComputeJacobian(void) {

        // 提取首帧（第 i 帧）观测帧相关信息
        auto observe_i = *this->observes.find(this->cameraID_i);
        auto camera_i = observe_i.second->camera;
        Vector2<Scalar> measure_i = observe_i.second->norm;
        size_t colIdx_i = (*this->cameraID_idx.find(this->cameraID_i)).second;

        for (auto &observe_j : this->observes) {
            // 提取第 j 帧相关信息
            size_t cameraID_j = observe_j.first;
            if (cameraID_j == this->cameraID_i) {
                continue;
            }
            auto &camera_j = observe_j.second->camera;
            Vector2<Scalar> &measure_j = observe_j.second->norm;
            auto &kernel_j = observe_j.second->kernel;
            Matrix2<Scalar> &sqrtInfo = observe_j.second->sqrtInfo;

            // 提取位姿参数
            Quaternion<Scalar> &q_wb_i = camera_i->Get_q_wb();
            Vector3<Scalar> &t_wb_i = camera_i->Get_t_wb();
            Quaternion<Scalar> &q_wb_j = camera_j->Get_q_wb();
            Vector3<Scalar> &t_wb_j = camera_j->Get_t_wb();
            Quaternion<Scalar> &q_bc = this->exPose->Get_q_bc();
            Vector3<Scalar> &t_bc = this->exPose->Get_t_bc();
            Scalar &invdep_i = this->landmark->Get_invdep();

            // 计算重投影的归一化平面坐标 predict = [ux, uy, 1].T
            Vector3<Scalar> p_c_i = Vector3<Scalar>(measure_i(0), measure_i(1), Scalar(1)) / invdep_i;
            Vector3<Scalar> p_b_i = q_bc * p_c_i + t_bc;
            Vector3<Scalar> p_w = q_wb_i * p_b_i + t_wb_i;
            Vector3<Scalar> p_b_j = q_wb_j.inverse() * (p_w - t_wb_j);
            Vector3<Scalar> p_c_j = q_bc.inverse() * (p_b_j - t_bc);
            Scalar invdep_j = Scalar(1) / p_c_j.z();

            // 计算临时变量
            Matrix3<Scalar> R_bc(q_bc);
            Matrix3<Scalar> R_wb_i(q_wb_i);
            Matrix3<Scalar> R_wb_j(q_wb_j);

            // 计算（归一化二维平面）对（相机坐标系三维坐标）的雅可比矩阵
            Eigen::Matrix<Scalar, 2, 3> reduce;
            if (std::isnan(invdep_i) || std::isinf(invdep_i) ||
                std::isnan(invdep_j) || std::isinf(invdep_j)) {
                reduce.setZero();
            } else {
                reduce << invdep_j, 0, - p_c_j(0) * invdep_j * invdep_j,
                          0, invdep_j, - p_c_j(1) * invdep_j * invdep_j;
            }
            
            // 计算（相机坐标系三维坐标的误差）对（第 i 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> tempJpi;
            if (camera_i->IsFixed()) {
                tempJpi.setZero();
            } else {
                tempJpi.template leftCols<3>() = R_bc.transpose() * R_wb_j.transpose();
                tempJpi.template rightCols<3>() = - R_bc.transpose() * R_wb_j.transpose() * R_wb_i * SkewSymmetricMatrix(p_b_i);
            }

            // 计算（相机坐标系三维坐标的误差）对（第 j 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> tempJpj;
            if (camera_j->IsFixed()) {
                tempJpj.setZero();
            } else {
                tempJpj.template leftCols<3>() = - R_bc.transpose() * R_wb_j.transpose();
                tempJpj.template rightCols<3>() = R_bc.transpose() * SkewSymmetricMatrix(p_b_j);
            }

            // 计算（相机坐标系三维坐标的误差）对（第 i 帧中特征点逆深度）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 1> tempJl;
            if (this->landmark->IsFixed()) {
                tempJl.setZero();
            } else {
                Vector3<Scalar> normPoint_i(measure_i(0), measure_i(1), Scalar(1));
                tempJl = - R_bc.transpose() * R_wb_j.transpose() * R_wb_i * R_bc * normPoint_i / (invdep_i * invdep_i);
            }

            // 计算（相机坐标系三维坐标的误差）对（IMU 和相机相对位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> tempJex;
            if (this->exPose->IsFixed()) {
                tempJex.setZero();
            } else {
                Matrix3<Scalar> temp = R_bc.transpose() * R_wb_j.transpose() * R_wb_i * R_bc;
                tempJex.template leftCols<3>() = R_bc.transpose() * (R_wb_j.transpose() * R_wb_i - Matrix3<Scalar>::Identity());
                tempJex.template rightCols<3>() = - temp * SkewSymmetricMatrix(p_c_i) + SkewSymmetricMatrix(temp * p_c_i) +
                    SkewSymmetricMatrix(R_bc.transpose() * (R_wb_j.transpose() * (R_wb_i * t_bc + t_wb_i - t_wb_j) - t_bc));
            }

            // 通过级联求导的方式，计算雅可比矩阵，填充 storage 矩阵
            size_t colIdx_j = (*this->cameraID_idx.find(observe_j.first)).second;
            size_t rowIdx = this->FindRowIndex(observe_j.first) * 2;
            Scalar w = std::sqrt(kernel_j->y_);
            this->storage.template block<2, 6>(rowIdx, 0) = w * sqrtInfo * reduce * tempJex;
            this->storage.template block<2, 1>(rowIdx, this->storage.cols() - 2) = w * sqrtInfo * reduce * tempJl;
            this->storage.template block<2, 6>(rowIdx, 6 + colIdx_i * 6) = w * sqrtInfo * reduce * tempJpi;
            this->storage.template block<2, 6>(rowIdx, 6 + colIdx_j * 6) = w * sqrtInfo * reduce * tempJpj;
        }
        // std::cout << "LandmarkBlockInvdep<Scalar>::ComputeJacobian :\n" << this->storage << std::endl;
        return true;
    }


    /* 对 storage 中的 Jl 进行 QR 分解，基于分解结果调整 storage 矩阵 */
    /* 此时基于 QR 分解调整后的 storage 矩阵：
        [ Q1.T * Jp |   R1   | Q1.T * r ]
        [ Q2.T * Jp | R2(=0) | Q2.T * r ]
        [     0     |    0   |     0    ] */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::PerformQR(void) {
        // 提取 storage 矩阵中的三个部分
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        MatrixX<Scalar> Jl = this->storage.block(0, cols - 2, rows - 1, 1);
        MatrixX<Scalar> Jp = this->storage.block(0, 0, rows - 1, cols - 2);
        VectorX<Scalar> r = this->storage.block(0, cols - 1, rows - 1, 1);

        // 对 Jl 进行 QR 分解
        Eigen::HouseholderQR<MatrixX<Scalar>> qr;
        qr.compute(Jl);
        MatrixX<Scalar> Qt = qr.householderQ().transpose();

        // 对 storage 矩阵左乘 Q.T
        this->storage.block(0, cols - 2, rows - 1, 1)
            = qr.matrixQR().template triangularView<Eigen::Upper>();
        this->storage.block(0, 0, rows - 1, cols - 2) = Qt * Jp;
        this->storage.block(0, cols - 1, rows - 1, 1) = Qt * r;
        // std::cout << "LandmarkBlockInvdep<Scalar>::PerformQR :\n" << this->storage << std::endl;
        return true;
    }


    /* 给 storage 矩阵添加阻尼因子，并计算 givens 矩阵使得 storage 矩阵形式上保持不变，记录下 givens 矩阵在 this->Q_lambda */
    /* 添加阻尼因子后的 storage 矩阵：
        [ Q1.T * Jp |        R1        | Q1.T * r ]
        [ Q2.T * Jp |      R2(=0)      | Q2.T * r ]
        [     0     | sqrt(lambda) * I |     0    ] */
    /* 经过 Givens 矩阵调整后的 storage 矩阵，维度和上面的矩阵保持一致：
        [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
        [ Q2_.T * Jp |   0   | Q2_.T * r ] */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::Damp(Scalar lambda) {
        // 将 storage 矩阵的最下面一行置为零
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        this->storage.block(rows - 1, 0, 1, cols).setZero();

        // 在指定位置添加阻尼因子
        this->storage.block(rows - 1, cols - 2, 1, 1).diagonal().array() = std::sqrt(lambda);

        // 为更正 storage 矩阵中对应于 Jl_ 位置的格式（上为上三角，下为全零），计算 Givens 矩阵
        Eigen::JacobiRotation<Scalar> gr;
        this->Q_lambda.setIdentity(rows, rows);
        size_t n = cols - 2;    // 列索引
        size_t m = rows - 1;    // 行索引
        // makeGivens() 方法本质上就是传入 x1 和 x2，计算出旋转矩阵对应的 sin 和 cos
        gr.makeGivens(this->storage(0, n), this->storage(m, n));
        // 对 this->storage 矩阵进行初等变换，并将连乘结果记录下来
        this->storage.applyOnTheLeft(m, 0, gr);
        this->Q_lambda.applyOnTheLeft(m, 0, gr);

        // 运算精度会导致部分位置不为零，因此这些位置需要强制给零
        for (size_t m = rows - 1; m > 0; --m) {
            this->storage(m, n) = 0;
        }
        // std::cout << "LandmarkBlockInvdep<Scalar>::Damp :\n" << this->storage << std::endl;
        return true;
    }


    /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
    /* 用于 problem 填充大雅可比矩阵 */
    template<typename Scalar>
    typename LandmarkBlockBase<Scalar>::DividedResult LandmarkBlockInvdep<Scalar>::GetDampedDividedStorage(void) {
        typename LandmarkBlockBase<Scalar>::DividedResult res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        res.Q2T_Jex = this->storage.block(1, 0, rows - 1, 6);
        res.Q2T_r = this->storage.block(1, cols - 1, rows - 1, 1);
        for (auto &observe : this->observes) {
            size_t idx = (*this->cameraID_idx.find(observe.first)).second;
            MatrixX<Scalar> &&Q2T_Jp = this->storage.block(1, idx * 6 + 6, rows - 1, 6);
            res.Q2T_Jps.insert(std::make_pair(observe.first, Q2T_Jp));
        }
        return res;
    }


    /* 返回 damp 之后的 storage 矩阵有效行数 */
    template<typename Scalar>
    size_t LandmarkBlockInvdep<Scalar>::GetDampedDividedStorageRows(void) {
        return this->storage.rows() - 1;
    }


    /* 在 storage 矩阵没有被 damp 的前提下 */
    /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
    /* 用于 problem 填充大雅可比矩阵 */
    template<typename Scalar>
    typename LandmarkBlockBase<Scalar>::DividedResult LandmarkBlockInvdep<Scalar>::GetUndampedDividedStorage(void) {
        typename LandmarkBlockBase<Scalar>::DividedResult res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        res.Q2T_Jex = this->storage.block(1, 0, rows - 2, 6);
        res.Q2T_r = this->storage.block(1, cols - 1, rows - 2, 1);
        for (auto &observe : this->observes) {
            size_t idx = (*this->cameraID_idx.find(observe.first)).second;
            MatrixX<Scalar> &&Q2T_Jp = this->storage.block(1, idx * 6 + 6, rows - 2, 6);
            res.Q2T_Jps.insert(std::make_pair(observe.first, Q2T_Jp));
        }
        return res;
    }


    /* 返回 damp 之后的 storage 矩阵有效行数 */
    template<typename Scalar>
    size_t LandmarkBlockInvdep<Scalar>::GetUndampedDividedStorageRows(void) {
        return this->storage.rows() - 2;
    }


    /* 返回 delta_Xl 增量方程的 H 矩阵 */
    /* 本质上就是从 storage 矩阵中提取出 R1 */
    template<typename Scalar>
    MatrixX<Scalar> LandmarkBlockInvdep<Scalar>::Get_R1(void) {
        MatrixX<Scalar> &&R1 = this->storage.block(0, this->storage.cols() - 2, 1, 1);
        return R1;
    }


    /* 返回 delta_Xl 增量方程的 b 向量 */
    /* 本质上就是从 storage 矩阵中提取出 Q1_.T * r */
    template<typename Scalar>
    VectorX<Scalar> LandmarkBlockInvdep<Scalar>::Get_Q1T_r(void) {
        VectorX<Scalar> &&Q1T_r = this->storage.block(0, this->storage.cols() - 1, 1, 1);
        return Q1T_r;
    }


    /* 基于已知的相机 pose 增量的结果 delta_Xp，求解此特征点 position 对应的增量 */
    /* 就是计算 delta_Xl = - R1.inv * (Q1T * r + Q1T * Jp * delta_Xp) */
    template<typename Scalar>
    VectorX<Scalar> LandmarkBlockInvdep<Scalar>::ComputeDeltaXl(const VectorX<Scalar> &delta_Xp) {
        VectorX<Scalar> res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        // 首先需要从全局的 delta_Xp 中提取出此 landmark block 对应的 delta_Xps
        VectorX<Scalar> delta_Xps;
        delta_Xps.setZero(cols - 2, 1);
        delta_Xps.head(6) = delta_Xp.head(6);
        for (auto &item : this->observes) {
            size_t idx = (*this->cameraID_idx.find(item.first)).second;
            delta_Xps.segment(idx * 6 + 6, 6) = delta_Xp.segment(item.first * 6 + 6, 6);
        }
        // 然后再去求解 delta_Xl
        MatrixX<Scalar> &&Q1T_Jp = this->storage.block(0, 0, 1, cols - 2);
        VectorX<Scalar> &&Q1T_r = this->storage.block(0, cols - 1, 1, 1);
        MatrixX<Scalar> &&R1 = this->storage.block(0, cols - 2, 1, 1);
        res = - R1.inverse() * (Q1T_r + Q1T_Jp * delta_Xps);
        // std::cout << "delta_xl is " << res.transpose() << std::endl;
        return res;
    }


    /* 基于保存的 Q_lambda 矩阵，回退 storage 矩阵至原来的状态 */
    /* 同时，为了方便再次调用 Damp 方法，此处在回退后需要去除因 damp 而增加的空白行 */
    /* 回退之前的 storage 矩阵：
        [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
        [ Q2_.T * Jp |   0   | Q2_.T * r ] */
    /* 回退之后的 storage 矩阵：
        [ Q1.T * Jp |  R1  | Q1.T * r ]
        [ Q2.T * Jp |   0  | Q2.T * r ]
        [     0     |   0  |     0    ] */
    template<typename Scalar>
    bool LandmarkBlockInvdep<Scalar>::RollBack(void) {
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        this->storage = this->Q_lambda.transpose() * this->storage;
        // 运算精度导致部分为零的地方不为零，这里强制归零
        for (size_t i = 0; i < 1; ++i) {
            size_t n = i + cols - 1 - 1;
            for (size_t m = rows - 1; m > i; --m) {
                this->storage(m, n) = 0;
            }
        }
        this->storage.block(rows - 1, 0, 1, cols).setZero();
        // std::cout << "LandmarkBlockInvdep<Scalar>::RollBack :\n" << this->storage << std::endl;
        return true;
    }


    /* 根据 camera ID 查找其在 storage 矩阵中的行索引 */
    template<typename Scalar>
    size_t LandmarkBlockInvdep<Scalar>::FindRowIndex(size_t cameraID) {
        size_t idx = (*this->cameraID_idx.find(cameraID)).second;
        size_t firstIdx = (*this->cameraID_idx.find(this->cameraID_i)).second;
        if (idx > firstIdx) {
            --idx;
        }
        return idx;
    }
}