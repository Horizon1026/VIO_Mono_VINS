#pragma once

#include <include/ba_solver/root_vio/landmark_block_base.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {

    // 定义 landmark block pos 类，管理数据存储与相关运算方法
    template<typename Scalar>
    class LandmarkBlockPos : public LandmarkBlockBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        LandmarkBlockPos() : LandmarkBlockBase<Scalar>() {}
        LandmarkBlockPos(const std::shared_ptr<VertexLandmark<Scalar>> &landmark,
                         const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
                         const std::shared_ptr<VertexExPose<Scalar>> &exPose) :
            LandmarkBlockBase<Scalar>(landmark, observes, exPose) {}
        ~LandmarkBlockPos() {}
    
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
    };
}


// 此处为模板类的成员的定义
namespace rootVIO {
    /* 重新初始化 storage 的尺寸，并清空其中的数据 */
    /* 此时得到一个确定尺寸的 0 矩阵：
        [ 0 | 0 | 0 ] */
    template<typename Scalar>
    size_t LandmarkBlockPos<Scalar>::ResizeStorage(void) {
        // T_bc, T_wb_1, T_wb_2, ... , T_wb_n, p_w, r
        // observe1, observe2, ... , observe_n, damp
        /*
        [  Jex   Jp1    0     0     0   |  Jl1  |  r1   ]
        [  Jex    0    Jp2    0     0   |  Jl2  |  r2   ]
        [  Jex    0     0    Jp3    0   |  Jl3  |  r3   ]
        [  Jex    0     0     0    Jp4  |  Jl4  |  r4   ]
        [   0     0     0     0     0   |   0   |   0   ] * 3
        */
        size_t cameraNums = static_cast<size_t>(this->observes.size());
        size_t cols = 6 + cameraNums * 6 + 3 + 1;
        size_t rows = cameraNums * 2 + 3;
        this->storage.setZero(rows, cols);
        // std::cout << "LandmarkBlockPos<Scalar>::ResizeStorage :\n" << this->storage << std::endl;
        return rows - 3;
    }


    /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，暂时不填入到 storage 中 */
    /* 此时 storate 矩阵保持不变 */
    template<typename Scalar>
    bool LandmarkBlockPos<Scalar>::PrecomputeResidual(void) {
        this->residuals.clear();
        for (auto &observe : this->observes) {
            // 提取相关信息
            size_t cameraID = observe.first;
            auto &camera = observe.second->camera;
            Vector2<Scalar> &measure = observe.second->norm;
            auto &kernel = observe.second->kernel;
            Matrix2<Scalar> &sqrtInfo = observe.second->sqrtInfo;

            // 提取位姿参数
            Quaternion<Scalar> &q_wb = camera->Get_q_wb();
            Vector3<Scalar> &t_wb = camera->Get_t_wb();
            Quaternion<Scalar> &q_bc = this->exPose->Get_q_bc();
            Vector3<Scalar> &t_bc = this->exPose->Get_t_bc();
            Vector3<Scalar> &p_w = this->landmark->Get_p_w();

            // 计算重投影的归一化平面坐标 predict = [ux, uy, 1].T
            Vector3<Scalar> p_b = q_wb.inverse() * (p_w - t_wb);
            Vector3<Scalar> p_c = q_bc.inverse() * (p_b - t_bc);
            Scalar invDep = 1.0 / p_c.z();
            Vector2<Scalar> predict = p_c.template head<2>() / p_c.z();

            // 计算重投影误差
            Vector2<Scalar> residual;
            if (std::isnan(invDep) || std::isinf(invDep)) {
                residual.setZero();
            } else {
                residual = predict - measure;
            }

            // 计算 rho(r.T * S * r) 即 y，以及鲁棒核权重 rho'(r.T * S * r) 即 y_
            Scalar x = residual.transpose() * sqrtInfo * sqrtInfo * residual;
            kernel->Compute(x);

            // 计算 sqrt(S) * r，作为误差输出结果
            Vector2<Scalar> weightedResidual = std::sqrt(kernel->y_) * sqrtInfo * residual;
            this->residuals.insert(std::make_pair(cameraID, weightedResidual));
        }
        return true;
    }


    /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，填入到 storage 中 */
    /* 此时填充了 storage 的一部分：
        [ 0 | 0 | r ]
        [ 0 | 0 | 0 ] */
    template<typename Scalar>
    bool LandmarkBlockPos<Scalar>::ComputeResidual(void) {
        // 如果还没有预计算过误差，则需要预计算，然后才能进行填充
        if (this->residuals.empty()) {
            this->PrecomputeResidual();
        }
        for (auto &residual : this->residuals) {
            size_t idx = (*this->cameraID_idx.find(residual.first)).second;
            this->storage.template block(idx * 2, this->storage.cols() - 1, 2, 1) = residual.second;
        }
        this->residuals.clear();
        // std::cout << "LandmarkBlockPos<Scalar>::ComputeResidual :\n" << this->storage << std::endl;
        return true;
    }

    
    /* 计算雅可比矩阵，填充 storage，完成线性化操作 */
    /* 此时填充完成 storage 矩阵:
        [ Jp(format diag block matrix) | Jl | r ]
        [     0                        | 0  | 0 ] */
    template<typename Scalar>
    bool LandmarkBlockPos<Scalar>::ComputeJacobian(void) {
        for (auto &observe : this->observes) {
            // 提取相关信息
            auto &camera = observe.second->camera;
            Vector2<Scalar> &measure = observe.second->norm;
            auto &kernel = observe.second->kernel;
            Matrix2<Scalar> &sqrtInfo = observe.second->sqrtInfo;

            // 提取位姿参数
            Quaternion<Scalar> &q_wb = camera->Get_q_wb();
            Matrix3<Scalar> R_bw(q_wb.inverse());
            Vector3<Scalar> &t_wb = camera->Get_t_wb();
            Quaternion<Scalar> &q_bc = this->exPose->Get_q_bc();
            Matrix3<Scalar> R_cb(q_bc.inverse());
            Vector3<Scalar> &t_bc = this->exPose->Get_t_bc();
            Vector3<Scalar> &p_w = this->landmark->Get_p_w();

            // 计算重投影的归一化平面坐标 norm = [ux, uy, 1].T
            Vector3<Scalar> p_b = R_bw * (p_w - t_wb);
            Vector3<Scalar> p_c = R_cb * (p_b - t_bc);
            Scalar dep = p_c.z();
            // dep 小于 0.05 时，雅可比归零，或者给很大的残差 //

            // 计算雅可比矩阵 d(归一化平面位置) / d(相机坐标系位置)
            Eigen::Matrix<Scalar, 2, 3> reduce;
            if (std::isnan(1.0 / dep) || std::isinf(1.0 / dep)) {
                reduce.setZero();
            } else {
                reduce << 1. / dep, 0, - p_c(0) / (dep * dep),
                          0, 1. / dep, - p_c(1) / (dep * dep);
            }

            // 计算雅可比矩阵 d(相机坐标位置误差) / d(相机 6 自由度位姿)
            Eigen::Matrix<Scalar, 3, 6> tempJp = Eigen::Matrix<Scalar, 3, 6>::Zero();
            if (camera->IsFixed() == false) {
                tempJp.template block<3, 3>(0, 0) = - R_cb * R_bw;
                tempJp.template block<3, 3>(0, 3) = R_cb * SkewSymmetricMatrix(p_b);
            }

            // 计算雅可比矩阵 d(相机坐标位置误差) / d(特征点 3 自由度位置)
            Eigen::Matrix<Scalar, 3, 3> tempJl = Eigen::Matrix<Scalar, 3, 3>::Zero();
            if (this->landmark->IsFixed() == false) {
                tempJl = R_cb * R_bw;
            }

            // 计算雅可比矩阵 d(相机坐标位置误差) / d(相机外参 6 自由度位姿)
            Eigen::Matrix<Scalar, 3, 6> tempJex = Eigen::Matrix<Scalar, 3, 6>::Zero();
            if (this->exPose->IsFixed() == false) {
                tempJex.template block<3, 3>(0, 0) = - R_cb;
                tempJex.template block<3, 3>(0, 3) = SkewSymmetricMatrix(p_c);
            }

            // 通过级联求导的方式，计算雅可比矩阵，填充 storage 矩阵
            size_t idx = (*this->cameraID_idx.find(observe.first)).second;
            Scalar w = std::sqrt(kernel->y_);
            this->storage.template block<2, 6>(idx * 2, 0)
                = w * sqrtInfo * reduce * tempJex;
            this->storage.template block<2, 3>(idx * 2, this->storage.cols() - 3 - 1)
                = w * sqrtInfo * reduce * tempJl;
            this->storage.template block<2, 6>(idx * 2, 6 + idx * 6)
                = w * sqrtInfo * reduce * tempJp;
        }
        // std::cout << "LandmarkBlockPos<Scalar>::ComputeJacobian :\n" << this->storage << std::endl;
        return true;
    }


    /* 对 storage 中的 Jl 进行 QR 分解，基于分解结果调整 storage 矩阵 */
    /* 此时基于 QR 分解调整后的 storage 矩阵：
        [ Q1.T * Jp |   R1   | Q1.T * r ]
        [ Q2.T * Jp | R2(=0) | Q2.T * r ]
        [     0     |    0   |     0    ] */
    template<typename Scalar>
    bool LandmarkBlockPos<Scalar>::PerformQR(void) {
        // 提取 storage 矩阵中的三个部分
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        MatrixX<Scalar> Jl = this->storage.block(0, cols - 3 - 1, rows - 3, 3);
        MatrixX<Scalar> Jp = this->storage.block(0, 0, rows - 3, cols - 3 - 1);
        VectorX<Scalar> r = this->storage.block(0, cols - 1, rows - 3, 1);

        // 对 Jl 进行 QR 分解
        Eigen::HouseholderQR<MatrixX<Scalar>> qr;
        qr.compute(Jl);
        MatrixX<Scalar> Qt = qr.householderQ().transpose();

        // 对 storage 矩阵左乘 Q.T
        this->storage.block(0, cols - 3 - 1, rows - 3, 3)
            = qr.matrixQR().template triangularView<Eigen::Upper>();
        this->storage.block(0, 0, rows - 3, cols - 3 - 1) = Qt * Jp;
        this->storage.block(0, cols - 1, rows - 3, 1) = Qt * r;
        // std::cout << "LandmarkBlockPos<Scalar>::PerformQR :\n" << this->storage << std::endl;
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
    bool LandmarkBlockPos<Scalar>::Damp(Scalar lambda) {
        // 将 storage 矩阵的最下面三行置为零
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        this->storage.block(rows - 3, 0, 3, cols).setZero();

        // 在指定位置添加阻尼因子
        this->storage.block(rows - 3, cols - 3 - 1,
            3, 3).diagonal().array() = std::sqrt(lambda);
    
        // 为更正 storage 矩阵中对应于 Jl_ 位置的格式（上为上三角，下为全零），计算 Givens 矩阵
        Eigen::JacobiRotation<Scalar> gr;
        this->Q_lambda.setIdentity(rows, rows);
        for (size_t i = 0; i < 3; ++i) {
            size_t n = i + this->storage.cols() - 3 - 1;    // 列索引
            for (size_t j = 0; j <= i; ++j) {
                size_t m = rows - 3 + i - j;
                // makeGivens() 方法本质上就是传入 x1 和 x2，计算出旋转矩阵对应的 sin 和 cos
                gr.makeGivens(this->storage(i, n), this->storage(m, n));
                // 对 this->storage 矩阵进行初等变换，并将连乘结果记录下来
                this->storage.applyOnTheLeft(m, i, gr);
                this->Q_lambda.applyOnTheLeft(m, i, gr);
            }
        }

        // 运算精度会导致部分位置不为零，因此这些位置需要强制给零
        for (size_t i = 0; i < 3; ++i) {
            size_t n = i + cols - 3 - 1;
            for (size_t m = i + rows - 3; m > i; --m) {
                this->storage(m, n) = 0;
            }
        }
        // std::cout << "LandmarkBlockPos<Scalar>::Damp :\n" << this->storage << std::endl;
        return true;
    }


    /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
    /* 用于 problem 填充大雅可比矩阵 */
    template<typename Scalar>
    typename LandmarkBlockBase<Scalar>::DividedResult LandmarkBlockPos<Scalar>::GetDampedDividedStorage(void) {
        typename LandmarkBlockBase<Scalar>::DividedResult res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        res.Q2T_Jex = this->storage.block(3, 0, rows - 3, 6);
        res.Q2T_r = this->storage.block(3, cols - 1, rows - 3, 1);
        for (auto &observe : this->observes) {
            size_t idx = (*this->cameraID_idx.find(observe.first)).second;
            MatrixX<Scalar> &&Q2T_Jp = this->storage.block(3, idx * 6 + 6, rows - 3, 6);
            res.Q2T_Jps.insert(std::make_pair(observe.first, Q2T_Jp));
        }
        return res;
    }


    /* 返回 damp 之后的 storage 矩阵有效行数 */
    template<typename Scalar>
    size_t LandmarkBlockPos<Scalar>::GetDampedDividedStorageRows(void) {
        return this->storage.rows() - 3;
    }


    /* 在 storage 矩阵没有被 damp 的前提下 */
    /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
    /* 用于 problem 填充大雅可比矩阵 */
    template<typename Scalar>
    typename LandmarkBlockBase<Scalar>::DividedResult LandmarkBlockPos<Scalar>::GetUndampedDividedStorage(void) {
        typename LandmarkBlockBase<Scalar>::DividedResult res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        res.Q2T_Jex = this->storage.block(3, 0, rows - 2 * 3, 6);
        res.Q2T_r = this->storage.block(3, cols - 1, rows - 2 * 3, 1);
        for (auto &observe : this->observes) {
            size_t idx = (*this->cameraID_idx.find(observe.first)).second;
            MatrixX<Scalar> &&Q2T_Jp = this->storage.block(3, idx * 6 + 6, rows - 2 * 3, 6);
            res.Q2T_Jps.insert(std::make_pair(observe.first, Q2T_Jp));
        }
        return res;
    }


    /* 返回 damp 之后的 storage 矩阵有效行数 */
    template<typename Scalar>
    size_t LandmarkBlockPos<Scalar>::GetUndampedDividedStorageRows(void) {
        return this->storage.rows() - 2 * 3;
    }


    /* 返回 delta_Xl 增量方程的 H 矩阵 */
    /* 本质上就是从 storage 矩阵中提取出 R1 */
    template<typename Scalar>
    MatrixX<Scalar> LandmarkBlockPos<Scalar>::Get_R1(void) {
        MatrixX<Scalar> &&R1 = this->storage.block(0, this->storage.cols() - 3 - 1, 3, 3);
        return R1;
    }


    /* 返回 delta_Xl 增量方程的 b 向量 */
    /* 本质上就是从 storage 矩阵中提取出 Q1_.T * r */
    template<typename Scalar>
    VectorX<Scalar> LandmarkBlockPos<Scalar>::Get_Q1T_r(void) {
        VectorX<Scalar> &&Q1T_r = this->storage.block(0, this->storage.cols() - 1, 3, 1);
        return Q1T_r;
    }


    /* 基于已知的相机 pose 增量的结果 delta_Xp，求解此特征点 position 对应的增量 */
    /* 就是计算 delta_Xl = - R1.inv * (Q1T * r + Q1T * Jp * delta_Xp) */
    template<typename Scalar>
    VectorX<Scalar> LandmarkBlockPos<Scalar>::ComputeDeltaXl(const VectorX<Scalar> &delta_Xp) {
        VectorX<Scalar> res;
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        // 首先需要从全局的 delta_Xp 中提取出此 landmark block 对应的 delta_Xps
        VectorX<Scalar> delta_Xps;
        delta_Xps.setZero(cols - 3 - 1, 1);
        delta_Xps.head(6) = delta_Xp.head(6);
        for (auto &item : this->observes) {
            size_t idx = (*this->cameraID_idx.find(item.first)).second;
            delta_Xps.segment(idx * 6 + 6, 6)
                = delta_Xp.segment(item.first * 6 + 6, 6);
        }
        // 然后再去求解 delta_Xl
        MatrixX<Scalar> &&Q1T_Jp = this->storage.block(0, 0, 3, cols - 3 - 1);
        VectorX<Scalar> &&Q1T_r = this->storage.block(0, cols - 1, 3, 1);
        MatrixX<Scalar> &&R1 = this->storage.block(0, cols - 3 - 1, 3, 3);
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
    bool LandmarkBlockPos<Scalar>::RollBack(void) {
        size_t rows = this->storage.rows();
        size_t cols = this->storage.cols();
        this->storage = this->Q_lambda.transpose() * this->storage;
        // 运算精度导致部分为零的地方不为零，这里强制归零
        for (size_t i = 0; i < 3; ++i) {
            size_t n = i + cols - 3 - 1;
            for (size_t m = rows - 1; m > i; --m) {
                this->storage(m, n) = 0;
            }
        }
        this->storage.block(rows - 3, 0, 3, cols).setZero();
        // std::cout << "LandmarkBlockPos<Scalar>::RollBack :\n" << this->storage << std::endl;
        return true;
    }
}