#pragma once

#include <include/ba_solver/root_vio/vertex.hpp>
#include <include/ba_solver/root_vio/kernel_function.hpp>
#include <include/ba_solver/root_vio/camera_observe.hpp>
#include <include/ba_solver/root_vio/timer.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {

    // 定义 landmark block base 类，管理数据存储与相关运算方法
    template<typename Scalar>
    class LandmarkBlockBase {
    public:
        // storage 矩阵的拆分输出体
        struct DividedResult {
            MatrixX<Scalar> Q2T_Jex;
            VectorX<Scalar> Q2T_r;
            std::unordered_map<size_t, MatrixX<Scalar>> Q2T_Jps;
        };

    public:
        // 整个块的存储矩阵 [ Jp | Jl | residual ]
        MatrixX<Scalar> storage;
        // 保存用于回退的 Q_lambda 矩阵
        MatrixX<Scalar> Q_lambda;
        // 此 landmark 的首个观测帧的 ID
        size_t cameraID_i = 0;

        // 所关联的 landmark 对象
        std::shared_ptr<VertexLandmark<Scalar>> landmark;
        // 观测到此特征点的相机 pose 节点 id，相机节点的指针，以及对应的归一化平面观测结果
        std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> observes;
        // 整个 VIO 的相机与 IMU 之间的相对位姿
        std::shared_ptr<VertexExPose<Scalar>> exPose;
        // cameraID 和在 storage 矩阵中索引的映射表
        std::unordered_map<size_t, size_t> cameraID_idx;
        // 归一化平面的观测误差
        std::unordered_map<size_t, VectorX<Scalar>> residuals;
        // 每一个观测误差对应的权重，即 rho'(r.T * S * r)，会保存在 CameraObserve 中的 kernel 里面的 y_ 中

    public:
        /* 构造函数与析构函数 */
        LandmarkBlockBase();
        LandmarkBlockBase(const std::shared_ptr<VertexLandmark<Scalar>> &landmark,
                          const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
                          const std::shared_ptr<VertexExPose<Scalar>> &exPose);
        ~LandmarkBlockBase();

    public:
        /* 为此 landmark block 对象绑定 landmark */
        bool SetLandmark(const std::shared_ptr<VertexLandmark<Scalar>> &landmark);
        /* 为此 landmark block 对象绑定观测 */
        bool SetObserves(const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes);
        /* 为此 landmark block 对象绑定相机外参 */
        bool SetExPose(const std::shared_ptr<VertexExPose<Scalar>> &exPose);
        /* 提取此 landmark block 对象绑定的观测信息（可用此提取误差的马氏距离） */
        std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &GetObserves(void);
        /* 提取此 landmark block 对象绑定的 landmark */
        std::shared_ptr<VertexLandmark<Scalar>> &GetLandmark(void);
        /* 提取此 landmark block 对象绑定的相机外参 */
        std::shared_ptr<VertexExPose<Scalar>> &GetExPose(void);
        /* 提取每一帧相机观测对应的观测误差，即返回 this->residuals */
        std::unordered_map<size_t, VectorX<Scalar>> &GetResiduals(void);

    public:
        /* 重新初始化 storage 的尺寸，并清空其中的数据，返回作用于 problem 中大雅可比矩阵中的行数 */
        /* 此时得到一个确定尺寸的 0 矩阵：
            [ 0 | 0 | 0 ] */
        virtual size_t ResizeStorage(void) = 0;
        /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，暂时不填入到 storage 中 */
        /* 此时 storate 矩阵保持不变 */
        virtual bool PrecomputeResidual(void) = 0;
        /* 根据 camera 和 landmark 对象的参数与 observe，计算观测误差 r，填入到 storage 中 */
        /* 此时填充了 storage 的一部分：
            [ 0 | 0 | r ]
            [ 0 | 0 | 0 ] */
        virtual bool ComputeResidual(void) = 0;
        /* 计算雅可比矩阵，填充 storage，完成线性化操作 */
        /* 此时填充完成 storage 矩阵:
            [ Jp(format diag block matrix) | Jl | r ]
            [     0                        | 0  | 0 ] */
        virtual bool ComputeJacobian(void) = 0;
        /* 对 storage 中的 Jl 进行 QR 分解，基于分解结果调整 storage 矩阵 */
        /* 此时基于 QR 分解调整后的 storage 矩阵：
            [ Q1.T * Jp |   R1   | Q1.T * r ]
            [ Q2.T * Jp | R2(=0) | Q2.T * r ]
            [     0     |    0   |     0    ] */
        virtual bool PerformQR(void) = 0;
        /* 给 storage 矩阵添加阻尼因子，并计算 givens 矩阵使得 storage 矩阵形式上保持不变，记录下 givens 矩阵在 this->Q_lambda */
        /* 添加阻尼因子后的 storage 矩阵：
            [ Q1.T * Jp |        R1        | Q1.T * r ]
            [ Q2.T * Jp |      R2(=0)      | Q2.T * r ]
            [     0     | sqrt(lambda) * I |     0    ] */
        /* 经过 Givens 矩阵调整后的 storage 矩阵，维度和上面的矩阵保持一致：
            [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
            [ Q2_.T * Jp |   0   | Q2_.T * r ]
            [     0     |    0   |     0    ] */
        virtual bool Damp(Scalar lambda) = 0;
        /* 在 storage 矩阵被 damp 的前提下 */
        /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
        /* 用于 problem 填充大雅可比矩阵 */
        virtual DividedResult GetDampedDividedStorage(void) = 0;
        /* 返回 damp 之后的 storage 矩阵有效行数 */
        virtual size_t GetDampedDividedStorageRows(void) = 0;
        /* 在 storage 矩阵没有被 damp 的前提下 */
        /* 按照 observe 的索引顺序，提取出指定于 cameraID 的 Q2.T * Jex、Q2.T * Jp 和 Q2.T * r 矩阵块 */
        /* 用于 problem 填充大雅可比矩阵 */
        virtual DividedResult GetUndampedDividedStorage(void) = 0;
        /* 返回 damp 之后的 storage 矩阵有效行数 */
        virtual size_t GetUndampedDividedStorageRows(void) = 0;
        /* 返回 delta_Xl 增量方程的 H 矩阵 */
        /* 本质上就是从 storage 矩阵中提取出 R1 */
        virtual MatrixX<Scalar> Get_R1(void) = 0;
        /* 返回 delta_Xl 增量方程的 b 向量 */
        /* 本质上就是从 storage 矩阵中提取出 Q1_.T * r */
        virtual VectorX<Scalar> Get_Q1T_r(void) = 0;
        /* 基于已知的相机 pose 增量的结果 delta_Xp，求解此特征点 position 对应的增量 */
        /* 就是计算 delta_Xl = - R1.inv * (Q1T * r + Q1T * Jp * delta_Xp) */
        virtual VectorX<Scalar> ComputeDeltaXl(const VectorX<Scalar> &delta_Xp) = 0;
        /* 基于保存的 Q_lambda 矩阵，回退 storage 矩阵至原来的状态 */
        /* 同时，为了方便再次调用 Damp 方法，此处在回退后需要去除因 damp 而增加的空白行 */
        /* 回退之前的 storage 矩阵：
            [ Q1_.T * Jp |  R1_  | Q1_.T * r ]
            [ Q2_.T * Jp |   0   | Q2_.T * r ] */
        /* 回退之后的 storage 矩阵：
            [ Q1.T * Jp |  R1  | Q1.T * r ]
            [ Q2.T * Jp |   0  | Q2.T * r ]
            [     0     |   0  |     0    ] */
        virtual bool RollBack(void) = 0;
    };
}


// 此处为模板类的成员的定义
namespace rootVIO {
    /* 构造函数与析构函数 */
    template<typename Scalar>
    LandmarkBlockBase<Scalar>::LandmarkBlockBase() {}
    template<typename Scalar>
    LandmarkBlockBase<Scalar>::LandmarkBlockBase(const std::shared_ptr<VertexLandmark<Scalar>> &landmark,
        const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes,
        const std::shared_ptr<VertexExPose<Scalar>> &exPose) {
        this->SetLandmark(landmark);
        this->SetObserves(observes);
        this->SetExPose(exPose);
    }
    template<typename Scalar>
    LandmarkBlockBase<Scalar>::~LandmarkBlockBase() {}

    /* 为此 landmark block 对象绑定 landmark */
    template<typename Scalar>
    bool LandmarkBlockBase<Scalar>::SetLandmark(const std::shared_ptr<VertexLandmark<Scalar>> &landmark) {
        if (landmark == nullptr) {
            return false;
        } else {
            this->landmark = landmark;
            return true;
        }
    }


    /* 为此 landmark block 对象绑定观测 */
    template<typename Scalar>
    bool LandmarkBlockBase<Scalar>::SetObserves(const std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &observes) {
        if (observes.empty()) {
            return false;
        } else {
            this->observes = observes;
            this->cameraID_idx.clear();
            size_t idx = 0;
            this->cameraID_i = static_cast<size_t>(0xFFFFFFFFFFFFFFFF);
            for (auto &ob : this->observes) {
                // 记录观测到此特征点的首帧在 problem 中的 ID
                this->cameraID_i = std::min(this->cameraID_i, ob.first);
                this->cameraID_idx.insert(std::make_pair(ob.first, idx));
                // std::cout << "this->cameraID_idx.insert [ " << ob.first << ", " << idx << " ]\n";
                ++idx;
            }
            // std::cout << "this->cameraID_i is " << this->cameraID_i << std::endl;
            return true;
        }
    }


    /* 为此 landmark block 对象绑定相机外参 */
    template<typename Scalar>
    bool LandmarkBlockBase<Scalar>::SetExPose(const std::shared_ptr<VertexExPose<Scalar>> &exPose) {
        if (exPose == nullptr) {
            return false;
        } else {
            this->exPose = exPose;
            return true;
        }
    }


    /* 提取此 landmark block 对象绑定的观测信息（可用此提取误差的马氏距离） */
    template<typename Scalar>
    std::unordered_map<size_t, std::shared_ptr<CameraObserve<Scalar>>> &LandmarkBlockBase<Scalar>::GetObserves(void) {
        return this->observes;
    }


    /* 提取此 landmark block 对象绑定的 landmark */
    template<typename Scalar>
    std::shared_ptr<VertexLandmark<Scalar>> &LandmarkBlockBase<Scalar>::GetLandmark(void) {
        return this->landmark;
    }


    /* 提取此 landmark block 对象绑定的相机外参 */
    template<typename Scalar>
    std::shared_ptr<VertexExPose<Scalar>> &LandmarkBlockBase<Scalar>::GetExPose(void) {
        return this->exPose;
    }


    /* 提取每一帧相机观测对应的观测误差，即返回 this->residuals */
    template<typename Scalar>
    std::unordered_map<size_t, VectorX<Scalar>> &LandmarkBlockBase<Scalar>::GetResiduals(void) {
        return this->residuals;
    }
}