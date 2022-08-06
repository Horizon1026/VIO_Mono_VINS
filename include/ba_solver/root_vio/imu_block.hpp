#pragma once

#include <include/ba_solver/root_vio/vertex.hpp>
#include <include/ba_solver/root_vio/kernel_function.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {
    // 定义 IMU 约束信息管理类，管理相邻两帧之间的 IMU 约束
    template<typename Scalar>
    class IMUBlock {
    public:
        // 预积分值的线性化点
        struct LinearlizedPoint {
            Quaternion<Scalar> delta_r;
            Vector3<Scalar> delta_v;
            Vector3<Scalar> delta_p;
            Vector3<Scalar> bias_a;
            Vector3<Scalar> bias_g;
        };
        // 预积分值的矫正观测结果
        struct CorrectedObserve {
            Quaternion<Scalar> delta_r;
            Vector3<Scalar> delta_v;
            Vector3<Scalar> delta_p;
        };
        // 预积分值的雅可比矩阵
        struct IMUJacobians {
            Matrix3<Scalar> dr_dbg;
            Matrix3<Scalar> dv_dbg;
            Matrix3<Scalar> dv_dba;
            Matrix3<Scalar> dp_dbg;
            Matrix3<Scalar> dp_dba;
        };
        // 各状态分量的位置
        struct Order {
            size_t P;
            size_t R;
            size_t V;
            size_t BA;
            size_t BG;
        };
        // IMU 块的误差与雅可比输出
        struct Result {
            Vector15<Scalar> residual;
            Scalar rT_S_r;
            Eigen::Matrix<Scalar, 15, 6> dr_dpose_i;
            Eigen::Matrix<Scalar, 15, 9> dr_dvb_i;
            Eigen::Matrix<Scalar, 15, 6> dr_dpose_j;
            Eigen::Matrix<Scalar, 15, 9> dr_dvb_j;
            size_t cameraID_j;
        } result;

    private:
        // 相关的相机 Pose 节点和 velocityBias 节点
        std::vector<std::shared_ptr<VertexCameraPose<Scalar>>> cameras;
        std::vector<std::shared_ptr<VertexVelocityBias<Scalar>>> velocityBiases;
        // 重力加速度向量
        Vector3<Scalar> g_w;
        // 总积分时间
        Scalar sumTime;
        // 协方差矩阵的逆，以及平方根
        Matrix15<Scalar> information;
        Matrix15<Scalar> sqrtInfo;
        // 预积分值的线性化点
        LinearlizedPoint linear;
        // 预积分值的矫正观测结果
        CorrectedObserve observe;
        // 预积分值的雅可比矩阵
        IMUJacobians jacobians;
        // 各状态分量的位置
        Order order;

    public:
        /* 构造函数与析构函数 */
        IMUBlock();
        IMUBlock(const std::vector<std::shared_ptr<VertexCameraPose<Scalar>>> &cameras,
                 const std::vector<std::shared_ptr<VertexVelocityBias<Scalar>>> &velocityBiases,
                 const size_t cameraID_j,
                 const LinearlizedPoint &linear,
                 const IMUJacobians &jacobians,
                 const Vector3<Scalar> &g_w,
                 const Scalar sumTime,
                 const Order &order,
                 const Matrix15<Scalar> &covariance);
        ~IMUBlock();
        
    public:
        /* 为此 imu block 对象绑定 camera pose 节点 */
        bool SetCameraPose(const std::vector<std::shared_ptr<VertexCameraPose<Scalar>>> &cameras,
            const size_t cameraID_j);
        /* 为此 imu block 对象绑定 velocity bias 节点 */
        bool SetVelocityBias(const std::vector<std::shared_ptr<VertexVelocityBias<Scalar>>> &velocityBiases);
        /* 为此 imu block 对象绑定对应的 IMU 预积分块的预积分值和 bias，以此线性化点 */
        bool SetLinearlizePoint(const LinearlizedPoint &linear);
        /* 为此 imu block 对象绑定对应的 IMU 预积分块的积分时间 */
        bool SetIntegratedTime(const Scalar sumTime);
        /* 为此 imu block 对象绑定对应的 IMU 预积分块的线性化雅可比矩阵 */
        bool SetJacobians(const IMUJacobians &jacobians);
        /* 为此 imu block 对象绑定 w 系下的重力加速度向量 */
        bool SetGravityOnWorldFrame(const Vector3<Scalar> &g_w);
        /* 为此 imu block 对象绑定各分量对应的位置 */
        bool SetOrder(const Order &order);
        /* 为此 imu block 对象设置协方差矩阵，并计算信息矩阵以及其平方根 */
        bool SetCovariance(const Matrix15<Scalar> &covariance);
        /* 从此 imu block 中提取出信息矩阵 */
        Matrix15<Scalar> &GetInformation(void);
        /* 基于最新的加速度和角速度偏差，以及记录的线性化点，计算出修正的观测 this->observe */
        void CorrectObservation(const Vector3<Scalar> &newBias_a,
                                const Vector3<Scalar> &newBias_g);
        /* 计算误差向量 */
        void ComputeResidual(void);
        /* 计算误差向量的马氏距离二范数，即返回 r.T * S * r 的结果 */
        Scalar Get_rT_S_r(void);
        /* 计算雅可比矩阵 */
        void ComputeJacobians(void);
        /* 从 imu block 中提取出误差和雅可比计算结果 */
        Result &GetResidualJacobians(void);
    };
}

// 此处为模板类的成员的定义
namespace rootVIO {
    /* 构造函数与析构函数 */
    template<typename Scalar>
    IMUBlock<Scalar>::IMUBlock() {}
    template<typename Scalar>
    IMUBlock<Scalar>::IMUBlock(const std::vector<std::shared_ptr<VertexCameraPose<Scalar>>> &cameras,
        const std::vector<std::shared_ptr<VertexVelocityBias<Scalar>>> &velocityBiases,
        const size_t cameraID_j,
        const LinearlizedPoint &linear,
        const IMUJacobians &jacobians,
        const Vector3<Scalar> &g_w,
        const Scalar sumTime,
        const Order &order,
        const Matrix15<Scalar> &covariance) {
        this->SetCameraPose(cameras, cameraID_j);
        this->SetVelocityBias(velocityBiases);
        this->SetLinearlizePoint(linear);
        this->SetJacobians(jacobians);
        this->SetGravityOnWorldFrame(g_w);
        this->SetIntegratedTime(sumTime);
        this->SetOrder(order);
        this->SetCovariance(covariance);
    }
    template<typename Scalar>
    IMUBlock<Scalar>::~IMUBlock() {}


    /* 为此 imu block 对象绑定 camera pose 节点 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetCameraPose(const std::vector<std::shared_ptr<VertexCameraPose<Scalar>>> &cameras,
        const size_t cameraID_j) {
        if (cameras.size() != 2) {
            return false;
        }
        if (cameras[0] == cameras[1]) {
            return false;
        }
        this->cameras = cameras;
        this->result.cameraID_j = cameraID_j;
        return true;
    }


    /* 为此 imu block 对象绑定 velocity bias 节点 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetVelocityBias(const std::vector<std::shared_ptr<VertexVelocityBias<Scalar>>> &velocityBiases) {
        if (velocityBiases.size() != 2) {
            return false;
        }
        if (velocityBiases[0] == velocityBiases[1]) {
            return false;
        }
        this->velocityBiases = velocityBiases;
        return true;
    }


    /* 为此 imu block 对象绑定对应的 IMU 预积分块的预积分值和 bias，以此线性化点 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetLinearlizePoint(const LinearlizedPoint &linear) {
        this->linear = linear;
        this->observe.delta_p = this->linear.delta_p;
        this->observe.delta_r = this->linear.delta_r;
        this->observe.delta_v = this->linear.delta_v;
        return true;
    }


    /* 为此 imu block 对象绑定对应的 IMU 预积分块的积分时间 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetIntegratedTime(const Scalar sumTime) {
        if (sumTime < 0) {
            return false;
        }
        this->sumTime = sumTime;
        return true;
    }


    /* 为此 imu block 对象绑定对应的 IMU 预积分块的线性化雅可比矩阵 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetJacobians(const IMUJacobians &jacobians) {
        this->jacobians = jacobians;
        return true;
    }


    /* 为此 imu block 对象绑定 w 系下的重力加速度向量 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetGravityOnWorldFrame(const Vector3<Scalar> &g_w) {
        this->g_w = g_w;
        return true;
    }


    /* 为此 imu block 对象绑定各分量对应的位置 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetOrder(const Order &order) {
        this->order = order;
        return true;
    }


    /* 为此 imu block 对象设置协方差矩阵，并计算信息矩阵以及其平方根 */
    template<typename Scalar>
    bool IMUBlock<Scalar>::SetCovariance(const Matrix15<Scalar> &covariance) {
        this->information = covariance.inverse();
        Eigen::SelfAdjointEigenSolver<Matrix15<Scalar>> saes(this->information);
        Vector15<Scalar> S = Vector15<Scalar>((saes.eigenvalues().array() > Scalar(1e-8)).select(saes.eigenvalues().array(), 0));
        Vector15<Scalar> sqrtS = S.cwiseSqrt();
        this->sqrtInfo = saes.eigenvectors() * sqrtS.asDiagonal() * saes.eigenvectors().transpose();

        // std::cout << "covariance is\n" << covariance << std::endl;
        // std::cout << "this->information is\n" << this->information << std::endl;
        // std::cout << "this->sqrtInfo is\n" << this->sqrtInfo << std::endl;
        // std::cout << "this->sqrtInfo ^ 2 is\n" << this->sqrtInfo * this->sqrtInfo << std::endl;
        return true;
    }


    /* 从此 imu block 中提取出信息矩阵 */
    template<typename Scalar>
    Matrix15<Scalar> &IMUBlock<Scalar>::GetInformation(void) {
        return this->information;
    }


    /* 基于最新的加速度和角速度偏差，以及记录的线性化点，计算出修正的观测 this->observe */
    template<typename Scalar>
    void IMUBlock<Scalar>::CorrectObservation(const Vector3<Scalar> &newBias_a,
                                              const Vector3<Scalar> &newBias_g) {
        Vector3<Scalar> delta_ba = newBias_a - this->linear.bias_a;
        Vector3<Scalar> delta_bg = newBias_g - this->linear.bias_g;
        this->observe.delta_r = this->linear.delta_r * DeltaQ(this->jacobians.dr_dbg * delta_bg);
        this->observe.delta_v = this->linear.delta_v + this->jacobians.dv_dba * delta_ba + this->jacobians.dv_dbg * delta_bg;
        this->observe.delta_p = this->linear.delta_p + this->jacobians.dp_dba * delta_ba + this->jacobians.dp_dbg * delta_bg; 
    }
                        

    /* 计算误差向量 */
    template<typename Scalar>
    void IMUBlock<Scalar>::ComputeResidual(void) {
        // 提取参数
        Quaternion<Scalar> &q_wb_i = this->cameras[0]->Get_q_wb();
        Matrix3<Scalar> R_wb_i(q_wb_i);
        Vector3<Scalar> &p_wb_i = this->cameras[0]->Get_t_wb();
        Vector3<Scalar> &v_wb_i = this->velocityBiases[0]->Get_v_wb();
        Vector3<Scalar> &ba_i = this->velocityBiases[0]->Get_bias_a();
        Vector3<Scalar> &bg_i = this->velocityBiases[0]->Get_bias_g();
        Quaternion<Scalar> &q_wb_j = this->cameras[1]->Get_q_wb();
        Matrix3<Scalar> R_wb_j(q_wb_j);
        Vector3<Scalar> &p_wb_j = this->cameras[1]->Get_t_wb();
        Vector3<Scalar> &v_wb_j = this->velocityBiases[1]->Get_v_wb();
        Vector3<Scalar> &ba_j = this->velocityBiases[1]->Get_bias_a();
        Vector3<Scalar> &bg_j = this->velocityBiases[1]->Get_bias_g();
        Scalar &dt = this->sumTime;

        // 基于线性化点和 bias 的改变量，计算矫正后的预积分观测
        this->CorrectObservation(ba_i, bg_i);
    
        // 计算误差，同时计算加权的误差马氏距离二范数
        this->result.residual.template block<3, 1>(this->order.P, 0) = R_wb_i.transpose() * (p_wb_j - p_wb_i - v_wb_i * dt + 0.5 * this->g_w * dt * dt) - this->observe.delta_p;
        this->result.residual.template block<3, 1>(this->order.R, 0) = 2.0 * (this->observe.delta_r.inverse() * (q_wb_i.inverse() * q_wb_j)).vec();
        this->result.residual.template block<3, 1>(this->order.V, 0) = R_wb_i.transpose() * (v_wb_j - v_wb_i + this->g_w * dt) - this->observe.delta_v;
        this->result.residual.template block<3, 1>(this->order.BA, 0) = ba_j - ba_i;
        this->result.residual.template block<3, 1>(this->order.BG, 0) = bg_j - bg_i;
        this->result.rT_S_r = this->result.residual.transpose() * this->information * this->result.residual;
        this->result.residual = this->sqrtInfo * this->result.residual;
    }


    /* 计算误差向量的马氏距离二范数，即返回 r.T * S * r 的结果 */
    template<typename Scalar>
    Scalar IMUBlock<Scalar>::Get_rT_S_r(void) {
        return this->result.rT_S_r;
    }


    /* 计算雅可比矩阵 */
    template<typename Scalar>
    void IMUBlock<Scalar>::ComputeJacobians(void) {
        // 提取参数
        Quaternion<Scalar> &q_wb_i = this->cameras[0]->Get_q_wb();
        Matrix3<Scalar> R_wb_i(q_wb_i);
        Vector3<Scalar> &p_wb_i = this->cameras[0]->Get_t_wb();
        Vector3<Scalar> &v_wb_i = this->velocityBiases[0]->Get_v_wb();
        Vector3<Scalar> &ba_i = this->velocityBiases[0]->Get_bias_a();
        Vector3<Scalar> &bg_i = this->velocityBiases[0]->Get_bias_g();
        Quaternion<Scalar> &q_wb_j = this->cameras[1]->Get_q_wb();
        Matrix3<Scalar> R_wb_j(q_wb_j);
        Vector3<Scalar> &p_wb_j = this->cameras[1]->Get_t_wb();
        Vector3<Scalar> &v_wb_j = this->velocityBiases[1]->Get_v_wb();
        Vector3<Scalar> &ba_j = this->velocityBiases[1]->Get_bias_a();
        Vector3<Scalar> &bg_j = this->velocityBiases[1]->Get_bias_g();
        Scalar &dt = this->sumTime;
        Scalar dt2 = dt * dt;
        Vector3<Scalar> tempV;
        Quaternion<Scalar> tempQ;

        // 矫正后的 IMU 预积分值已经保存在了 this->observe 中，此处不用重新矫正
        this->CorrectObservation(ba_i, bg_i);

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 i 帧位置、姿态）的雅可比矩阵
        this->result.dr_dpose_i.setZero();
        if (this->cameras[0]->IsFixed() == false) {
            this->result.dr_dpose_i.template block<3, 3>(this->order.P, this->order.P) =
                - R_wb_i.transpose();
            this->result.dr_dpose_i.template block<3, 3>(this->order.P, this->order.R) =
                SkewSymmetricMatrix(q_wb_i.inverse() * (0.5 * this->g_w * dt2 + p_wb_j - p_wb_i - v_wb_i * dt));
            this->result.dr_dpose_i.template block<3, 3>(this->order.R, this->order.R) =
                - (Qleft(q_wb_j.inverse() * q_wb_i) * Qright(this->observe.delta_r)).template bottomRightCorner<3, 3>();
            this->result.dr_dpose_i.template block<3, 3>(this->order.V, this->order.R) =
                SkewSymmetricMatrix(q_wb_i.inverse() * (this->g_w * dt + v_wb_j - v_wb_i));
        }

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 i 帧速度、加速度偏差、角速度偏差）的雅可比矩阵
        this->result.dr_dvb_i.setZero();
        if (this->velocityBiases[0]->IsFixed() == false) {
            this->result.dr_dvb_i.template block<3, 3>(this->order.P, this->order.V - this->order.V) =
                - R_wb_i.transpose() * dt;
            this->result.dr_dvb_i.template block<3, 3>(this->order.P, this->order.BA - this->order.V) =
                - this->jacobians.dp_dba;
            this->result.dr_dvb_i.template block<3, 3>(this->order.P, this->order.BG - this->order.V) =
                - this->jacobians.dp_dbg;
            this->result.dr_dvb_i.template block<3, 3>(this->order.R, this->order.BG - this->order.V) =
                - Qleft(q_wb_j.inverse() * q_wb_i * this->observe.delta_r).template bottomRightCorner<3, 3>() * this->jacobians.dr_dbg;
            this->result.dr_dvb_i.template block<3, 3>(this->order.V, this->order.V - this->order.V) =
                - R_wb_i.transpose();
            this->result.dr_dvb_i.template block<3, 3>(this->order.V, this->order.BA - this->order.V) =
                - this->jacobians.dv_dba;
            this->result.dr_dvb_i.template block<3, 3>(this->order.V, this->order.BG - this->order.V) =
                - this->jacobians.dv_dbg;
            this->result.dr_dvb_i.template block<3, 3>(this->order.BA, this->order.BA - this->order.V) =
                - Matrix3<Scalar>::Identity();
            this->result.dr_dvb_i.template block<3, 3>(this->order.BG, this->order.BG - this->order.V) =
                - Matrix3<Scalar>::Identity();
        }

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 j 帧位置、姿态）的雅可比矩阵
        this->result.dr_dpose_j.setZero();
        if (this->cameras[1]->IsFixed() == false) {
            this->result.dr_dpose_j.template block<3, 3>(this->order.P, this->order.P) =
                R_wb_i.transpose();
            this->result.dr_dpose_j.template block<3, 3>(this->order.R, this->order.R) =
                Qleft(this->observe.delta_r.inverse() * q_wb_i.inverse() * q_wb_j).template bottomRightCorner<3, 3>();
        }

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 j 帧速度、加速度偏差、角速度偏差）的雅可比矩阵
        this->result.dr_dvb_j.setZero();
        if (this->velocityBiases[1]->IsFixed() == false) {
            this->result.dr_dvb_j.template block<3, 3>(this->order.V, this->order.V - this->order.V) =
                R_wb_i.transpose();
            this->result.dr_dvb_j.template block<3, 3>(this->order.BA, this->order.BA - this->order.V) =
                Matrix3<Scalar>::Identity();
            this->result.dr_dvb_j.template block<3, 3>(this->order.BG, this->order.BG - this->order.V) =
                Matrix3<Scalar>::Identity();
        }

        this->result.dr_dpose_i = this->sqrtInfo * this->result.dr_dpose_i;
        this->result.dr_dvb_i = this->sqrtInfo * this->result.dr_dvb_i;
        this->result.dr_dpose_j = this->sqrtInfo * this->result.dr_dpose_j;
        this->result.dr_dvb_j = this->sqrtInfo * this->result.dr_dvb_j;
    }


    /* 从 imu block 中提取出误差和雅可比计算结果 */
    template<typename Scalar>
    typename IMUBlock<Scalar>::Result &IMUBlock<Scalar>::GetResidualJacobians(void) {
        return this->result;
    }
}