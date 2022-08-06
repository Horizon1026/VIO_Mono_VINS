#pragma once

#include <include/ba_solver/graph_optimizor/edge.hpp>
#include <include/measurement/imu_preintegration.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Edge IMU Factor

        此边为 4 元边，与之相连的顶点有：
            <VertexPose>            第 i 帧相机在 IMU 系中的位姿 R_wb_i, t_wb_i
            <VertexVelocityBias>    第 i 帧相机对应的帧速度、IMU 的加速度和角速度偏差 bias_a 和 bias_g
            <VertexPose>            第 j 帧相机在 IMU 系中的位姿 R_wb_i, t_wb_i
            <VertexVelocityBias>    第 j 帧相机对应的帧速度、IMU 的加速度和角速度偏差 bias_a 和 bias_g
    
        此边的观测为 IMU 状态量，维度为 15
    */
    template<typename Scalar>
    class EdgeIMUFactor : public EdgeBase<Scalar> {
    private:
        // 预积分块的线性化点
        struct LinearlizedPoint {
            Quaternion<Scalar> delta_r;
            Vector3<Scalar> delta_v;
            Vector3<Scalar> delta_p;
            Vector3<Scalar> bias_g;
            Vector3<Scalar> bias_a;
        } linear;
        // 对应于线性化点的雅可比矩阵
        struct IMUJacobians {
            Matrix3<Scalar> dr_dbg;
            Matrix3<Scalar> dv_dbg;
            Matrix3<Scalar> dv_dba;
            Matrix3<Scalar> dp_dbg;
            Matrix3<Scalar> dp_dba;
        } jacobians;
        // 重力加速度向量
        Vector3<Scalar> gravity_w;
        // 总积分时间
        Scalar sumTime;
        // 各状态分量的位置
        struct Order {
            size_t P;
            size_t R;
            size_t V;
            size_t BA;
            size_t BG;
        } order;
    public:
        /* 构造函数与析构函数 */
        EdgeIMUFactor(const std::shared_ptr<IMUPreintegration> &setImu,
                      const Vector3<Scalar> &gravity) :
            EdgeBase<Scalar>(15, 4, nullptr), gravity_w(gravity), sumTime(setImu->GetSumTime()) {
            Quaternion<float> delta_r;
            Vector3<float> delta_v, delta_p;
            Matrix3<float> dr_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba;
            setImu->GetDeltaRVP(delta_r, delta_v, delta_p);
            this->linear.delta_r = delta_r.template cast<Scalar>();
            this->linear.delta_p = delta_p.template cast<Scalar>();
            this->linear.delta_v = delta_v.template cast<Scalar>();
            setImu->GetJacobians(dr_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba);
            this->jacobians.dr_dbg = dr_dbg.template cast<Scalar>();
            this->jacobians.dv_dbg = dv_dbg.template cast<Scalar>();
            this->jacobians.dv_dba = dv_dba.template cast<Scalar>();
            this->jacobians.dp_dbg = dp_dbg.template cast<Scalar>();
            this->jacobians.dp_dba = dp_dba.template cast<Scalar>();
            this->linear.bias_g = setImu->GetBiasG().cast<Scalar>();
            this->linear.bias_a = setImu->GetBiasA().cast<Scalar>();
            // 设置信息矩阵（也就是协方差矩阵的逆）
            this->SetInformation(setImu->GetCovariance().inverse().cast<Scalar>());
            // 传递分量索引
            this->order.P = IMUPreintegration::Order::P;
            this->order.R = IMUPreintegration::Order::R;
            this->order.V = IMUPreintegration::Order::V;
            this->order.BA = IMUPreintegration::Order::Ba;
            this->order.BG = IMUPreintegration::Order::Bg;
        }
        ~EdgeIMUFactor() {}

    public:
        /* 计算残差 */
        virtual void ComputeResidual(void) override;
        /* 计算残差对于每一个节点参数的雅可比矩阵 */
        virtual void ComputeJacobians(void) override;

    private:
        /* 基于最新的加速度和角速度偏差，以及记录的线性化点，计算出修正的观测 */
        void CorrectObservation(Quaternion<Scalar> &delta_r,
                                Vector3<Scalar> &delta_v,
                                Vector3<Scalar> &delta_p,
                                const Vector3<Scalar> &newBias_a,
                                const Vector3<Scalar> &newBias_g);
    };


    /* 类成员方法定义如下 */
    /* 计算残差 */
    template<typename Scalar>
    void EdgeIMUFactor<Scalar>::ComputeResidual(void) {
        // 从节点中提取参数
        VectorX<Scalar> param_pose_i = this->GetVertex(0)->GetParameters();
        VectorX<Scalar> param_motion_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_pose_j = this->GetVertex(2)->GetParameters();
        VectorX<Scalar> param_motion_j = this->GetVertex(3)->GetParameters();

        // 参数整理
        Quaternion<Scalar> q_wb_i(param_pose_i[6], param_pose_i[3], param_pose_i[4], param_pose_i[5]);
        Matrix3<Scalar> R_wb_i(q_wb_i);
        Vector3<Scalar> p_wb_i = param_pose_i.template head<3>();
        Vector3<Scalar> v_wb_i = param_motion_i.template head<3>();
        Vector3<Scalar> ba_i = param_motion_i.segment(3, 3);
        Vector3<Scalar> bg_i = param_motion_i.template tail<3>();
        Quaternion<Scalar> q_wb_j(param_pose_j[6], param_pose_j[3], param_pose_j[4], param_pose_j[5]);
        Matrix3<Scalar> R_wb_j(q_wb_j);
        Vector3<Scalar> p_wb_j = param_pose_j.template head<3>();
        Vector3<Scalar> v_wb_j = param_motion_j.template head<3>();
        Vector3<Scalar> ba_j = param_motion_j.segment(3, 3);
        Vector3<Scalar> bg_j = param_motion_j.template tail<3>();
        Scalar &dt = this->sumTime;

        // 基于线性化点和改变量，计算矫正的预积分观测
        Quaternion<Scalar> delta_r;
        Vector3<Scalar> delta_v, delta_p;
        this->CorrectObservation(delta_r, delta_v, delta_p, ba_i, bg_i);

        // 计算误差，同时处理异常
        VectorX<Scalar> r(15);
        r.template block<3, 1>(this->order.P, 0) = R_wb_i.transpose() * (p_wb_j - p_wb_i - v_wb_i * dt + 0.5 * this->gravity_w * dt * dt) - delta_p;
        r.template block<3, 1>(this->order.R, 0) = 2.0 * (delta_r.inverse() * (q_wb_i.inverse() * q_wb_j)).vec();
        r.template block<3, 1>(this->order.V, 0) = R_wb_i.transpose() * (v_wb_j - v_wb_i + this->gravity_w * dt) - delta_v;
        r.template block<3, 1>(this->order.BA, 0) = ba_j - ba_i;
        r.template block<3, 1>(this->order.BG, 0) = bg_j - bg_i;

        // 更新误差
        this->SetResidual(r);
    }


    /* 计算残差对于每一个节点参数的雅可比矩阵 */
    template<typename Scalar>
    void EdgeIMUFactor<Scalar>::ComputeJacobians(void) {
        // 从节点中提取参数
        VectorX<Scalar> param_pose_i = this->GetVertex(0)->GetParameters();
        VectorX<Scalar> param_motion_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_pose_j = this->GetVertex(2)->GetParameters();
        VectorX<Scalar> param_motion_j = this->GetVertex(3)->GetParameters();

        // 参数整理
        Quaternion<Scalar> q_wb_i(param_pose_i[6], param_pose_i[3], param_pose_i[4], param_pose_i[5]);
        Matrix3<Scalar> R_wb_i(q_wb_i);
        Vector3<Scalar> p_wb_i = param_pose_i.template head<3>();
        Vector3<Scalar> v_wb_i = param_motion_i.template head<3>();
        Vector3<Scalar> ba_i = param_motion_i.segment(3, 3);
        Vector3<Scalar> bg_i = param_motion_i.template tail<3>();
        Quaternion<Scalar> q_wb_j(param_pose_j[6], param_pose_j[3], param_pose_j[4], param_pose_j[5]);
        Matrix3<Scalar> R_wb_j(q_wb_j);
        Vector3<Scalar> p_wb_j = param_pose_j.template head<3>();
        Vector3<Scalar> v_wb_j = param_motion_j.template head<3>();
        Vector3<Scalar> ba_j = param_motion_j.segment(3, 3);
        Vector3<Scalar> bg_j = param_motion_j.template tail<3>();
        Scalar &dt = this->sumTime;
        Scalar dt2 = dt * dt;
        Vector3<Scalar> tempV;
        Quaternion<Scalar> tempQ;

        // 基于线性化点和改变量，计算矫正的预积分观测
        Quaternion<Scalar> delta_r;
        Vector3<Scalar> delta_v, delta_p;
        this->CorrectObservation(delta_r, delta_v, delta_p, ba_i, bg_i);

        // 计算观测误差对于每一个状态节点的雅可比矩阵
        MatrixX<Scalar> jacobian0, jacobian1, jacobian2, jacobian3;

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 i 帧位置、姿态）的雅可比矩阵
        Eigen::Matrix<Scalar, 15, 6> dr_dpose_i;
        dr_dpose_i.setZero();
        dr_dpose_i.template block<3, 3>(this->order.P, this->order.P) =
            - R_wb_i.transpose();
        dr_dpose_i.template block<3, 3>(this->order.P, this->order.R) =
            SkewSymmetricMatrix(q_wb_i.inverse() * (0.5 * this->gravity_w * dt2 + p_wb_j - p_wb_i - v_wb_i * dt));
        dr_dpose_i.template block<3, 3>(this->order.R, this->order.R) =
            - (Qleft(q_wb_j.inverse() * q_wb_i) * Qright(delta_r)).template bottomRightCorner<3, 3>();
        dr_dpose_i.template block<3, 3>(this->order.V, this->order.R) =
            SkewSymmetricMatrix(q_wb_i.inverse() * (this->gravity_w * dt + v_wb_j - v_wb_i));

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 i 帧速度、加速度偏差、角速度偏差）的雅可比矩阵
        Eigen::Matrix<Scalar, 15, 9> dr_dmotion_i;
        dr_dmotion_i.setZero();
        dr_dmotion_i.template block<3, 3>(this->order.P, this->order.V - this->order.V) =
            - R_wb_i.transpose() * dt;
        dr_dmotion_i.template block<3, 3>(this->order.P, this->order.BA - this->order.V) =
            - this->jacobians.dp_dba;
        dr_dmotion_i.template block<3, 3>(this->order.P, this->order.BG - this->order.V) =
            - this->jacobians.dp_dbg;
        dr_dmotion_i.template block<3, 3>(this->order.R, this->order.BG - this->order.V) =
            - Qleft(q_wb_j.inverse() * q_wb_i * delta_r).template bottomRightCorner<3, 3>() * this->jacobians.dr_dbg;
        dr_dmotion_i.template block<3, 3>(this->order.V, this->order.V - this->order.V) =
            - R_wb_i.transpose();
        dr_dmotion_i.template block<3, 3>(this->order.V, this->order.BA - this->order.V) =
            - this->jacobians.dv_dba;
        dr_dmotion_i.template block<3, 3>(this->order.V, this->order.BG - this->order.V) =
            - this->jacobians.dv_dbg;
        dr_dmotion_i.template block<3, 3>(this->order.BA, this->order.BA - this->order.V) =
            - Matrix3<Scalar>::Identity();
        dr_dmotion_i.template block<3, 3>(this->order.BG, this->order.BG - this->order.V) =
            - Matrix3<Scalar>::Identity();

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 j 帧位置、姿态）的雅可比矩阵
        Eigen::Matrix<Scalar, 15, 6> dr_dpose_j;
        dr_dpose_j.setZero();
        dr_dpose_j.template block<3, 3>(this->order.P, this->order.P) =
            R_wb_i.transpose();
        dr_dpose_j.template block<3, 3>(this->order.R, this->order.R) =
            Qleft(delta_r.inverse() * q_wb_i.inverse() * q_wb_j).template bottomRightCorner<3, 3>();

        // 计算（位置、姿态、速度、加速度偏差、角速度偏差）对（第 j 帧速度、加速度偏差、角速度偏差）的雅可比矩阵
        Eigen::Matrix<Scalar, 15, 9> dr_dmotion_j;
        dr_dmotion_j.setZero();
        dr_dmotion_j.template block<3, 3>(this->order.V, this->order.V - this->order.V) =
            R_wb_i.transpose();
        dr_dmotion_j.template block<3, 3>(this->order.BA, this->order.BA - this->order.V) =
            Matrix3<Scalar>::Identity();
        dr_dmotion_j.template block<3, 3>(this->order.BG, this->order.BG - this->order.V) =
            Matrix3<Scalar>::Identity();

        // 利用级联偏导
        jacobian0 = dr_dpose_i;
        jacobian1 = dr_dmotion_i;
        jacobian2 = dr_dpose_j;
        jacobian3 = dr_dmotion_j;
        this->SetJacobian(0, jacobian0);
        this->SetJacobian(1, jacobian1);
        this->SetJacobian(2, jacobian2);
        this->SetJacobian(3, jacobian3);
    }


    /* 基于最新的加速度和角速度偏差，以及记录的线性化点，计算出修正的观测 */
    template<typename Scalar>
    void EdgeIMUFactor<Scalar>::CorrectObservation(Quaternion<Scalar> &delta_r,
                                                   Vector3<Scalar> &delta_v,
                                                   Vector3<Scalar> &delta_p,
                                                   const Vector3<Scalar> &newBias_a,
                                                   const Vector3<Scalar> &newBias_g) {
        Vector3<Scalar> delta_ba = newBias_a - this->linear.bias_a;
        Vector3<Scalar> delta_bg = newBias_g - this->linear.bias_g;
        delta_r = this->linear.delta_r * DeltaQ(this->jacobians.dr_dbg * delta_bg);
        delta_v = this->linear.delta_v + this->jacobians.dv_dba * delta_ba + this->jacobians.dv_dbg * delta_bg;
        delta_p = this->linear.delta_p + this->jacobians.dp_dba * delta_ba + this->jacobians.dp_dbg * delta_bg; 
    }
}