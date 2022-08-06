#pragma once

#include <include/ba_solver/graph_optimizor/edge.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Edge Camera Factor

        此边为 4 元边，与之相连的顶点有：
            <VertexLandmarkInvDepth>    landmark 第一次被观测到的逆深度 invDepth
            <VertexPose>                第一次观测到此路标点的相机的位姿 R_wb_i, t_wb_i
            <VertexPose>                非第一次观测到此路标点的相机的位姿 R_wb_j, t_wb_j
            <VertexPose>                相机和 IMU 之间的相对位姿 R_bc, t_bc
    
        此边的观测为归一化平面坐标，其维度为 2
    */
    template<typename Scalar>
    class EdgeCameraFactor : public EdgeBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        EdgeCameraFactor(const Vector2<Scalar> &normPoint_i, const Vector2<Scalar> &normPoint_j) :
            EdgeBase<Scalar>(2, 4, nullptr) {
            Eigen::Matrix<Scalar, 4, 1> observe;
            observe.head(2) = normPoint_i;
            observe.tail(2) = normPoint_j;
            this->SetObservation(observe);
        }
        ~EdgeCameraFactor() {}

    public:
        /* 计算残差 */
        virtual void ComputeResidual(void) override;
        /* 计算残差对于每一个节点参数的雅可比矩阵 */
        virtual void ComputeJacobians(void) override;
    };


    /* 类成员方法定义如下 */
    /* 计算残差 */
    template<typename Scalar>
    void EdgeCameraFactor<Scalar>::ComputeResidual(void) {
        // 从节点中提取参数
        Scalar invdep_i = this->GetVertex(0)->GetParameters()(0);
        VectorX<Scalar> param_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_j = this->GetVertex(2)->GetParameters();
        VectorX<Scalar> param_bc = this->GetVertex(3)->GetParameters();
        Quaternion<Scalar> q_wb_i(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vector3<Scalar> t_wb_i = param_i.template head<3>();
        Quaternion<Scalar> q_wb_j(param_j[6], param_j[3], param_j[4], param_j[5]);
        Vector3<Scalar> t_wb_j = param_j.template head<3>();
        Quaternion<Scalar> q_bc(param_bc[6], param_bc[3], param_bc[4], param_bc[5]);
        Vector3<Scalar> t_bc = param_bc.template head<3>();
        Vector2<Scalar> norm_i = this->GetObservation().template head<2>();
        Vector2<Scalar> norm_j = this->GetObservation().template tail<2>();
        
        // 计算投影过程
        Vector3<Scalar> p_c_i = Vector3<Scalar>(norm_i(0), norm_i(1), Scalar(1)) / invdep_i;
        Vector3<Scalar> p_b_i = q_bc * p_c_i + t_bc;
        Vector3<Scalar> p_w = q_wb_i * p_b_i + t_wb_i;
        Vector3<Scalar> p_b_j = q_wb_j.inverse() * (p_w - t_wb_j);
        Vector3<Scalar> p_c_j = q_bc.inverse() * (p_b_j - t_bc);
        Scalar invdep_j = Scalar(1) / p_c_j.z();

        // 计算重投影误差，同时处理异常
        Vector2<Scalar> r;
        if (std::isnan(invdep_i) || std::isinf(invdep_i) || std::isnan(invdep_j) || std::isinf(invdep_j)) {
            r.setZero();
        } else {
            r = (p_c_j * invdep_j).template head<2>() - norm_j;
        }
        this->SetResidual(r);
    }


    /* 计算残差对于每一个节点参数的雅可比矩阵 */
    template<typename Scalar>
    void EdgeCameraFactor<Scalar>::ComputeJacobians(void) {
        // 从节点中提取参数
        Scalar invdep_i = this->GetVertex(0)->GetParameters()(0);
        VectorX<Scalar> param_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_j = this->GetVertex(2)->GetParameters();
        VectorX<Scalar> param_bc = this->GetVertex(3)->GetParameters();
        Quaternion<Scalar> q_wb_i(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vector3<Scalar> t_wb_i = param_i.template head<3>();
        Quaternion<Scalar> q_wb_j(param_j[6], param_j[3], param_j[4], param_j[5]);
        Vector3<Scalar> t_wb_j = param_j.template head<3>();
        Quaternion<Scalar> q_bc(param_bc[6], param_bc[3], param_bc[4], param_bc[5]);
        Vector3<Scalar> t_bc = param_bc.template head<3>();
        Vector2<Scalar> norm_i = this->GetObservation().template head<2>();
        Vector2<Scalar> norm_j = this->GetObservation().template tail<2>();
        
        // 计算投影过程
        Vector3<Scalar> p_c_i = Vector3<Scalar>(norm_i(0), norm_i(1), Scalar(1)) / invdep_i;
        Vector3<Scalar> p_b_i = q_bc * p_c_i + t_bc;
        Vector3<Scalar> p_w = q_wb_i * p_b_i + t_wb_i;
        Vector3<Scalar> p_b_j = q_wb_j.inverse() * (p_w - t_wb_j);
        Vector3<Scalar> p_c_j = q_bc.inverse() * (p_b_j - t_bc);
        Scalar invdep_j = Scalar(1) / p_c_j.z();

        // 计算临时变量
        Matrix3<Scalar> R_bc(q_bc);
        Matrix3<Scalar> R_wb_i(q_wb_i);
        Matrix3<Scalar> R_wb_j(q_wb_j);

        // 计算观测误差对于每一个状态节点的雅可比矩阵
        MatrixX<Scalar> jacobian0, jacobian1, jacobian2, jacobian3;
        if (std::isnan(invdep_i) || std::isinf(invdep_i) || std::isnan(invdep_j) || std::isinf(invdep_j)) {
            jacobian0 = Eigen::Matrix<Scalar, 2, 1>::Zero();
            jacobian1 = Eigen::Matrix<Scalar, 2, 6>::Zero();
            jacobian2 = Eigen::Matrix<Scalar, 2, 6>::Zero();
            jacobian3 = Eigen::Matrix<Scalar, 2, 6>::Zero();
        } else {
            // 计算（归一化二维平面）对（相机坐标系三维坐标）的雅可比矩阵
            Eigen::Matrix<Scalar, 2, 3> jacobian_norm_3d;
            jacobian_norm_3d << invdep_j, 0, - p_c_j(0) * invdep_j * invdep_j,
                                0, invdep_j, - p_c_j(1) * invdep_j * invdep_j;

            // 计算（相机坐标系三维坐标的误差）对（第 i 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> jacobian_i;
            jacobian_i.template leftCols<3>() = R_bc.transpose() * R_wb_j.transpose();
            jacobian_i.template rightCols<3>() = - R_bc.transpose() * R_wb_j.transpose() * R_wb_i * SkewSymmetricMatrix(p_b_i);

            // 计算（相机坐标系三维坐标的误差）对（第 j 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> jacobian_j;
            jacobian_j.template leftCols<3>() = - R_bc.transpose() * R_wb_j.transpose();
            jacobian_j.template rightCols<3>() = R_bc.transpose() * SkewSymmetricMatrix(p_b_j);

            // 计算（相机坐标系三维坐标的误差）对（第 i 帧中特征点逆深度）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 1> jacobian_invdep;
            Vector3<Scalar> normPoint_i(norm_i(0), norm_i(1), Scalar(1));
            jacobian_invdep = - R_bc.transpose() * R_wb_j.transpose() * R_wb_i * R_bc * normPoint_i / (invdep_i * invdep_i);

            // 计算（相机坐标系三维坐标的误差）对（IMU 和相机相对位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> jacobian_ex;
            Matrix3<Scalar> temp = R_bc.transpose() * R_wb_j.transpose() * R_wb_i * R_bc;
            jacobian_ex.template leftCols<3>() = R_bc.transpose() * (R_wb_j.transpose() * R_wb_i - Matrix3<Scalar>::Identity());
            jacobian_ex.template rightCols<3>() = - temp * SkewSymmetricMatrix(p_c_i) + SkewSymmetricMatrix(temp * p_c_i) +
                SkewSymmetricMatrix(R_bc.transpose() * (R_wb_j.transpose() * (R_wb_i * t_bc + t_wb_i - t_wb_j) - t_bc));

            // 利用级联求导，计算（归一化平面坐标误差）对（第 i 帧相机位姿）、（第 j 帧相机位姿）和（第 i 帧中特征点逆深度）的雅可比矩阵，并保存在对应位置
            jacobian0 = jacobian_norm_3d * jacobian_invdep;
            jacobian1 = jacobian_norm_3d * jacobian_i;
            jacobian2 = jacobian_norm_3d * jacobian_j;
            jacobian3 = jacobian_norm_3d * jacobian_ex;
        }
        this->SetJacobian(0, jacobian0);
        this->SetJacobian(1, jacobian1);
        this->SetJacobian(2, jacobian2);
        this->SetJacobian(3, jacobian3);
    }
}