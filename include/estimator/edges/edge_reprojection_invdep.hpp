#pragma once

#include <include/ba_solver/graph_optimizor/edge.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Edge Reprojection Inverse Depth

        此边为 3 元边，与之相连的顶点有：
            <VertexLandmarkInvDepth>    landmark 在首帧观测相机坐标系中的逆深度 inv_d_c，
            <VertexPose>                首次观测到此路标点的相机的位姿 R_wc_i, t_wc_i，
            <VertexPose>                非首次观测到此路标点的相机的位姿 R_wc_j, t_wc_j

        此边的观测为归一化平面坐标，其维度为 2
    */
    template<typename Scalar>
    class EdgeReprojectionInvdep : public EdgeBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        EdgeReprojectionInvdep(const Vector2<Scalar> &normPoint_i, const Vector2<Scalar> &normPoint_j) :
            EdgeBase<Scalar>(2, 3, nullptr) {
            Eigen::Matrix<Scalar, 4, 1> observe;
            observe.head(2) = normPoint_i;
            observe.tail(2) = normPoint_j;
            this->SetObservation(observe);
        }
        ~EdgeReprojectionInvdep() {}

    public:
        /* 计算残差 */
        virtual void ComputeResidual(void) override;
        /* 计算残差对于每一个节点参数的雅可比矩阵 */
        virtual void ComputeJacobians(void) override;
    };


    /* 类成员方法定义如下 */
    /* 计算残差 */
    template<typename Scalar>
    void EdgeReprojectionInvdep<Scalar>::ComputeResidual(void) {
        // 从节点中提取参数
        Scalar invdep_i = this->GetVertex(0)->GetParameters()(0);
        VectorX<Scalar> param_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_j = this->GetVertex(2)->GetParameters();
        Quaternion<Scalar> q_wc_i(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vector3<Scalar> t_wc_i = param_i.template head<3>();
        Quaternion<Scalar> q_wc_j(param_j[6], param_j[3], param_j[4], param_j[5]);
        Vector3<Scalar> t_wc_j = param_j.template head<3>();
        Vector2<Scalar> norm_i = this->GetObservation().template head<2>();
        Vector2<Scalar> norm_j = this->GetObservation().template tail<2>();
        
        // 计算投影过程
        Vector3<Scalar> p_c_i = Vector3<Scalar>(norm_i(0), norm_i(1), Scalar(1)) / invdep_i;
        Vector3<Scalar> p_w = q_wc_i * p_c_i + t_wc_i;
        Vector3<Scalar> p_c_j = q_wc_j.inverse() * (p_w - t_wc_j);
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
    void EdgeReprojectionInvdep<Scalar>::ComputeJacobians(void) {
        // 从节点中提取参数
        Scalar invdep_i = this->GetVertex(0)->GetParameters()(0);
        VectorX<Scalar> param_i = this->GetVertex(1)->GetParameters();
        VectorX<Scalar> param_j = this->GetVertex(2)->GetParameters();
        Quaternion<Scalar> q_wc_i(param_i[6], param_i[3], param_i[4], param_i[5]);
        Matrix3<Scalar> R_wc_i(q_wc_i);
        Vector3<Scalar> t_wc_i = param_i.template head<3>();
        Quaternion<Scalar> q_wc_j(param_j[6], param_j[3], param_j[4], param_j[5]);
        Matrix3<Scalar> R_wc_j(q_wc_j);
        Vector3<Scalar> t_wc_j = param_j.template head<3>();
        Vector2<Scalar> norm_i = this->GetObservation().template head<2>();
        Vector2<Scalar> norm_j = this->GetObservation().template tail<2>();
        
        // 计算投影过程
        Vector3<Scalar> p_c_i = Vector3<Scalar>(norm_i(0), norm_i(1), Scalar(1)) / invdep_i;
        Vector3<Scalar> p_w = q_wc_i * p_c_i + t_wc_i;
        Vector3<Scalar> p_c_j = q_wc_j.inverse() * (p_w - t_wc_j);
        Scalar invdep_j = Scalar(1) / p_c_j.z();

        // 计算观测误差对于每一个状态节点的雅可比矩阵
        MatrixX<Scalar> jacobian0, jacobian1, jacobian2;
        if (std::isnan(invdep_i) || std::isinf(invdep_i) || std::isnan(invdep_j) || std::isinf(invdep_j)) {
            jacobian0 = Eigen::Matrix<Scalar, 2, 1>::Zero();
            jacobian1 = Eigen::Matrix<Scalar, 2, 6>::Zero();
            jacobian2 = Eigen::Matrix<Scalar, 2, 6>::Zero();
        } else {
            // 计算（归一化二维平面）对（相机坐标系三维坐标）的雅可比矩阵
            Eigen::Matrix<Scalar, 2, 3> jacobian_norm_3d;
            jacobian_norm_3d << invdep_j, 0, - p_c_j(0) * invdep_j * invdep_j,
                                0, invdep_j, - p_c_j(1) * invdep_j * invdep_j;

            // 计算（相机坐标系三维坐标的误差）对（第 i 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> jacobian_i;
            jacobian_i.template leftCols<3>() = R_wc_j.transpose();
            jacobian_i.template rightCols<3>() = - R_wc_j.transpose() * R_wc_i * SkewSymmetricMatrix(p_c_i);

            // 计算（相机坐标系三维坐标的误差）对（第 j 帧相机位姿）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 6> jacobian_j;
            jacobian_j.template leftCols<3>() = - R_wc_j.transpose();
            jacobian_j.template rightCols<3>() = SkewSymmetricMatrix(p_c_j);

            // 计算（相机坐标系三维坐标的误差）对（第 i 帧中特征点逆深度）的雅可比矩阵
            Eigen::Matrix<Scalar, 3, 1> jacobian_invdep;
            Vector3<Scalar> normPoint_i(norm_i(0), norm_i(1), Scalar(1));
            jacobian_invdep = - R_wc_j.transpose() * R_wc_i * normPoint_i / (invdep_i * invdep_i);

            // 利用级联求导，计算（归一化平面坐标误差）对（第 i 帧相机位姿）、（第 j 帧相机位姿）和（第 i 帧中特征点逆深度）的雅可比矩阵，并保存在对应位置
            jacobian0 = jacobian_norm_3d * jacobian_invdep;
            jacobian1 = jacobian_norm_3d * jacobian_i;
            jacobian2 = jacobian_norm_3d * jacobian_j;
        }
        this->SetJacobian(0, jacobian0);
        this->SetJacobian(1, jacobian1);
        this->SetJacobian(2, jacobian2);
    }
}