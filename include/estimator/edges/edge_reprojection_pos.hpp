#pragma once

#include <include/ba_solver/graph_optimizor/edge.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Edge Reprojection Position

        此边为 2 元边，与之相连的顶点有：
            <VertexLandmarkPosition>    landmark 在世界坐标系中的三轴位置 t_w，
            <VertexPose>                观测到此路标点的相机的位姿 R_wc, t_wc

        此边的观测为归一化平面坐标，其维度为 2
    */
    template<typename Scalar>
    class EdgeReprojectionPos : public EdgeBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        EdgeReprojectionPos(const Vector2<Scalar> &normPoint) : EdgeBase<Scalar>(2, 2, nullptr) {
            this->SetObservation(normPoint);
        }
        ~EdgeReprojectionPos() {}

    public:
        /* 计算残差 */
        virtual void ComputeResidual(void) override;
        /* 计算残差对于每一个节点参数的雅可比矩阵 */
        virtual void ComputeJacobians(void) override;
    };


    /* 类成员方法定义如下 */
    /* 计算残差 */
    template<typename Scalar>
    void EdgeReprojectionPos<Scalar>::ComputeResidual(void) {
        // 从节点中提取参数
        Vector3<Scalar> p_w = this->GetVertex(0)->GetParameters();
        VectorX<Scalar> param = this->GetVertex(1)->GetParameters();
        Quaternion<Scalar> q_wc(param[6], param[3], param[4], param[5]);
        Vector3<Scalar> t_wc = param.template head<3>();
        Vector3<Scalar> p_c = q_wc.inverse() * (p_w - t_wc);
        Scalar invDepth = 1.0 / p_c.z();

        // 计算归一化平面误差向量
        Vector2<Scalar> r;
        if (std::isinf(invDepth) == true || invDepth < 0) {
            // r = Vector2<Scalar>::Ones() * Scalar(1e6);
            r.setZero();
        } else {
            r = (p_c * invDepth).template head<2>() - this->GetObservation();
        }
        this->SetResidual(r);
    }


    /* 计算残差对于每一个节点参数的雅可比矩阵 */
    template<typename Scalar>
    void EdgeReprojectionPos<Scalar>::ComputeJacobians(void) {
        // 从节点中提取参数
        Vector3<Scalar> p_w = this->GetVertex(0)->GetParameters();
        VectorX<Scalar> param = this->GetVertex(1)->GetParameters();
        Quaternion<Scalar> q_wc(param[6], param[3], param[4], param[5]);
        Vector3<Scalar> t_wc = param.template head<3>();
        Vector3<Scalar> p_c = q_wc.inverse() * (p_w - t_wc);
        Scalar invDepth = 1.0 / p_c.z();

        // 计算（归一化二维平面）对（相机坐标系三维坐标）的雅可比矩阵
        Eigen::Matrix<Scalar, 2, 3> Jacobian_norm_point3D = Eigen::Matrix<Scalar, 2, 3>::Zero();
        // 计算（相机坐标系三维坐标误差）对（相机位姿）的雅可比矩阵
        Eigen::Matrix<Scalar, 3, 6> Jacobian_cam = Eigen::Matrix<Scalar, 3, 6>::Zero();
        // 计算（相机坐标系三维坐标误差）对（特征点在世界坐标系的位置）的雅可比矩阵
        Eigen::Matrix<Scalar, 3, 3> Jacobian_landmark = Eigen::Matrix<Scalar, 3, 3>::Zero();

        if (std::isinf(invDepth) == false && invDepth > 0) {
            Matrix3<Scalar> R_wc(q_wc);
            Jacobian_norm_point3D << invDepth, 0, - p_c(0) * invDepth * invDepth,
                                     0, invDepth, - p_c(1) * invDepth * invDepth;
            Jacobian_cam.template leftCols<3>() = - R_wc.transpose();
            Jacobian_cam.template rightCols<3>() = SkewSymmetricMatrix(p_c);
            Jacobian_landmark = R_wc.transpose();
        }

        // 利用级联求导，计算（归一化平面坐标误差）对（相机位姿）和（特征点在世界坐标系的位置）的雅可比矩阵，并保存在对应位置
        MatrixX<Scalar> jacobian0 = Jacobian_norm_point3D * Jacobian_landmark;
        MatrixX<Scalar> jacobian1 = Jacobian_norm_point3D * Jacobian_cam;
        this->SetJacobian(0, jacobian0);
        this->SetJacobian(1, jacobian1);
    }
}