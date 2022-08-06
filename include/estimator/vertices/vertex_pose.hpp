#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Vertex Pose

        存储参数为：tx, ty, tz, qx, qy, qz, qw，总共 7 DOF
        运算参数为：tx, ty, tz, r_x, r_y, r_z，总共 6 DOF

        不论是存储参数，还是运算参数，都是 R_wb/R_wc/R_bc 和 t_wb/t_wc/t_bc
    */
    template<typename Scalar>
    class VertexPose : public VertexBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        VertexPose() : VertexBase<Scalar>(7, 6) {}
        ~VertexPose() {}

    public:
        /* 重写参数更新方法，因为四元数更新不是直接相加 */
        virtual void Update(const VectorX<Scalar> &deltaParams) override;
    };


    /* 类成员方法定义如下 */
    /* 重写参数更新方法，因为四元数更新不是直接相加 */
    template<typename Scalar>
    void VertexPose<Scalar>::Update(const VectorX<Scalar> &deltaParams) {
        VectorX<Scalar> &param = this->GetParameters();
        param.template head<3>() += deltaParams.template head<3>();
        Quaternion<Scalar> dq(1.0,
                              deltaParams[3] * Scalar(0.5),
                              deltaParams[4] * Scalar(0.5),
                              deltaParams[5] * Scalar(0.5));
        Quaternion<Scalar> q(param[6], param[3], param[4], param[5]);
        q = q * dq;
        q.normalize();
        param[3] = q.x();
        param[4] = q.y();
        param[5] = q.z();
        param[6] = q.w();
    }
}