#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Vertex Velocity and Bias

        存储参数为：vx, vy, vz, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z，总共 9 DOF
        运算参数为：vx, vy, vz, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z，总共 9 DOF

        不论是存储参数，还是运算参数，都是速度相对于世界坐标系 v_wb，偏差相对于机体坐标系的 bias_b
    */
    template<typename Scalar>
    class VertexVelocityBias : public VertexBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        VertexVelocityBias() : VertexBase<Scalar>(9, 9) {}
        ~VertexVelocityBias() {}
    };
}