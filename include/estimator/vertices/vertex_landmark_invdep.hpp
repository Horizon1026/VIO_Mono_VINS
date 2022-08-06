#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Vertex Landmark Position

        存储参数为：inv_d，总共 1 DOF
        运算参数为：inv_d，总共 1 DOF

        不论是存储参数，还是运算参数，都是 inv_d_c，也就是首次观测到此点的相机坐标系下的 Z 轴深度
    */
    template<typename Scalar>
    class VertexLandmarkInvDepth : public VertexBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        VertexLandmarkInvDepth() : VertexBase<Scalar>(1, 1) {}
        ~VertexLandmarkInvDepth() {}
    };
}