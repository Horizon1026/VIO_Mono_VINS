#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>

// 定义命名空间为 GraphOptimizor
namespace GraphOptimizor {
    /*
        Vertex Landmark Position

        存储参数为：tx, ty, tz，总共 3 DOF
        运算参数为：tx, ty, tz，总共 3 DOF

        不论是存储参数，还是运算参数，都是 t_w，也就是世界坐标系下的位置
    */
    template<typename Scalar>
    class VertexLandmarkPosition : public VertexBase<Scalar> {
    public:
        /* 构造函数与析构函数 */
        VertexLandmarkPosition() : VertexBase<Scalar>(3, 3) {}
        ~VertexLandmarkPosition() {}
    };
}