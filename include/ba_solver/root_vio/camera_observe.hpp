#pragma once

#include <include/ba_solver/root_vio/vertex.hpp>
#include <include/ba_solver/root_vio/kernel_function.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {
    // 定义相机观测特征点的信息管理类，管理某一个特征点的某一次观测
    template<typename Scalar>
    class CameraObserve {
    public:
        // 对应的相机 Pose 节点
        std::shared_ptr<VertexCameraPose<Scalar>> camera;
        // 观测的归一化平面坐标
        Vector2<Scalar> norm;
        // 协方差矩阵的逆的根号形式，用于 J.T * J -> J.T * S * J
        Matrix2<Scalar> sqrtInfo;
        // 鲁棒核函数
        std::shared_ptr<KernelBase<Scalar>> kernel;

    public:
        /* 构造函数与析构函数 */
        CameraObserve(std::shared_ptr<VertexCameraPose<Scalar>> camera,
            const Vector2<Scalar> &norm,
            const Matrix2<Scalar> &infoMatrix,
            std::shared_ptr<KernelBase<Scalar>> kernel = nullptr) {
            this->camera = camera;
            this->norm = norm;
            this->sqrtInfo.setIdentity();
            for (size_t i = 0; i < 2; ++i) {
                this->sqrtInfo(i, i) = std::sqrt(infoMatrix(i, i));
            }
            this->SetKernelFunction(kernel);
        }
        CameraObserve(std::shared_ptr<VertexCameraPose<Scalar>> camera,
            const Vector2<Scalar> &norm,
            std::shared_ptr<KernelBase<Scalar>> kernel = nullptr) {
            this->camera = camera;
            this->norm = norm;
            this->sqrtInfo = Matrix2<Scalar>::Identity();
            this->SetKernelFunction(kernel);
        }
        ~CameraObserve() {}

    public:
        /* 设置鲁棒核函数 */
        bool SetKernelFunction(std::shared_ptr<KernelBase<Scalar>>) {
            if (kernel == nullptr) {
                std::shared_ptr<TrivalKernel<Scalar>> trival(new TrivalKernel<Scalar>);
                this->kernel = trival;
            } else {
                this->kernel = kernel;
            }
            return true;
        }
    };
}
    