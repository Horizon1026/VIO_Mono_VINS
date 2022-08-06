#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>
#include <include/ba_solver/graph_optimizor/kernel_function.hpp>

// 命名空间为 GraphOptimizor
namespace GraphOptimizor {

    // 构造图的边的基类，在定义实例时需要指定维度
    template<typename Scalar>
    class EdgeBase {
    private:
        // 定义静态变量，用于自动生成节点的 ID
        static size_t globalID;
    private:
        // 此边的观测，可能会有很多有关观测，因此不一定和 residual 维度一致
        VectorX<Scalar> observation;
        // 此边的观测误差
        VectorX<Scalar> residual;
        // 此边观测误差的协方差矩阵的逆，即信息矩阵
        MatrixX<Scalar> information;
        // 与此边相关联的节点（与此误差相关的待优化参数）
        std::vector<std::shared_ptr<VertexBase<Scalar>>> vertices;
        // 此边的观测误差，对每一个关联节点的雅可比矩阵
        std::vector<MatrixX<Scalar>> jacobians;
        // 鲁棒核函数
        std::shared_ptr<KernelBase<Scalar>> kernel;

    private:
        // 此边的唯一标识 ID，自动生成
        size_t ID;
        // 此边的可变索引 ID
        size_t orderedID;
        // 此边的类型
        size_t type;

    public:
        /* 构造函数与析构函数 */
        EdgeBase(int residualDimension, int verticesNum, std::shared_ptr<KernelBase<Scalar>> kernel);
        ~EdgeBase() {}

    public:
        /* 重置全局 ID 计数器 */
        static void ResetGlobalID(void);
        /* 增加一个节点 */
        bool AddVertex(std::shared_ptr<VertexBase<Scalar>> vertex, size_t vertexID);
        /* 增加多个节点 */
        bool AddVertices(const std::vector<std::shared_ptr<VertexBase<Scalar>>> &newVertices);
        /* 返回指定节点 */
        std::shared_ptr<VertexBase<Scalar>> GetVertex(size_t ID);
        /* 返回所有节点 */
        std::vector<std::shared_ptr<VertexBase<Scalar>>> &GetVertices(void);
        /* 返回关联的节点数量 */
        size_t GetVerticesNum(void);
        /* 设置鲁棒核函数 */
        void SetKernel(std::shared_ptr<KernelBase<Scalar>> kernel);
        /* 返回鲁棒核函数 */
        std::shared_ptr<KernelBase<Scalar>> &GetKernel(void);
        /* 计算误差的平方，即马氏距离二范数 */
        Scalar ComputeSquareResidual(void);
        /* 覆写此边对应的残差向量 */
        void SetResidual(const VectorX<Scalar> &r);
        /* 返回此边对应的残差向量，计算时需要带上核函数 */
        VectorX<Scalar> &GetResidual(void);
        /* 覆写此边指定节点的雅可比矩阵 */
        void SetJacobian(size_t ID, const MatrixX<Scalar> &J);
        /* 返回此边指定节点的雅可比矩阵，计算时需要带上核函数 */
        MatrixX<Scalar> &GetJacobian(size_t ID);
        /* 返回此边所有的雅可比矩阵 */
        std::vector<MatrixX<Scalar>> &GetJacobians(void);
        /* 设置观测信息对应的协方差矩阵的逆 */
        void SetInformation(const MatrixX<Scalar> &information);
        /* 返回观测信息对应的协方差矩阵的逆 */
        MatrixX<Scalar> &GetInformation(void);
        /* 设置此边的观测 */
        void SetObservation(const VectorX<Scalar> &observation);
        /* 返回此边的观测 */
        VectorX<Scalar> &GetObservation(void);
        /* 返回此边的唯一标识 ID 号 */
        size_t GetID(void);
        /* 设置此边的可变索引 ID 号 */
        void SetOrderedID(size_t orderedID);
        /* 返回此边的可变索引 ID 号 */
        size_t GetOrderedID(void);
        /* 检查此边是否完整 */
        bool IsValid(void);
        /* 设置此边的类型 */
        void SetType(size_t type);
        /* 返回此边的类型 */
        size_t GetType(void);

    public:
        /* 计算残差，由继承此基类的子类完成，需要注意计算鲁棒和函数 */
        virtual void ComputeResidual(void) = 0;
        /* 计算所有雅可比矩阵，由继承此基类的子类完成，需要注意计算鲁棒和函数 */
        virtual void ComputeJacobians(void) = 0;
    };

    // 静态变量初始化
    template<typename Scalar> size_t EdgeBase<Scalar>::globalID = 0;

    /* 以下为类方法的定义 */
    /* 构造函数与析构函数 */
    template<typename Scalar>
    EdgeBase<Scalar>::EdgeBase(int residualDimension, int verticesNum, std::shared_ptr<KernelBase<Scalar>> kernel) {
        this->residual.resize(residualDimension, 1);
        this->information.setIdentity(residualDimension, residualDimension);
        this->vertices.resize(verticesNum);
        this->jacobians.resize(verticesNum);
        this->ID = EdgeBase::globalID;
        ++EdgeBase::globalID;
        this->SetKernel(kernel);
    }


    /* 重置全局 ID 计数器 */
    template<typename Scalar>
    void EdgeBase<Scalar>::ResetGlobalID(void) {
        globalID = 0;
    }


    /* 增加一个节点 */
    template<typename Scalar>
    bool EdgeBase<Scalar>::AddVertex(std::shared_ptr<VertexBase<Scalar>> vertex, size_t vertexID) {
        if (vertexID < this->vertices.size()) {
            this->vertices[vertexID] = vertex;
            return true;
        } else {
            return false;
        }
    }


    /* 增加多个节点 */
    template<typename Scalar>
    bool EdgeBase<Scalar>::AddVertices(const std::vector<std::shared_ptr<VertexBase<Scalar>>> &newVertices) {
        if (this->vertices.size() == newVertices.size()) {
            this->vertices = newVertices;
            return true;
        } else {
            return false;
        }
    }


    /* 返回指定节点 */
    template<typename Scalar>
    std::shared_ptr<VertexBase<Scalar>> EdgeBase<Scalar>::GetVertex(size_t ID) {
        if (ID < this->vertices.size()) {
            return this->vertices[ID];
        } else {
            return nullptr;
        }
    }


    /* 返回所有节点 */
    template<typename Scalar>
    std::vector<std::shared_ptr<VertexBase<Scalar>>> &EdgeBase<Scalar>::GetVertices(void) {
        return this->vertices;
    }


    /* 返回关联的节点数量 */
    template<typename Scalar>
    size_t EdgeBase<Scalar>::GetVerticesNum(void) {
        return this->vertices.size();
    }


    /* 设置鲁棒核函数 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetKernel(std::shared_ptr<KernelBase<Scalar>> kernel) {
        if (kernel == nullptr) {
            std::shared_ptr<TrivalKernel<Scalar>> trival(new TrivalKernel<Scalar>());
            this->kernel = trival;
        } else {
            this->kernel = kernel;
        }
    }


    /* 返回鲁棒核函数 */
    template<typename Scalar>
    std::shared_ptr<KernelBase<Scalar>> &EdgeBase<Scalar>::GetKernel(void) {
        return this->kernel;
    }


    /* 计算误差的平方，即马氏距离二范数 */
    template<typename Scalar>
    Scalar EdgeBase<Scalar>::ComputeSquareResidual(void) {
        return this->residual.transpose() * this->information * this->residual;
    }


    /* 覆写此边对应的残差向量 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetResidual(const VectorX<Scalar> &r) {
        if (this->residual.rows() == r.rows()) {
            this->residual = r;
        }
    }


    /* 返回此边对应的误差，计算时需要带上核函数 */
    template<typename Scalar>
    VectorX<Scalar> &EdgeBase<Scalar>::GetResidual(void) {
        return this->residual;
    }


    /* 覆写此边指定节点的雅可比矩阵 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetJacobian(size_t ID, const MatrixX<Scalar> &J) {
        if (ID < this->jacobians.size()) {
            this->jacobians[ID] = J;
        }
    }


    /* 返回此边指定节点的雅可比矩阵，计算时需要带上核函数 */
    template<typename Scalar>
    MatrixX<Scalar> &EdgeBase<Scalar>::GetJacobian(size_t ID) {
        if (ID < this->jacobians.size()) {
            return this->jacobians[ID];
        } else {
            MatrixX<Scalar> temp;
            return temp;
        }
    }


    /* 返回此边所有的雅可比矩阵 */
    template<typename Scalar>
    std::vector<MatrixX<Scalar>> &EdgeBase<Scalar>::GetJacobians(void) {
        return this->jacobians;
    }


    /* 设置观测信息对应的协方差矩阵的逆 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetInformation(const MatrixX<Scalar> &information) {
        if (this->information.rows() == information.rows() &&
            this->information.cols() == information.cols()) {
            this->information = information;
        }
    }


    /* 返回观测信息对应的协方差矩阵的逆 */
    template<typename Scalar>
    MatrixX<Scalar> &EdgeBase<Scalar>::GetInformation(void) {
        return this->information;
    }


    /* 设置此边的观测 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetObservation(const VectorX<Scalar> &observation) {
        this->observation = observation;
    }


    /* 返回此边的观测 */
    template<typename Scalar>
    VectorX<Scalar> &EdgeBase<Scalar>::GetObservation(void) {
        return this->observation;
    }


    /* 返回此边的唯一标识 ID 号 */
    template<typename Scalar>
    size_t EdgeBase<Scalar>::GetID(void) {
        return this->ID;
    }


    /* 设置此边的可变索引 ID 号 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetOrderedID(size_t orderedID) {
        this->orderedID = orderedID;
    }


    /* 返回此边的可变索引 ID 号 */
    template<typename Scalar>
    size_t EdgeBase<Scalar>::GetOrderedID(void) {
        return this->orderedID;
    }


    /* 检查此边是否完整 */
    template<typename Scalar>
    bool EdgeBase<Scalar>::IsValid(void) {
        if (this->information.rows() != this->residual.rows() ||
            this->information.cols() != this->residual.rows() ||
            this->observation.rows() != this->residual.rows()) {
            return false;
        }
        for (size_t i = 0; i < this->vertices.size(); ++i) {
            if (this->jacobians[i].rows() != this->residual.rows() ||
                this->jacobians[i].cols() != this->vertices[i]->GetCalculationDimension()) {
                return false;
            }
        }
        return true;
    }


    /* 设置此边的类型 */
    template<typename Scalar>
    void EdgeBase<Scalar>::SetType(size_t type) {
        this->type = type;
    }


    /* 返回此边的类型 */
    template<typename Scalar>
    size_t EdgeBase<Scalar>::GetType(void) {
        return this->type;
    }

}