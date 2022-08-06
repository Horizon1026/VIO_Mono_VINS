#pragma once

#include <include/ba_solver/graph_optimizor/utility.hpp>

// 命名空间为 GraphOptimizor
namespace GraphOptimizor {

    // 构造图的顶点的基类，在定义实例时需要指定维度
    template<typename Scalar>
    class VertexBase {
    private:
        // 定义静态变量，用于自动生成节点的 ID
        static size_t globalID;
    private:
        // 此节点存储的待优化参数
        VectorX<Scalar> parameters;
        // 此节点存储的备份参数
        VectorX<Scalar> backpack;
        // 此节点在优化问题中的最小维度，与 this->parameter 不一定一致
        size_t dimension;

    private:
        // 此节点的唯一标识符，自动生成
        size_t id;
        // 此节点的可变动索引
        size_t orderedId = 0;
        // 此节点是否在优化过程中被固定
        bool fixed = false;
        // 此节点的节点类型
        size_t type = 0;

    public:
        /* 构造函数，需要指明存储参数维度，以及运算参数维度 */
        VertexBase(size_t storedDimension, size_t calculationDimension);
        /* 基类的析构函数 */
        virtual ~VertexBase() {}

    public:
        /* 重置全局 ID 计数器 */
        static void ResetGlobalID(void);
        /* 返回此节点存储参数的维度 */
        size_t GetStoredDimension(void) const;
        /* 返回此节点参与运算的维度 */
        size_t GetCalculationDimension(void) const;
        /* 返回此节点的唯一索引 ID 号 */
        size_t GetID(void) const;
        /* 设置此节点的可变索引 ID 号 */
        void SetOrderedID(size_t orderedID);
        /* 返回此节点的可变索引 ID 号 */
        size_t GetOrderedID(void) const;
        /* 返回此节点保存的参数 */
        VectorX<Scalar> &GetParameters(void);
        /* 设置此节点保存的参数 */
        void SetParameters(const VectorX<Scalar> &params);
        /* 设置此节点的节点类型 */
        void SetType(const size_t &type);
        /* 返回此节点的节点类型 */
        size_t GetType(void) const;
        /* 更新节点参数 */
        virtual void Update(const VectorX<Scalar> &deltaParams);
        /* 备份节点参数 */
        void BackUp(void);
        /* 读取备份，重置节点参数 */
        void RollBack(void);
        /* 设置此节点的固定状态，默认调用此函数则固定 */
        void SetFixed(bool fixed = true);
        /* 返回此节点的固定状态 */
        bool IsFixed(void) const;
    };

    // 初始化静态变量
    template<typename Scalar> size_t VertexBase<Scalar>::globalID = 0;

    /* 以下为类方法的定义 */
    /* 构造函数，需要指明存储参数维度，以及运算参数维度 */
    template<typename Scalar>
    VertexBase<Scalar>::VertexBase(size_t storedDimension, size_t calculationDimension) {
        this->parameters.resize(storedDimension, 1);
        this->parameters.setZero();
        this->dimension = calculationDimension;
        this->id = VertexBase::globalID;
        ++VertexBase::globalID;
    }


    /* 重置全局 ID 计数器 */
    template<typename Scalar>
    void VertexBase<Scalar>::ResetGlobalID(void) {
        globalID = 0;
    }


    /* 返回此节点存储参数的维度 */
    template<typename Scalar>
    size_t VertexBase<Scalar>::GetStoredDimension(void) const {
        return this->parameters.rows();
    }


    /* 返回此节点参与运算的维度 */
    template<typename Scalar>
    size_t VertexBase<Scalar>::GetCalculationDimension(void) const {
        return this->dimension;
    }


    /* 返回此节点的唯一索引 ID 号 */
    template<typename Scalar>
    size_t VertexBase<Scalar>::GetID(void) const {
        return this->id;
    }


    /* 设置此节点的可变索引 ID 号 */
    template<typename Scalar>
    void VertexBase<Scalar>::SetOrderedID(size_t orderedID) {
        this->orderedId = orderedID;
    }


    /* 返回此节点的可变索引 ID 号 */
    template<typename Scalar>
    size_t VertexBase<Scalar>::GetOrderedID(void) const {
        return this->orderedId;
    }


    /* 返回此节点保存的参数 */
    template<typename Scalar>
    VectorX<Scalar> &VertexBase<Scalar>::GetParameters(void) {
        return this->parameters;
    }


    /* 设置此节点保存的参数 */
    template<typename Scalar>
    void VertexBase<Scalar>::SetParameters(const VectorX<Scalar> &params) {
        this->parameters = params;
    }


    /* 设置此节点的节点类型 */
    template<typename Scalar>
    void VertexBase<Scalar>::SetType(const size_t &type) {
        this->type = type;
    }


    /* 返回此节点的节点类型 */
    template<typename Scalar>
    size_t VertexBase<Scalar>::GetType(void) const {
        return this->type;
    }


    /* 更新节点参数 */
    template<typename Scalar>
    void VertexBase<Scalar>::Update(const VectorX<Scalar> &deltaParams) {
        this->parameters += deltaParams;
    }


    /* 备份节点参数 */
    template<typename Scalar>
    void VertexBase<Scalar>::BackUp(void) {
        this->backpack = this->parameters;
    }


    /* 读取备份，重置节点参数 */
    template<typename Scalar>
    void VertexBase<Scalar>::RollBack(void) {
        this->parameters = this->backpack;
    }


    /* 设置此节点的固定状态，默认调用此函数则固定 */
    template<typename Scalar>
    void VertexBase<Scalar>::SetFixed(bool fixed) {
        this->fixed = fixed;
    }


    /* 返回此节点的固定状态 */
    template<typename Scalar>
    bool VertexBase<Scalar>::IsFixed(void) const {
        return this->fixed;
    }
}