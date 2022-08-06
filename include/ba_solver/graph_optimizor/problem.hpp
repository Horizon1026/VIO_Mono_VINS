#pragma once

#include <include/ba_solver/graph_optimizor/vertex.hpp>
#include <include/ba_solver/graph_optimizor/edge.hpp>
#include <include/ba_solver/graph_optimizor/timer.hpp>
#include <tbb/tbb.h>

// 命名空间为 GraphOptimizor
namespace GraphOptimizor {

    // 定义图优化待求解问题
    template<typename Scalar>
    class Problem {
    public:
        // 通过枚举类型，定义求解器所使用的方法
        enum Method {
            LM_Manual = 0,
            LM_Auto,
            DogLeg
        };
        // 通过枚举类型，定义线性求解器所使用的方法
        enum LinearSolver {
            PCG_Solver = 0,
            LDLT_Solver,
            LLT_Solver,
            QR_Solver
        };
        // 通过结构体进行参数的配置
        struct Options {
            Scalar minCostDownRate = 1e-6;          // 误差下降率低于此值则提前终止迭代
            Scalar minNormDeltaX = 1e-6;            // 状态增量低于此值则提前终止迭代
            size_t maxMinCostHold = 3;              // 最小误差持续没有更新则提前终止迭代
            size_t maxInvalidStep = 3;              // 连续多次错误迭代则提前终止迭代
            Scalar minLambda = 1e-8;                // 阻尼因子变化的下限
            Scalar maxLambda = 1e32;                // 阻尼因子变化的上限
            Scalar initLambda = 1e-4;               // 阻尼因子的初值
            Scalar minRadius = 1e-8;                // 信赖域半径变化的下限
            Scalar maxRadius = 20;                  // 信赖域半径变化的上限
            Scalar initRadius = 1e4;                // 信赖域半径的初值
            Scalar minPCGCostDownRate = 1e-6;       // PCG 线性方程组求解器的误差下降率阈值
            Scalar maxTimeCost = INFINITY;          // 最大优化耗时，超出此阈值则提前终止迭代
            Scalar minPriorItem = 1e-8;             // 先验信息中某个数小于此值则认为是 0
        };
        // 通过结构体管理先验信息
        struct PriorInfo {
            MatrixX<Scalar> H;
            MatrixX<Scalar> JTinv;
            VectorX<Scalar> b;
            VectorX<Scalar> r;
        };

    /* 通用管理的成员变量 */
    private:
        // 与此问题关联的节点，以及对应的唯一标识 ID。因先验信息对于重新添加到 problem 中的节点的顺序索引要求，此处不可用 unordered_map
        std::map<size_t, std::shared_ptr<VertexBase<Scalar>>> vertices;
        std::vector<std::shared_ptr<VertexBase<Scalar>>> docker_vertices;    // 并行计算快速访问
        // 与此问题关联的边，以及对应的唯一标识 ID
        std::unordered_map<size_t, std::shared_ptr<EdgeBase<Scalar>>> edges;
        std::vector<std::shared_ptr<EdgeBase<Scalar>>> docker_edges;    // 并行计算快速访问
        // 某一个节点，以及与其相关的所有的边
        std::unordered_multimap<size_t, std::shared_ptr<EdgeBase<Scalar>>> vertex_edges;
        // 求解此问题采用的数值优化方法
        Method method = Method::LM_Auto;
        // 求解此问题采用的线性求解器
        LinearSolver linearSolver = LinearSolver::PCG_Solver;
        // 此问题构造的增量方程，并对未增加阻尼因子的 H 矩阵的对角线元素进行备份
        MatrixX<Scalar> matrix_H;
        VectorX<Scalar> vector_b;
        VectorX<Scalar> diag_H;
        // 此问题增量方程的锁
        using mutexType = tbb::spin_rw_mutex;
        using lockerType = std::vector<std::shared_ptr<mutexType>>;
        lockerType locker;
        // 此问题迭代过程中的所有节点参数的增量
        VectorX<Scalar> delta_x;
        // 增量方程中 H 矩阵的维度
        size_t sizeof_matrix_H;
        // 求解增量方程时被暂时边缘化的顶点，其对应的唯一标识 ID，以及其在 H 矩阵中的总维度
        std::unordered_map<size_t, std::shared_ptr<VertexBase<Scalar>>> margWhenSolving;
        std::vector<std::shared_ptr<VertexBase<Scalar>>> docker_margWhenSolving;    // 并行计算快速访问
        size_t sizeof_margWhenSolving;
        size_t sizeof_reverseWhenSolving;
        // 求解增量方程时被暂时边缘化的顶点的类型
        std::unordered_set<size_t> typeof_margWhenSolving;

    /* 误差计算，以及终止迭代阈值相关参数定义 */
    private:
        // 某一次线性化并求解出增量之后的增益比
        Scalar rho;
        // 整体增量、整体误差、最小误差及其对应的迭代步数、成功迭代一次时对应线性化点的总误差
        Scalar normDeltaX;
        Scalar sumCost;
        std::pair<size_t, Scalar> minCost;
        Scalar linearizedCost;
        // 连续迭代错误更新的次数
        size_t invalidCount;
        // 求解器整体耗时
        Scalar timeCost;
        // 配置参数，内含迭代相关阈值
        Options options;

    /* 构造函数与析构函数 */
    public:
        Problem();
        ~Problem() {}

    /* 通用对外接口方法 */
    public:
        /* 清空重置一个空的问题 */
        bool Reset(void);
        /* 增加一个节点 */
        bool AddVertex(const std::shared_ptr<VertexBase<Scalar>> &vertex);
        /* 移除一个节点 */
        bool RemoveVertex(std::shared_ptr<VertexBase<Scalar>> vertex);
        /* 按顺序打印出所有节点的信息，格式为 <id, ordered_id, type, [params]> */
        bool PrintVerticesInfo(void);
        /* 增加一个边 */
        bool AddEdge(const std::shared_ptr<EdgeBase<Scalar>> &edge);
        /* 移除一个边 */
        bool RemoveEdge(std::shared_ptr<EdgeBase<Scalar>> edge);
        /* 返回与某一个节点有关的所有边 */
        std::vector<std::shared_ptr<EdgeBase<Scalar>>> GetRelativeEdges(std::shared_ptr<VertexBase<Scalar>> vertex);
        /* 返回此问题中边的数量 */
        size_t GetEdgeNum(void);
        /* 返回此问题中节点的数量 */
        size_t GetVertexNum(void);
        /* 设置此问题的数值优化方法 */
        bool SetMethod(Method method);
        /* 设置此问题采用的线性求解器 */
        bool SetLinearSolver(LinearSolver solver);
        /* 为此问题设置迭代阈值条件 */
        bool SetOptions(const struct Options &options);
        /* 设置在求解过程中暂时被边缘化的节点的类型 */
        bool SetMargnedVertexTypesWhenSolving(size_t typeof_margWhenSolving);
        /* 清空在求解过程中暂时被边缘化的节点的类型 */
        bool ClearMargnedVertexTypesWhenSolving(void);
        /* 采用由 this->method 指定的方法，求解问题，并可指定最大迭代次数 */
        bool Solve(int maxIteration);
        /* TODO：测试相关函数 */
        bool Test(void);

    /* 通用内部接口方法 */
    private:
        /* 判断某一种类型是否需要在求解过程中暂时边缘化 */
        bool NeedMargWhenSolving(size_t vertexType);
        /* 为此问题的每一个节点进行排序，确定哪些节点在求解过程中需要暂时被边缘化 */
        void OrderVertices(void);
        /* 构造并行访问容器，为并行化作准备 */
        void PrepareForParallel(void);
        /* 计算每一条边对应的误差，以及总误差 */
        void ComputeResidual(void);
        /* 计算每一条边中每一个节点参数对于误差的雅可比矩阵 */
        void ComputeJacobians(void);
        /* 构造增量方程 Hx=b */
        void ConstructIncrementalFunction(void);
        void ConstructIncrementalFunction(std::unordered_map<size_t, std::shared_ptr<EdgeBase<Scalar>>> &setEdges);
        void ConstructIncrementalFunction(std::vector<std::shared_ptr<EdgeBase<Scalar>>> &setEdges);
        /* 为增量方程增加先验信息 */
        void AddPriorInformation(void);
        /* 在求解过程中，暂时边缘化掉孤立的节点，这些节点存储在 this->margWhenSolving 中 */
        void MarginalizeIoslatedVertices(MatrixX<Scalar> &subH,
                                         VectorX<Scalar> &subb);
        /* 更新所有节点的待优化参数 */
        void Update(void);
        /* 回滚所有节点的待优化参数 */
        void RollBack(void);
        /* 求解线性方程，求解方法由 this->linearSolver 决定 */
        VectorX<Scalar> SolveLinearFunction(const MatrixX<Scalar> &A, const VectorX<Scalar> &b);
        /* 预条件共轭梯度法求解器 */
        VectorX<Scalar> PCG(const MatrixX<Scalar> &A, const VectorX<Scalar> &b);
        /* 判断是否满足终止迭代条件 */
        bool IsConverged(size_t iter, const std::string &name);
        bool IsUnconverged(size_t iter, const std::string &name);

    /* LM 算法相关参数 */
    private:
        // LM 算法的阻尼因子
        Scalar lambda;
        // LM 算法调整阻尼因子的相关参数
        Scalar Lup;
        Scalar Ldown;
        Scalar v;

    /* LM 算法相关方法 */
    public:
        /* 为此问题的 LM 算法设置阻尼因子相关参数 */
        bool LM_SetDampParameter(Scalar Lup, Scalar Ldown);

    private:
        /* LM 算法求解 */
        bool LM_Solve(int maxIteration);
        /* LM 算法初始化参数 */
        bool LM_Initialize(void);
        /* 求解增量方程 */
        bool LM_SolveIncrementalFunction(void);
        /* 判断一次迭代过程是否有效 */
        bool LM_IsValidStep(Scalar threshold);

    /* DogLeg 算法相关参数 */
    private:
        // 置信域半径
        Scalar radius;
        // 最速下降的步长
        Scalar alpha;

    /* DogLeg 算法相关方法 */
    private:
        /* DogLeg 算法求解 */
        bool DogLeg_Solve(int maxIteration);
        /* DogLeg 算法初始化参数 */
        bool DogLeg_Initialize(void);
        /* 求解增量方程 */
        bool DogLeg_SolveIncrementalFunction(void);
        /* 判断一次迭代过程是否有效 */
        bool DogLeg_IsValidStep(Scalar threshold);

    /* 边缘化与先验信息相关参数 */
    private:
        // 先验信息，按照 orderedID() 进行排序
        PriorInfo prior;
        // 先验 b 向量和误差向量的备份
        PriorInfo backupPrior;
        // 与将被边缘化的节点有关的边
        std::unordered_map<size_t, std::shared_ptr<EdgeBase<Scalar>>> relativeEdges;
        std::vector<std::shared_ptr<EdgeBase<Scalar>>> docker_relativeEdges;

    /* 边缘化与先验信息相关，对外接口函数 */
    /* 边缘化执行之前，要确保 problem 中仅参与了待边缘化节点和与之有边约束的节点 */
    public:
        /* 设置先验信息 */
        bool SetPrior(const MatrixX<Scalar> &prior_H,
                      const VectorX<Scalar> &prior_b,
                      const MatrixX<Scalar> &prior_JTinv,
                      const VectorX<Scalar> &prior_r);
        /* 提取先验信息 */
        void GetPrior(MatrixX<Scalar> &prior_H,
                      VectorX<Scalar> &prior_b,
                      MatrixX<Scalar> &prior_JTinv,
                      VectorX<Scalar> &prior_r);
        /* 指定待边缘化的节点，以及先验信息的维度，构造先验信息 */
        bool Marginalize(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg, size_t size);

    /* 边缘化与先验信息相关，内部接口函数 */
    private:
        /* 备份先验 b 向量以及误差 */
        void BackupPrior(void);
        /* 根据增量更新先验误差 */
        bool UpdatePrior(const VectorX<Scalar> &delta_x);
        /* 读取先验 b 向量以及误差的备份，回退先验误差 */
        void RollBackPrior(void);
        /* 找出所有与指定节点有关的边 */
        bool FindAllRelativeVertices(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg);
        /* 为所有非 typeof_margWhenSolving 类型的节点排序，并调整 typeof_margWhenSolving 类型的节点的排序 */
        bool AdjustedOrdering(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg, size_t priorSize);
        /* 将指定节点对应的增量方程中的元素块移动到右下角 */
        bool MoveBlockToRightBotton(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg);
        /* 对 this->matrix_H 进行 Schur 补操作，构造先验信息 */
        bool ConstructPrior(size_t margSize);
    };
}


// 此处对通用内部接口和通用外部接口进行定义
namespace GraphOptimizor {
    /* 构造函数 */
    template<typename Scalar>
    Problem<Scalar>::Problem() {
        this->Reset();
    }


    /* 清空重置一个空的问题 */
    template<typename Scalar>
    bool Problem<Scalar>::Reset(void) {
        this->vertices.clear();
        this->docker_vertices.clear();
        this->edges.clear();
        this->docker_edges.clear();
        this->vertex_edges.clear();
        this->method = Method::DogLeg;
        this->sizeof_matrix_H = 0;
        this->margWhenSolving.clear();
        this->docker_margWhenSolving.clear();
        this->sizeof_margWhenSolving = 0;
        this->sizeof_reverseWhenSolving = 0;
        this->typeof_margWhenSolving.clear();
        this->normDeltaX = INFINITY;
        this->sumCost = 0;
        this->minCost = std::make_pair(0, INFINITY);
        this->invalidCount = 0;
        this->options.maxInvalidStep = 3;
        this->options.maxLambda = 1e32;
        this->options.maxMinCostHold = 3;
        this->options.minCostDownRate = 1e-6;
        this->options.minLambda = 1e-8;
        this->options.initLambda = 1e-4;
        this->options.initRadius = 1e4;
        this->options.minNormDeltaX = 1e-6;
        this->options.minPCGCostDownRate = 1e-6;
        this->options.minRadius = 1e-8;
        this->options.maxRadius = 20;
        this->options.maxTimeCost = INFINITY;
        this->options.minPriorItem = 1e-8;
        this->LM_SetDampParameter(2.0, 9.0);
        this->relativeEdges.clear();
        this->docker_relativeEdges.clear();
        VertexBase<Scalar>::ResetGlobalID();
        EdgeBase<Scalar>::ResetGlobalID();
        return true;
    }


    /* 增加一个节点 */
    template<typename Scalar>
    bool Problem<Scalar>::AddVertex(const std::shared_ptr<VertexBase<Scalar>> &vertex) {
        // 不重复添加节点
        if (this->vertices.find(vertex->GetID()) != this->vertices.end()) {
            return false;
        } else {
            this->vertices.insert(std::make_pair(vertex->GetID(), vertex));
            return true;
        }
    }


    /* 移除一个节点 */
    template<typename Scalar>
    bool Problem<Scalar>::RemoveVertex(std::shared_ptr<VertexBase<Scalar>> vertex) {
        // 如果本身就不存在此节点
        if (this->vertices.find(vertex->GetID()) == this->vertices.end()) {
            return false;
        }
        // 移除与此节点相关的所有边
        auto relativeEdges = this->GetRelativeEdges(vertex);
        for (auto edge : relativeEdges) {
            this->RemoveEdge(edge);
        }
        // 移除此节点
        this->vertices.erase(vertex->GetID());
        // 如果此节点是在求解过程中需要暂时边缘化的节点，则移除记录信息
        this->margWhenSolving.erase(vertex->GetID());
        return true;
    }


    /* 按顺序打印出所有节点的信息，格式为 <id, ordered_id, type, [params]> */
    template<typename Scalar>
    bool Problem<Scalar>::PrintVerticesInfo(void) {
        if (this->vertices.empty()) {
            return false;
        }
        std::cout << "<Problem> All vertices in this problem with format <id, ordered_id, type, [params]> are:" << std::endl;
        for (auto &v : this->vertices) {
            auto &vertex = v.second;
            std::cout << "  vertex <" << vertex->GetID() << ", " << vertex->GetOrderedID() << ", " << vertex->GetType() <<
                ", [" << vertex->GetParameters().transpose() << "]>\n";
        }
        return true;
    }


    /* 增加一个边 */
    template<typename Scalar>
    bool Problem<Scalar>::AddEdge(const std::shared_ptr<EdgeBase<Scalar>> &edge) {
        // 不重复添加边
        if (this->edges.find(edge->GetID()) != this->edges.end()) {
            return false;
        } else {
            this->edges.insert(std::make_pair(edge->GetID(), edge));
        }
        // 更新由节点索引边的映射容器，便于寻找与某个节点有关的边
        for (auto &vertex : edge->GetVertices()) {
            this->vertex_edges.insert(std::make_pair(vertex->GetID(), edge));
        }
        return true;
    }


    /* 移除一个边 */
    template<typename Scalar>
    bool Problem<Scalar>::RemoveEdge(std::shared_ptr<EdgeBase<Scalar>> edge) {
        if (this->edges.find(edge->GetID()) == this->edges.end()) {
            return false;
        } else {
            // 从由节点索引边的映射容器 this->vertex_edges 中剔除与此边相关的映射关系
            std::list<typename std::unordered_multimap<size_t, std::shared_ptr<EdgeBase<Scalar>>>::iterator> needErase;
            for (auto &vertex : edge->GetVertices()) {
                auto range = this->vertex_edges.equal_range(vertex->GetID());
                for (auto iter = range.first; iter != range.second; ++iter) {
                    if (iter->second == edge) {
                        needErase.emplace_back(iter);
                    }
                }
            }
            for (auto &it : needErase) {
                this->vertex_edges.erase(it);
            }

            // 移除这条边
            this->edges.erase(edge->GetID());
            return true;
        }
    }


    /* 返回与某一个节点有关的所有边 */
    template<typename Scalar>
    std::vector<std::shared_ptr<EdgeBase<Scalar>>> Problem<Scalar>::GetRelativeEdges(std::shared_ptr<VertexBase<Scalar>> vertex) {
        std::vector<std::shared_ptr<EdgeBase<Scalar>>> retEdges;
        auto range = this->vertex_edges.equal_range(vertex->GetID());
        for (auto it = range.first; it != range.second; ++it) {
            retEdges.emplace_back(it->second);
        }
        return retEdges;
    }


    /* 返回此问题中边的数量 */
    template<typename Scalar>
    size_t Problem<Scalar>::GetEdgeNum(void) {
        return this->edges.size();
    }


    /* 返回此问题中节点的数量 */
    template<typename Scalar>
    size_t Problem<Scalar>::GetVertexNum(void) {
        return this->vertices.size();
    }


    /* 设置此问题的数值优化方法 */
    template<typename Scalar>
    bool Problem<Scalar>::SetMethod(Method method) {
        this->method = method;
        return true;
    }


    /* 设置此问题采用的线性求解器 */
    template<typename Scalar>
    bool Problem<Scalar>::SetLinearSolver(LinearSolver solver) {
        this->linearSolver = solver;
        return true;
    }


    /* 为此问题设置迭代阈值条件 */
    template<typename Scalar>
    bool Problem<Scalar>::SetOptions(const struct Options &options) {
        this->options = options;
        return true;
    }


    /* 设置在求解过程中暂时被边缘化的节点的类型 */
    template<typename Scalar>
    bool Problem<Scalar>::SetMargnedVertexTypesWhenSolving(size_t typeof_margWhenSolving) {
        this->typeof_margWhenSolving.insert(typeof_margWhenSolving);
        return true;
    }


    /* 清空在求解过程中暂时被边缘化的节点的类型 */
    template<typename Scalar>
    bool Problem<Scalar>::ClearMargnedVertexTypesWhenSolving(void) {
        this->typeof_margWhenSolving.clear();
        return true;
    }


    /* 采用由 this->method 指定的方法，求解问题，并可指定最大迭代次数 */
    template<typename Scalar>
    bool Problem<Scalar>::Solve(int maxIteration) {
        // 给所有节点进行排序，并构造并行访问容器
        Timer timer;
        this->OrderVertices();
        this->PrepareForParallel();
        this->timeCost = Scalar(0);

        // 根据所选数值优化方法，进入对应分支
        bool res = true;
        switch (this->method) {
            case Method::LM_Manual:
            case Method::LM_Auto:
                res = this->LM_Solve(maxIteration);
                break;
            case Method::DogLeg:
                res = this->DogLeg_Solve(maxIteration);
                break;
            default:
                res = false;
                break;
        }
        std::cout << "<Finish> Problem solve totally cost " << timer.Stop() << " ms\n";
        return res;
    }


    /* TODO：测试相关函数 */
    template<typename Scalar>
    bool Problem<Scalar>::Test(void) {
        return true;
    }


    /* 判断某一种类型是否需要在求解过程中暂时边缘化 */
    template<typename Scalar>
    bool Problem<Scalar>::NeedMargWhenSolving(size_t vertexType) {
        if (this->typeof_margWhenSolving.find(vertexType) == this->typeof_margWhenSolving.end()) {
            return false;
        } else {
            return true;
        }
    }


    /* 为此问题的每一个节点进行排序，确定哪些节点在求解过程中需要暂时被边缘化 */
    template<typename Scalar>
    void Problem<Scalar>::OrderVertices(void) {
        this->sizeof_matrix_H = 0;
        this->sizeof_reverseWhenSolving = 0;
        this->sizeof_margWhenSolving = 0;
        this->margWhenSolving.clear();
        for (auto &v : this->vertices) {
            auto &vertex = v.second;
            if (this->NeedMargWhenSolving(vertex->GetType()) == true) {
                // 在求解过程中需要暂时边缘化的节点，需要排在后面，这里单独排列，并作记录
                vertex->SetOrderedID(this->sizeof_margWhenSolving);
                this->sizeof_margWhenSolving += vertex->GetCalculationDimension();
                this->margWhenSolving.insert(std::make_pair(vertex->GetID(), vertex));
            } else {
                // 在求解过程中不需要暂时边缘化的节点也是单独排列
                vertex->SetOrderedID(this->sizeof_reverseWhenSolving);
                this->sizeof_reverseWhenSolving += vertex->GetCalculationDimension();
            }
        }

        // 将每一个需要暂时边缘化的节点的 orderedID 进行偏移，使其整体在不需要暂时边缘化的节点的后面
        for (auto &vertex : this->margWhenSolving) {
            vertex.second->SetOrderedID(vertex.second->GetOrderedID() + this->sizeof_reverseWhenSolving);
        }
        this->sizeof_matrix_H = this->sizeof_reverseWhenSolving + this->sizeof_margWhenSolving;

        // 打印出排列节点的信息 TODO
        // this->PrintVerticesInfo();
    }


    /* 构造并行访问容器，为并行化作准备 */
    template<typename Scalar>
    void Problem<Scalar>::PrepareForParallel(void) {
        if (this->vertices.size() > 0) {
            this->docker_vertices.clear();
            this->docker_vertices.reserve(this->vertices.size());
            for (auto &item : this->vertices) {
                this->docker_vertices.emplace_back(item.second);
            }
        }
        if (this->edges.size() > 0) {
            this->docker_edges.clear();
            this->docker_edges.reserve(this->edges.size());
            for (auto &item : this->edges) {
                this->docker_edges.emplace_back(item.second);
            }
        }
        if (this->margWhenSolving.size() > 0) {
            this->docker_margWhenSolving.clear();
            this->docker_margWhenSolving.reserve(this->margWhenSolving.size());
            for (auto &item : this->margWhenSolving) {
                this->docker_margWhenSolving.emplace_back(item.second);
            }
        }
        if (this->relativeEdges.size() > 0) {
            this->docker_relativeEdges.clear();
            this->docker_relativeEdges.reserve(this->relativeEdges.size());
            for (auto &item : this->relativeEdges) {
                this->docker_relativeEdges.emplace_back(item.second);
            }
        }
        // 初始化增量方程的锁
        this->locker.reserve(this->vertices.size());
        for (size_t i = 0; i < this->vertices.size(); ++i) {
            std::shared_ptr<mutexType> mutex(new mutexType());
            this->locker.emplace_back(mutex);
        }
    }


    /* 计算每一条边对应的误差，以及总误差 */
    template<typename Scalar>
    void Problem<Scalar>::ComputeResidual(void) {
        // 遍历每一条边中的每一个节点，计算其整体误差
        this->sumCost = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, this->docker_edges.size()), Scalar(0),
            [&] (tbb::blocked_range<size_t> range, Scalar localSum) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->docker_edges[i]->ComputeResidual();
                    auto residual = this->docker_edges[i]->GetResidual();
                    // 在此处计算 rho(r.T * S * r)，但是误差计算依旧是马氏距离 r.T * S * r，取代欧式距离 ||r||2
                    Scalar x = this->docker_edges[i]->ComputeSquareResidual();
                    this->docker_edges[i]->GetKernel()->Compute(x);
                    localSum += this->docker_edges[i]->GetKernel()->y;
                }
                return localSum;
            }, std::plus<Scalar>()
        );
        // 如果存在先验误差，也要加上
        if (this->prior.r.rows() > 0) {
            // 分解先验 H 矩阵所得的 prior_r 本质上是 sqrt(S) * r，因此这里直接 squaredNorm() 即可
            this->sumCost += this->prior.r.squaredNorm();
        }
    }


    /* 计算每一条边中每一个节点参数对于误差的雅可比矩阵 */
    template<typename Scalar>
    void Problem<Scalar>::ComputeJacobians(void) {
        // 遍历每一条边中的每一个节点，计算其雅可比矩阵
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_edges.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    this->docker_edges[i]->ComputeJacobians();
                }
            }
        );
    }


    /* 构造增量方程 Hx=b */
    template<typename Scalar>
    void Problem<Scalar>::ConstructIncrementalFunction(void) {
        this->ConstructIncrementalFunction(this->docker_edges);
        this->AddPriorInformation();
    }
    template<typename Scalar>
    void Problem<Scalar>::ConstructIncrementalFunction(std::unordered_map<size_t, std::shared_ptr<EdgeBase<Scalar>>> &setEdges) {
        // 清空或初始化 H 矩阵和 b 向量
        this->matrix_H.setZero(this->sizeof_matrix_H, this->sizeof_matrix_H);
        this->vector_b.setZero(this->sizeof_matrix_H);
        // 遍历每一条边中的每一个节点，提取其雅可比矩阵，按照 orderedID 填充到 H 和 b 的对应位置
        for (auto &e : setEdges) {
            auto &edge = e.second;
            auto &jacobians = edge->GetJacobians();
            auto &vertices = edge->GetVertices();

            for (size_t i = 0; i < vertices.size(); ++i) {
                auto &vertex_i = vertices[i];
                auto &jacobian_i = jacobians[i];
                // 如果这个节点固定，则认为其雅可比矩阵为零矩阵
                if (vertex_i->IsFixed() == true) {
                    continue;
                }
                size_t index_i = vertex_i->GetOrderedID();
                size_t dimension_i = vertex_i->GetCalculationDimension();

                // 需要构造增量方程 J.T * S * J * delta_x = - J.T * S * r
                // 并且在叠加到大增量方程中之前，对 H 和 b 增加权重 rho'(r.T * S * r)
                MatrixX<Scalar> JtS = jacobian_i.transpose() * edge->GetInformation() * edge->GetKernel()->y_;
                this->vector_b.segment(index_i, dimension_i).noalias() -= JtS * edge->GetResidual();

                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &vertex_j = vertices[j];
                    auto &jacobian_j = jacobians[j];
                    if (vertex_j->IsFixed() == true) {
                        continue;
                    }
                    size_t index_j = vertex_j->GetOrderedID();
                    size_t dimension_j = vertex_j->GetCalculationDimension();
                    MatrixX<Scalar> subH = JtS * jacobian_j;
                    this->matrix_H.block(index_i, index_j, dimension_i, dimension_j).noalias() += subH;
                    if (i != j) {
                        this->matrix_H.block(index_j, index_i, dimension_j, dimension_i).noalias() += subH.transpose();
                    }
                }
            }
        }
    }
    template<typename Scalar>
    void Problem<Scalar>::ConstructIncrementalFunction(std::vector<std::shared_ptr<EdgeBase<Scalar>>> &setEdges) {
        // 构造 reduce 并行化结构体
        struct Reductor {
            const std::vector<std::shared_ptr<EdgeBase<Scalar>>> &edges;
            MatrixX<Scalar> &H;
            VectorX<Scalar> &b;
            lockerType &locker;
            Reductor(const std::vector<std::shared_ptr<EdgeBase<Scalar>>> &edges_, MatrixX<Scalar> &H_, VectorX<Scalar> &b_, lockerType &locker_) :
                edges(edges_), H(H_), b(b_), locker(locker_) {}
            Reductor(Reductor &r, tbb::split) : edges(r.edges), H(r.H), b(r.b), locker(r.locker) {}
            inline void join(const Reductor &r) {}
            void operator()(const tbb::blocked_range<size_t> &range) {
                // 遍历每一条边中的每一个节点，提取其雅可比矩阵，按照 orderedID 填充到 H 和 b 的对应位置
                for (size_t k = range.begin(); k < range.end(); ++k) {
                    auto &edge = edges[k];
                    auto &jacobians = edge->GetJacobians();
                    auto &vertices = edge->GetVertices();
                    for (size_t i = 0; i < vertices.size(); ++i) {
                        auto &vertex_i = vertices[i];
                        auto &jacobian_i = jacobians[i];
                        // 如果这个节点固定，则认为其雅可比矩阵为零矩阵
                        if (vertex_i->IsFixed() == true) {
                            continue;
                        }
                        size_t index_i = vertex_i->GetOrderedID();
                        size_t id_i = vertex_i->GetID();
                        size_t dimension_i = vertex_i->GetCalculationDimension();
                        // 需要构造增量方程 J.T * S * J * delta_x = - J.T * S * r
                        // 并且在叠加到大增量方程中之前，对 H 和 b 增加权重 rho'(r.T * S * r)
                        MatrixX<Scalar> JtS = jacobian_i.transpose() * edge->GetInformation() * edge->GetKernel()->y_;
                        VectorX<Scalar> subb = - JtS * edge->GetResidual();

                        for (size_t j = i; j < vertices.size(); ++j) {
                            auto &vertex_j = vertices[j];
                            auto &jacobian_j = jacobians[j];
                            if (vertex_j->IsFixed() == true) {
                                continue;
                            }
                            size_t index_j = vertex_j->GetOrderedID();
                            size_t id_j = vertex_j->GetID();
                            size_t dimension_j = vertex_j->GetCalculationDimension();
                            // 填充 H 矩阵
                            MatrixX<Scalar> subH = JtS * jacobian_j;
                            locker[id_i]->lock();
                            if (i == j) {
                                H.block(index_i, index_j, dimension_i, dimension_j).noalias() += subH;
                                b.segment(index_i, dimension_i).noalias() += subb;
                            } else {
                                locker[id_j]->lock();
                                H.block(index_i, index_j, dimension_i, dimension_j).noalias() += subH;
                                H.block(index_j, index_i, dimension_j, dimension_i).noalias() += subH.transpose();
                                locker[id_j]->unlock();
                            }
                            locker[id_i]->unlock();
                        }
                    }
                }
            }
        };
        // 清空或初始化 H 矩阵和 b 向量
        this->matrix_H.setZero(this->sizeof_matrix_H, this->sizeof_matrix_H);
        this->vector_b.setZero(this->sizeof_matrix_H);

        // 并行执行拼接矩阵的过程
        Reductor r(setEdges, this->matrix_H, this->vector_b, this->locker);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, setEdges.size()), r);
    }


    /* 为增量方程增加先验信息 */
    template<typename Scalar>
    void Problem<Scalar>::AddPriorInformation(void) {
        if (this->prior.b.rows() > 0) {
            // 在 this->backupPrior 中的 H 并不真正参与备份，因此可用于临时变量
            this->backupPrior.H = this->prior.H;
            VectorX<Scalar> tempPrior_b = this->prior.b;
            // 遍历优先计算的所有节点，如果发现这个节点被 fix 了，则对应的先验信息也得为0
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_vertices.size()),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto &vertex = this->docker_vertices[i];
                        if (vertex->IsFixed()) {
                            size_t index = vertex->GetOrderedID();
                            size_t dimension = vertex->GetCalculationDimension();
                            if (index + dimension > static_cast<size_t>(this->backupPrior.H.rows())) {
                                continue;
                            }
                            this->backupPrior.H.block(index, 0, dimension, this->backupPrior.H.cols()).setZero();
                            this->backupPrior.H.block(0, index, this->backupPrior.H.rows(), dimension).setZero();
                            tempPrior_b.segment(index, dimension).setZero();
                        }
                    }
                }
            );
            // 之后将先验信息直接添加到 Hessian 矩阵和 b 向量中
            this->matrix_H.topLeftCorner(this->backupPrior.H.rows(), this->backupPrior.H.cols()) += this->backupPrior.H;
            this->vector_b.head(tempPrior_b.rows()) += tempPrior_b;
        }
    }


    /* 在求解过程中，暂时边缘化掉孤立的节点，这些节点存储在 this->margWhenSolving 中 */
    template<typename Scalar>
    void Problem<Scalar>::MarginalizeIoslatedVertices(MatrixX<Scalar> &subH,
                                                      VectorX<Scalar> &subb) {
        size_t reverse = this->sizeof_reverseWhenSolving;
        size_t marg = this->sizeof_margWhenSolving;

        // 对 Hmm 求逆，并进一步计算 Hrm * Hmm_inv
        MatrixX<Scalar> Hrm_Hmm_inv;
        Hrm_Hmm_inv.setZero(reverse, marg);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_margWhenSolving.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    auto &vertex = this->docker_margWhenSolving[i];
                    size_t index = vertex->GetOrderedID();
                    size_t dimension = vertex->GetCalculationDimension();
                    Hrm_Hmm_inv.block(0, index - reverse, reverse, dimension) =
                        this->matrix_H.block(0, index, reverse, dimension) *
                        this->matrix_H.block(index, index, dimension, dimension).inverse();
                }
            }
        );

        // 计算 Schur 补
        // subH = Hrr - Hrm_Hmm_inv * Hmr;
        // subb = br - Hrm_Hmm_inv * bm;
        subH.setZero(reverse, reverse);
        subb.setZero(reverse);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, subH.rows()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    subH.block(i, 0, 1, reverse) = this->matrix_H.block(i, 0, 1, reverse) - Hrm_Hmm_inv.block(i, 0, 1, Hrm_Hmm_inv.cols()) * this->matrix_H.block(reverse, 0, marg, reverse);
                    subb.segment(i, 1) = this->vector_b.segment(i, 1) - Hrm_Hmm_inv.block(i, 0, 1, Hrm_Hmm_inv.cols()) * this->vector_b.segment(reverse, marg);
                }
            }
        );
    }


    /* 更新所有节点的待优化参数 */
    template<typename Scalar>
    void Problem<Scalar>::Update(void) {
        // 更新所有节点的参数
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_vertices.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    auto &vertex = this->docker_vertices[i];
                    if (vertex->IsFixed()) {
                        continue;
                    }
                    // 更新参数之前需要先做备份
                    vertex->BackUp();
                    size_t index = vertex->GetOrderedID();
                    size_t dimension = vertex->GetCalculationDimension();
                    vertex->Update(this->delta_x.segment(index, dimension));
                }
            }
        );
        // 若存在先验信息，则需要根据 EFJ 调整先验误差
        if (this->prior.b.rows() > 0) {
            this->BackupPrior();
            this->UpdatePrior(this->delta_x);
        }
    }


    /* 回滚所有节点的待优化参数 */
    template<typename Scalar>
    void Problem<Scalar>::RollBack(void) {
        // 回滚所有节点的参数
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_vertices.size()),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    auto &vertex = this->docker_vertices[i];
                    if (vertex->IsFixed()) {
                        continue;
                    }
                    vertex->RollBack();
                }
            }
        );
        // 如果是 LM 算法，则需要去除掉增量方程 Hx=b 中增加的阻尼因子
        if (this->method == Method::LM_Manual || this->method == Method::LM_Auto) {
            this->matrix_H.diagonal() = this->diag_H;
        }
        // 先验信息也需要重新调整
        this->RollBackPrior();
    }


    /* 求解线性方程，求解方法由 this->linearSolver 决定 */
    template<typename Scalar>
    VectorX<Scalar> Problem<Scalar>::SolveLinearFunction(const MatrixX<Scalar> &A, const VectorX<Scalar> &b) {
        switch (this->linearSolver) {
            case LinearSolver::PCG_Solver:
                return this->PCG(A, b);
            case LinearSolver::LDLT_Solver:
                return A.ldlt().solve(b);
            case LinearSolver::LLT_Solver:
                return A.llt().solve(b);
            case LinearSolver::QR_Solver:
                return A.colPivHouseholderQr().solve(b);
            default:
                VectorX<Scalar> res = b;
                res.setZero();
                return res;
        }
    }


    /* 预条件共轭梯度法求解器 */
    template<typename Scalar>
    VectorX<Scalar> Problem<Scalar>::PCG(const MatrixX<Scalar> &A, const VectorX<Scalar> &b) {
        // 考虑到向量空间的基底数，以 b 向量的维度作为最大迭代次数
        int maxIteration = b.rows();

        // 设置初值，计算初始误差
        VectorX<Scalar> x(VectorX<Scalar>::Zero(b.rows()));
        VectorX<Scalar> r0(b);  // initial r = b - A*0 = b
        if (r0.norm() < this->options.minPCGCostDownRate) {
            return x;
        }

        // 计算预条件矩阵
        VectorX<Scalar> M_inv_diag = A.diagonal();
        M_inv_diag.array() = Scalar(1) / M_inv_diag.array();
        for (size_t i = 0; i < static_cast<size_t>(M_inv_diag.rows()); ++i) {
            if (std::isinf(M_inv_diag(i)) == true) {
                M_inv_diag(i) = 0;
            }
        }
        VectorX<Scalar> z0 = M_inv_diag.array() * r0.array();    // solve M * z0 = r0
        
        // 取得第一个基底，计算基底权重 alpha，并更新 x
        VectorX<Scalar> p(z0);
        VectorX<Scalar> w = A * p;
        Scalar r0z0 = r0.dot(z0);
        Scalar alpha = r0z0 / p.dot(w);
        x += alpha * p;
        VectorX<Scalar> r1 = r0 - alpha * w;

        // 设定迭代终止的阈值
        Scalar threshold = this->options.minPCGCostDownRate * r0.norm();

        // 迭代求解
        int i = 0;
        while (r1.norm() > threshold && i < maxIteration) {
            i++;
            VectorX<Scalar> z1 = M_inv_diag.array() * r1.array();
            Scalar r1z1 = r1.dot(z1);
            Scalar belta = r1z1 / r0z0;
            z0 = z1;
            r0z0 = r1z1;
            r0 = r1;
            p = belta * p + z1;
            w = A * p;
            alpha = r1z1 / p.dot(w);
            x += alpha * p;
            r1 -= alpha * w;
        }
        return x;
    }


    /* 判断是否满足终止迭代条件 */
    template<typename Scalar>
    bool Problem<Scalar>::IsConverged(size_t iter, const std::string &name) {
        if (this->linearizedCost - this->sumCost < this->linearizedCost * this->options.minCostDownRate &&
            this->linearizedCost > this->sumCost) {
            std::cout << "<" << name << " Stop> cost down rate up to threshold, finished." << std::endl;
            return true;
        }
        if (this->normDeltaX < this->options.minNormDeltaX) {
            std::cout << "<" << name << " Stop> norm delta X up to threshold, finished." << std::endl;
            return true;
        }
        // 更新最小误差，并判断是否终止迭代
        if (this->minCost.second > this->sumCost) {
            this->minCost.first = iter;
            this->minCost.second = this->sumCost;
        }
        if (iter - this->minCost.first > this->options.maxMinCostHold) {
            std::cout << "<" << name << " Stop> min cost holding times up to threshold, finished." << std::endl;
            return true;
        }
        return false;
    }
    template<typename Scalar>
    bool Problem<Scalar>::IsUnconverged(size_t iter, const std::string &name) {
        // 如果连续无效迭代步数过多，则认为求解失败
        if (this->invalidCount > this->options.maxInvalidStep) {
            std::cout << "<" << name << " Stop> invalid step count up to threshold, finished." << std::endl;
            return true;
        }
        // 如果耗时超出阈值，则终止迭代
        if (this->timeCost > this->options.maxTimeCost) {
            std::cout << "<" << name << " Stop> solving cost time up to threshold, finished." << std::endl;
            return true;
        }
        return false;
    }
}


// 此处对 LM 算法相关方法进行定义
namespace GraphOptimizor {
    /* 为此问题的 LM 算法设置阻尼因子相关参数，但只对 Policy 1 生效 */
    template<typename Scalar>
    bool Problem<Scalar>::LM_SetDampParameter(Scalar Lup, Scalar Ldown) {
        this->Lup = Lup;
        this->Ldown = Ldown;
        return true;
    }


    /* LM 算法求解 */
    template<typename Scalar>
    bool Problem<Scalar>::LM_Solve(int maxIteration) {
        Timer timer;
        // 计算所有边的误差和雅可比
        this->ComputeResidual();
        this->ComputeJacobians();
        std::cout << "<LM Begin> origin sum cost r.T * S * r is " << this->sumCost << std::endl;
        // 计算所有边对于节点的雅可比矩阵，构造增量方程
        this->ConstructIncrementalFunction();
        // 初始化 LM 算法
        this->LM_Initialize();
        // 迭代求解
        for (size_t iter = 0; iter < static_cast<size_t>(maxIteration); ++iter) {
            // 求解增量方程，得到 this->delta_x
            this->LM_SolveIncrementalFunction();
            // 对每一个节点的参数进行备份，之后更新参数
            this->Update();
            // 重新计算节点参数更新后的所有边的误差，根据误差和增量变化情况，判断是否提前终止迭代
            this->ComputeResidual();
            // 判断是否收敛，如果已经收敛，则可以提前结束
            if (this->IsConverged(iter, "LM")) {
                break;
            }
            // 判断此步更新是否有效，并修改阻尼因子
            bool res = this->LM_IsValidStep(0);
            Scalar stepTimeCost = timer.Stop();
            this->timeCost += stepTimeCost;
            std::cout << "<LM Iter " << iter << " / " << maxIteration <<"> cost " << this->sumCost << ", dx " <<
                this->normDeltaX << ", lambda " << this->lambda << ", rho " << this->rho << ", time cost " << stepTimeCost <<
                " ms" << std::endl;
            timer.Start();
            if (res) {
                // 如果此步有效，则为下一步迭代作准备
                this->invalidCount = 0;
                this->ComputeJacobians();
                this->ConstructIncrementalFunction();
            } else {
                // 如果此步无效，则回退节点参数的值，去除增量方程中 H 矩阵内增加的阻尼项，以便下一次循环重新增加新的阻尼
                ++this->invalidCount;
                this->RollBack();
            }
            // 判断是否无法收敛，是的话则提前结束
            if (this->IsUnconverged(iter, "LM")) {
                break;
            }
        }
        std::cout << "<LM End> final sum cost r.T * S * r is " << this->sumCost << std::endl;
        return true;
    }


    /* LM 算法初始化参数 */
    template<typename Scalar>
    bool Problem<Scalar>::LM_Initialize(void) {
        this->lambda = this->options.initLambda;
        this->linearizedCost = this->sumCost;
        this->v = 2.0;
        return true;
    }


    /* 求解增量方程 */
    template<typename Scalar>
    bool Problem<Scalar>::LM_SolveIncrementalFunction(void) {
        // 为 H 矩阵添加阻尼因子，添加策略参考 ceres
        this->diag_H = this->matrix_H.diagonal();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, static_cast<size_t>(this->matrix_H.rows())),
            [&] (tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    Scalar diag = std::min(Scalar(this->options.maxLambda), std::max(Scalar(this->options.minLambda), this->matrix_H(i, i)));
                    this->matrix_H(i, i) += this->lambda * diag;
                }
            }
        );

        size_t reverse = this->sizeof_reverseWhenSolving;
        size_t marg = this->sizeof_margWhenSolving;
        if (marg > 0) {
            // 如果设置了在求解过程中对部分节点进行暂时边缘化，则进行 Schur 补操作
            MatrixX<Scalar> subH;
            VectorX<Scalar> subb;
            this->MarginalizeIoslatedVertices(subH, subb);
            // 分两步求解增量方程，首先求解 delta_xp 部分
            if (this->delta_x.rows() != this->vector_b.rows()) {
                this->delta_x.resize(this->vector_b.rows());
            }
            this->delta_x.head(reverse) = this->SolveLinearFunction(subH, subb);
            // 接下来求解 delta_xl 部分
            VectorX<Scalar> tail_b;
            tail_b.setZero(marg);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, marg),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        tail_b.segment(i, 1) = this->vector_b.segment(i + reverse, 1) -
                            this->matrix_H.block(i + reverse, 0, 1, reverse) * this->delta_x.head(reverse);
                    }
                }
            );
            // 用于求解 delta_xl 的增量方程的 H 矩阵是块对角矩阵，因此可以进行并行加速
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_margWhenSolving.size()),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto &vertex = this->docker_margWhenSolving[i];
                        size_t index = vertex->GetOrderedID();
                        size_t dimension = vertex->GetCalculationDimension();
                        this->delta_x.segment(index, dimension) = this->SolveLinearFunction(
                            this->matrix_H.block(index, index, dimension, dimension), tail_b.segment(index - reverse, dimension));
                    }
                }
            );
        } else {
            // 否则直接求解
            this->delta_x = this->SolveLinearFunction(this->matrix_H, this->vector_b);
        }

        // 计算本次的步长
        this->normDeltaX = this->delta_x.norm();
        if (std::isnan(this->normDeltaX) || std::isinf(this->normDeltaX)) {
            return false;
        } else {
            return true;
        }
    }


    /* 判断一次迭代过程是否有效 */
    template<typename Scalar>
    bool Problem<Scalar>::LM_IsValidStep(Scalar threshold) {
        /* 参考文献：The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems */
        if (this->method == Method::LM_Manual) {
            // 阻尼因子更新策略 1
            Scalar scale = this->delta_x.transpose() * (this->lambda * this->diag_H.asDiagonal() * this->delta_x + this->vector_b);
            this->rho = Scalar(0.5) * (this->linearizedCost - this->sumCost) / (scale + 1e-6);
            if (this->rho > threshold && std::isfinite(this->sumCost)) {
                this->lambda = std::max(this->lambda / this->Ldown, Scalar(1e-7));
                this->linearizedCost = this->sumCost;
                return true;
            } else {
                this->lambda = std::min(this->lambda * this->Lup, Scalar(1e7));
                return false;
            }
        } else {
            // 阻尼因子更新策略 3
            Scalar scale = this->delta_x.transpose() * (this->lambda * this->delta_x + this->vector_b);
            this->rho = Scalar(0.5) * (this->linearizedCost - this->sumCost) / (scale + 1e-6);
            if (this->rho > threshold && std::isfinite(this->sumCost)) {
                this->lambda *= std::max(Scalar(1.0 / 3.0), Scalar(1.0 - std::pow(2.0 * this->rho - 1.0, 3)));
                this->v = 2.0;
                this->linearizedCost = this->sumCost;
                return true;
            } else {
                this->lambda *= this->v;
                this->v *= 2.0;
                return false;
            }
        }
        
    }
}



/* 以下为 Dogleg 算法相关方法定义 */
namespace GraphOptimizor {
    /* DogLeg 算法求解 */
    template<typename Scalar>
    bool Problem<Scalar>::DogLeg_Solve(int maxIteration) {
        Timer timer;
        // 计算所有边的误差
        this->ComputeResidual();
        // 计算所有边对应节点的雅可比
        this->ComputeJacobians();
        std::cout << "<DogLeg Begin> origin sum cost r.T * S * r is " << this->sumCost << std::endl;
        // 计算所有边对于节点的雅可比矩阵，构造增量方程
        this->ConstructIncrementalFunction();
        // 初始化 DogLeg 算法
        this->DogLeg_Initialize();

        // 迭代求解
        for (size_t iter = 0; iter < static_cast<size_t>(maxIteration); ++iter) {
            // 求解增量方程，得到 this->delta_x
            this->DogLeg_SolveIncrementalFunction();
            // 对每一个节点的参数进行备份，之后更新参数
            this->Update();
            // 重新计算节点参数更新后的所有边的误差，根据误差和增量变化情况，判断是否提前终止迭代
            this->ComputeResidual();
            // 判断是否收敛，如果已经收敛，则可以提前结束
            if (this->IsConverged(iter, "LM")) {
                break;
            }
            // 判断此步更新是否有效，并修改阻尼因子
            bool res = this->DogLeg_IsValidStep(0);
            Scalar stepTimeCost = timer.Stop();
            this->timeCost += stepTimeCost;
            std::cout << "<DogLeg Iter " << iter << " / " << maxIteration <<"> cost " << this->sumCost << ", dx " <<
                this->normDeltaX << ", radius " << this->radius << ", rho " << this->rho << ", alpha " << this->alpha <<
                ", time cost " << stepTimeCost << " ms" << std::endl;
            timer.Start();
            if (res) {
                // 如果此步有效，则为下一步迭代作准备
                this->invalidCount = 0;
                this->ComputeJacobians();
                this->ConstructIncrementalFunction();
            } else {
                // 如果此步无效，则回退节点参数的值，去除增量方程中 H 矩阵内增加的阻尼项，以便下一次循环重新增加新的阻尼
                ++this->invalidCount;
                this->RollBack();
            }
            // 判断是否终止迭代
            if (this->IsUnconverged(iter, "DogLeg")) {
                break;
            }
        }
        std::cout << "<DogLeg End> final sum cost r.T * S * r is " << this->sumCost << std::endl;
        return true;
    }


    /* DogLeg 算法初始化参数 */
    template<typename Scalar>
    bool Problem<Scalar>::DogLeg_Initialize(void) {
        this->linearizedCost = this->sumCost;
        this->radius = this->options.initRadius;
        for (size_t i = 0; i < static_cast<size_t>(this->matrix_H.rows()); ++i) {
            this->radius = std::max(this->matrix_H(i, i), this->radius);
        }
        this->radius = std::min(this->radius, Scalar(1e7));
        return true;
    }


    /* 求解增量方程 */
    template<typename Scalar>
    bool Problem<Scalar>::DogLeg_SolveIncrementalFunction(void) {
        /* 第一步：单独计算梯度下降法的增量 delta_x_SD */
        this->alpha = (this->vector_b.transpose() * this->vector_b)(0, 0) / (this->vector_b.transpose() * this->matrix_H * this->vector_b)(0, 0);
        VectorX<Scalar> delta_x_sd = this->alpha * this->vector_b;

        /* 第二步：单独计算高斯牛顿法的增量 delta_x_GN */
        VectorX<Scalar> delta_x_gn(this->matrix_H.cols());
        size_t reverse = this->sizeof_reverseWhenSolving;
        size_t marg = this->sizeof_margWhenSolving;
        if (marg > 0) {
            // 如果设置了在求解过程中对部分节点进行暂时边缘化，则进行 Schur 补操作
            MatrixX<Scalar> subH;
            VectorX<Scalar> subb;
            this->MarginalizeIoslatedVertices(subH, subb);
            // 分两步求解增量方程，首先求解 delta_xp 部分
            delta_x_gn.head(reverse) = this->SolveLinearFunction(subH, subb);
            // 接下来求解 delta_xl 部分
            VectorX<Scalar> tail_b;
            tail_b.setZero(marg);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, marg),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        tail_b.segment(i, 1) = this->vector_b.segment(i + reverse, 1) -
                            this->matrix_H.block(i + reverse, 0, 1, reverse) * delta_x_gn.head(reverse);
                    }
                }
            );
            // 用于求解 delta_xl 的增量方程的 H 矩阵是块对角矩阵，因此可以进行并行加速
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->docker_margWhenSolving.size()),
                [&] (tbb::blocked_range<size_t> range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto &vertex = this->docker_margWhenSolving[i];
                        size_t index = vertex->GetOrderedID();
                        size_t dimension = vertex->GetCalculationDimension();
                        delta_x_gn.segment(index, dimension) = this->SolveLinearFunction(
                            this->matrix_H.block(index, index, dimension, dimension), tail_b.segment(index - reverse, dimension));
                    }
                }
            );
        } else {
            // 否则直接求解
            delta_x_gn = this->SolveLinearFunction(this->matrix_H, this->vector_b);
        }

        /* 第三步：基于置信范围 radius，融合两个增量 */
        Scalar normSD = delta_x_sd.norm();
        Scalar normGN = delta_x_gn.norm();
        if (!std::isnan(normGN)) {
            if (normGN <= this->radius && normSD <= this->radius) {
                this->delta_x = delta_x_gn;
            } else if (normGN >= this->radius && normSD >= this->radius) {
                this->delta_x = delta_x_sd * this->radius / normSD;
            } else {
                VectorX<Scalar> diff = delta_x_gn - delta_x_sd;
                Scalar a = diff.transpose() * diff;
                Scalar b = 2.0 * delta_x_sd.transpose() * diff;
                Scalar c = normSD * normSD - this->radius * this->radius;
                Scalar weight = (std::sqrt(b * b - 4.0 * a * c) - b) / (2.0 * a);
                this->delta_x = delta_x_sd + weight * diff;
            }
        } else {
            Scalar scale = this->radius > normSD ? 1.0 : (this->radius / normSD);
            this->delta_x = delta_x_sd * scale;
        }

        // 计算本次增量的步长
        this->normDeltaX = this->delta_x.norm();
        if (std::isnan(this->normDeltaX) || std::isinf(this->normDeltaX)) {
            return false;
        } else {
            return true;
        }
    }


    /* 判断一次迭代过程是否有效 */
    template<typename Scalar>
    bool Problem<Scalar>::DogLeg_IsValidStep(Scalar threshold) {
        // 参考文献：The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems
        // 其中的公式 (14) 描述的增益比 rho
        // rho 的分母 = 2 * r.T * J * x - x.T * J.T * J * x
        // 其中 J.T * r = - vector_b，其中 J.T * J = matrix_H
        MatrixX<Scalar> scale = Scalar(2.0) * this->vector_b.transpose() * this->delta_x + this->delta_x.transpose() * this->matrix_H * this->delta_x;
        this->rho = Scalar(0.5) * (this->linearizedCost - this->sumCost) / (scale(0, 0) + 1e-6);
        if (this->rho > threshold && std::isfinite(this->sumCost) && !std::isnan(this->sumCost)) {
            if (this->rho > Scalar(0.75)) {
                this->radius = std::max(this->radius, Scalar(3.0) * this->normDeltaX);
            } else if (this->rho < Scalar(0.25)) {
                this->radius *= Scalar(0.5);
            }
            this->linearizedCost = this->sumCost;
            return true;
        } else {
            this->radius *= Scalar(0.25);
            return false;
        }
    }
}


/* 边缘化与先验信息相关方法定义 */
namespace GraphOptimizor {
    /* 设置先验信息 */
    template<typename Scalar>
    bool Problem<Scalar>::SetPrior(const MatrixX<Scalar> &prior_H,
                                   const VectorX<Scalar> &prior_b,
                                   const MatrixX<Scalar> &prior_JTinv,
                                   const VectorX<Scalar> &prior_r) {
        this->prior.H = prior_H;
        this->prior.b = prior_b;
        this->prior.JTinv = prior_JTinv;
        this->prior.r = prior_r;
        this->backupPrior.b = this->prior.b;
        this->backupPrior.r = prior_r;
        return true;
    }


    /* 提取先验信息 */
    template<typename Scalar>
    void Problem<Scalar>::GetPrior(MatrixX<Scalar> &prior_H,
                                   VectorX<Scalar> &prior_b,
                                   MatrixX<Scalar> &prior_JTinv,
                                   VectorX<Scalar> &prior_r) {
        prior_H = this->prior.H;
        prior_b = this->prior.b;
        prior_JTinv = this->prior.JTinv;
        prior_r = this->prior.r;
    }
    

    /* 指定待边缘化的节点，构造先验信息 */
    template<typename Scalar>
    bool Problem<Scalar>::Marginalize(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg, size_t size) {
        // 找出所有与待边缘化节点有关的边
        this->FindAllRelativeVertices(needMarg);
        // 为所有非 typeof_margWhenSolving 类型的节点排序，并调整 typeof_margWhenSolving 类型的节点的排序
        bool res = this->AdjustedOrdering(needMarg, size);
        if (res == false) {
            return false;
        }
        this->PrepareForParallel();
        // 用 this->relativeEdges 来构造增量方程，暂时不添加先验
        this->ComputeResidual();
        this->ComputeJacobians();
        this->ConstructIncrementalFunction(this->relativeEdges);
        // 添加先验信息
        this->AddPriorInformation();
        // 如果存在 typeof_margWhenSolving 类型的节点，则首先把他们边缘化
        if (this->sizeof_margWhenSolving > 0) {
            MatrixX<Scalar> subH;
            VectorX<Scalar> subb;
            this->MarginalizeIoslatedVertices(subH, subb);
            this->matrix_H = subH;
            this->vector_b = subb;
        }
        this->sizeof_matrix_H = this->matrix_H.rows();
        // 将所有需要被边缘化的节点移动到右下角
        this->MoveBlockToRightBotton(needMarg);
        // 确定待边缘化的参数维度，根据上述过程处理后的 this->matrix_H 来构造先验信息
        size_t margSize = 0;
        for (auto &vertex : needMarg) {
            if (this->NeedMargWhenSolving(vertex->GetType())) {
                continue;
            }
            margSize += vertex->GetCalculationDimension();
        }
        this->ConstructPrior(margSize);
        return true;
    }


    /* 备份先验 b 向量以及误差 */
    template<typename Scalar>
    void Problem<Scalar>::BackupPrior(void) {
        this->backupPrior.b = this->prior.b;
        this->backupPrior.r = this->prior.r;
    }


    /* 根据增量更新先验误差 */
    template<typename Scalar>
    bool Problem<Scalar>::UpdatePrior(const VectorX<Scalar> &delta_x) {
        this->prior.b -= this->prior.H * delta_x.head(this->prior.H.cols());
        this->prior.r = - this->prior.JTinv * this->prior.b;
        return true;
    }


    /* 读取先验 b 向量的备份，回退先验误差 */
    template<typename Scalar>
    void Problem<Scalar>::RollBackPrior(void) {
        this->prior.b = this->backupPrior.b;
        this->prior.r = this->backupPrior.r;
    }


    /* 找出所有与指定节点有关的边 */
    template<typename Scalar>
    bool Problem<Scalar>::FindAllRelativeVertices(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg) {
        this->relativeEdges.clear();
        for (auto &v : needMarg) {
            auto subEdges = this->GetRelativeEdges(v);
            for (auto &edge : subEdges) {
                this->relativeEdges.insert(std::make_pair(edge->GetID(), edge));
            }
        }
        return true;
    }


    /* 为所有非 typeof_margWhenSolving 类型的节点排序，并调整 typeof_margWhenSolving 类型的节点的排序 */
    template<typename Scalar>
    bool Problem<Scalar>::AdjustedOrdering(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg, size_t priorSize) {
        // 清空维度计数
        this->sizeof_matrix_H = 0;
        this->sizeof_reverseWhenSolving = 0;
        this->sizeof_margWhenSolving = 0;
        this->margWhenSolving.clear();

        // 为所有非 typeof_margWhenSolving 类型的节点排序
        for (auto &v : this->vertices) {
            auto &vertex = v.second;
            if (this->NeedMargWhenSolving(vertex->GetType()) == false) {
                // 在求解过程中不需要暂时边缘化的节点单独排列
                vertex->SetOrderedID(this->sizeof_reverseWhenSolving);
                this->sizeof_reverseWhenSolving += vertex->GetCalculationDimension();
            }
        }

        // 校验先验信息的尺寸
        size_t correctSize = priorSize;
        for (auto &vertex : needMarg) {
            if (this->NeedMargWhenSolving(vertex->GetType()) == true) {
                continue;
            }
            correctSize += vertex->GetCalculationDimension();
        }
        if (correctSize != this->sizeof_reverseWhenSolving) {
            return false;
        }

        // 遍历与边相关的 typeof_margWhenSolving 类型的节点，重新为他们排序
        for (auto &edge : this->relativeEdges) {
            for (auto &vertex : edge.second->GetVertices()) {
                if (this->NeedMargWhenSolving(vertex->GetType()) == true) {
                    // 如果已经排序了，则跳过
                    if (this->margWhenSolving.find(vertex->GetID()) != this->margWhenSolving.end()) {
                        continue;
                    }
                    // 否则进行排序
                    vertex->SetOrderedID(this->sizeof_reverseWhenSolving + this->sizeof_margWhenSolving);
                    this->sizeof_margWhenSolving += vertex->GetCalculationDimension();
                    this->margWhenSolving.insert(std::make_pair(vertex->GetID(), vertex));
                }
            }
        }

        // 两者加和即为增量方程的维度
        this->sizeof_matrix_H = this->sizeof_reverseWhenSolving + this->sizeof_margWhenSolving;
        return true;
    }


    /* 将指定节点对应的增量方程中的元素块移动到右下角 */
    template<typename Scalar>
    bool Problem<Scalar>::MoveBlockToRightBotton(std::vector<std::shared_ptr<VertexBase<Scalar>>> &needMarg) {
        size_t size = this->sizeof_matrix_H;

        // 按照节点的 orderedID 对待移动节点进行排序
        std::sort(needMarg.begin(), needMarg.end(), [] (std::shared_ptr<VertexBase<Scalar>> vi, std::shared_ptr<VertexBase<Scalar>> vj) {
            return vi->GetOrderedID() > vj->GetOrderedID();
        });

        // 遍历每一个待边缘化的节点
        // 移动时，如果移动了索引为 i 的节点的矩阵块，则索引大于 i 的矩阵块将会错位
        for (auto &vertex : needMarg) {
            if (this->NeedMargWhenSolving(vertex->GetType())) {
                continue;
            }
            size_t idx = vertex->GetOrderedID();
            size_t dim = vertex->GetCalculationDimension();
            if (idx + dim > static_cast<size_t>(this->matrix_H.rows())) {
                continue;
            }
            // 将第 i 行移动到矩阵 H 的最下面
            MatrixX<Scalar> tempRows = this->matrix_H.block(idx, 0, dim, size);
            MatrixX<Scalar> tempBottomRows = this->matrix_H.block(idx + dim, 0, size - dim - idx, size);
            this->matrix_H.block(idx, 0, size - dim - idx, size) = tempBottomRows;
            this->matrix_H.block(size - dim, 0, dim, size) = tempRows;
            // 将第 i 列移动到矩阵 H 的最右边
            MatrixX<Scalar> tempCols = this->matrix_H.block(0, idx, size, dim);
            MatrixX<Scalar> tempRightCols = this->matrix_H.block(0, idx + dim, size, size - dim - idx);
            this->matrix_H.block(0, idx, size, size - dim - idx) = tempRightCols;
            this->matrix_H.block(0, size - dim, size, dim) = tempCols;
            // 将第 i 行移动到向量 b 的最下面
            VectorX<Scalar> tempb = this->vector_b.segment(idx, dim);
            VectorX<Scalar> tempbTail = this->vector_b.segment(idx + dim, size - dim - idx);
            this->vector_b.segment(idx, size - dim - idx) = tempbTail;
            this->vector_b.segment(size - dim, dim) = tempb;
        }
        return true;
    }


    /* 对 this->matrix_H 进行 Schur 补操作，构造先验信息 */
    template<typename Scalar>
    bool Problem<Scalar>::ConstructPrior(size_t margSize) {
        // 确定 Schur 补分块尺寸，定义误差系数
        size_t marg = margSize;
        size_t reverse = this->sizeof_matrix_H - margSize;

        // 提取矩阵块
        MatrixX<Scalar> Hrr = this->matrix_H.block(0, 0, reverse, reverse);
        MatrixX<Scalar> Hrm = this->matrix_H.block(0, reverse, reverse, marg);
        MatrixX<Scalar> Hmr = this->matrix_H.block(reverse, 0, marg, reverse);
        MatrixX<Scalar> Hmm = 0.5 * (this->matrix_H.block(reverse, reverse, marg, marg) +
            this->matrix_H.block(reverse, reverse, marg, marg).transpose());
        VectorX<Scalar> br = this->vector_b.segment(0, reverse);
        VectorX<Scalar> bm = this->vector_b.segment(reverse, marg);

        // 计算 Hmm 的逆
        Eigen::SelfAdjointEigenSolver<MatrixX<Scalar>> saes(Hmm);
        MatrixX<Scalar> Hmm_inv = saes.eigenvectors() * VectorX<Scalar>(
            (saes.eigenvalues().array() > this->options.minPriorItem).select(
                saes.eigenvalues().array().inverse(), 0
            )).asDiagonal() * saes.eigenvectors().transpose();

        // 计算 Schur 补
        MatrixX<Scalar> temp = Hrm * Hmm_inv;
        this->prior.H = Hrr - temp * Hmr;
        this->prior.b = br - temp * bm;

        // 分解先验 H 矩阵，即求 H 矩阵的特征值，剔除掉小于 this->options.minPriorItem 的那一部分
        Eigen::SelfAdjointEigenSolver<MatrixX<Scalar>> saes2(this->prior.H);
        VectorX<Scalar> S = VectorX<Scalar>((saes2.eigenvalues().array() > this->options.minPriorItem)
            .select(saes2.eigenvalues().array(), 0));
        VectorX<Scalar> S_inv = VectorX<Scalar>((saes2.eigenvalues().array() > this->options.minPriorItem)
            .select(saes2.eigenvalues().array().inverse(), 0));
        VectorX<Scalar> S_sqrt = S.cwiseSqrt();
        VectorX<Scalar> S_inv_sqrt = S_inv.cwiseSqrt();

        // 基于分解结果和先验 b 向量，进一步计算出先验雅可比和先验误差
        this->prior.JTinv = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
        this->prior.r = -this->prior.JTinv * this->prior.b;

        // 优化先验 H 矩阵和先验 b 向量的数学性质，因为 Schur 补得到的先验 H 不完全对称
        MatrixX<Scalar> J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
        this->prior.H = J.transpose() * J;
        MatrixX<Scalar> tmp_h = MatrixX<Scalar>((this->prior.H.array().abs() > 1e-9).select(this->prior.H.array(),0));
        this->prior.H = tmp_h;

        // 因为在这里，构造的 Hx=b 增量方程，本质上是 J.T * S * J * delta_x = - J.T * S * r
        // 所以分解所得的 prior_J 其实等于 sqrt(S) * J
        // 分解所得的 prior_r 其实等于 sqrt(S) * r

        return true;
    }
}