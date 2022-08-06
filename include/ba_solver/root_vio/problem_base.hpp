#pragma once

#include <include/ba_solver/root_vio/utility.hpp>
#include <tbb/tbb.h>

// 全局命名空间定义为 rootVIO
// 此处为模板类的声明
namespace rootVIO {
    // 定义待求解问题的基类
    template<typename Scalar>
    class ProblemBase {
    /* 构造函数与析构函数 */
    public:
        ProblemBase();
        ~ProblemBase();

    /* 配置类参数定义 */
    public:
        // 通过枚举类型，定义求解器所使用的方法
        enum DampPolicy {
            Manual = 0,     // Manual -> LM Policy 1
            Auto            // Auto   -> LM Policy 3
        } policy = DampPolicy::Auto;
        // 通过枚举类型，定义线性求解器所使用的方法
        enum LinearSolver {
            PCG_Solver = 0,
            LDLT_Solver,
            LLT_Solver,
            QR_Solver
        } linearSolver = LinearSolver::PCG_Solver;
        // 通过结构体进行参数的配置
        struct Options {
            Scalar minCostDownRate = 1e-6;          // 误差下降率低于此值则提前终止迭代
            Scalar minNormDeltaX = 1e-6;            // 状态增量低于此值则提前终止迭代
            size_t maxMinCostHold = 3;              // 最小误差持续没有更新则提前终止迭代
            size_t maxInvalidStep = 3;              // 连续多次错误迭代则提前终止迭代
            Scalar minLambda = 1e-8;                // 阻尼因子变化的下限
            Scalar maxLambda = 1e32;                // 阻尼因子变化的上限
            Scalar initLambda = 1e-6;               // 阻尼因子的初值 
            Scalar minPCGCostDownRate = 1e-6;       // PCG 线性方程组求解器的误差下降率阈值
            Scalar maxTimeCost = INFINITY;          // 最大优化耗时，超出此阈值则提前终止迭代
            Scalar minPriorItem = 1e-8;             // 先验信息中某个数小于此值则认为是 0
        } options;

    /* 通用成员方法定义 */
    public:
        /* 重置待求解问题中的基类内容 */
        bool ResetBase(void);
        /* 设置此问题采用的线性求解器 */
        bool SetLinearSolver(LinearSolver solver);
        /* 为此问题设置迭代阈值条件 */
        bool SetOptions(const struct Options &options);
        /* 为此问题的 LM 算法设置阻尼因子更新策略 */
        bool SetDampPolicy(DampPolicy policy);
        /* 为此问题的 LM 算法设置阻尼因子相关参数 */
        bool SetDampParameter(Scalar Lup = 3, Scalar Ldown = 9);

    /* LM 算法参数定义 */
    public:
        // LM 算法的阻尼因子
        Scalar lambda;
        // LM 算法衡量线性化结果好坏的参数
        Scalar rho;
        // LM 算法调整阻尼因子的相关参数
        Scalar Lup;
        Scalar Ldown;
        Scalar v;
        // 整体耗时
        Scalar timeCost;
        // 整体误差、一次迭代成功时对应的线性化点的误差
        Scalar sumCost;
        Scalar linearizedCost;
        // 连续迭代失误次数的计数器
        int invalidCount = 0;
        // 记录最小误差
        std::pair<int, Scalar> minCost;
        // 整体增量（靠后）以及其模长
        VectorX<Scalar> delta_X;
        Scalar normDeltaX;
        // 整体增量方程的 H 矩阵、H 矩阵的对角线、b 向量
        MatrixX<Scalar> matrix_H;
        VectorX<Scalar> diag_H;
        VectorX<Scalar> vector_b;
        // 用于构造整体增量方程的大雅可比矩阵 J 和大误差向量 r
        MatrixX<Scalar> matrix_J;
        VectorX<Scalar> vector_r;
        size_t rowsof_matrix_J;
        size_t colsof_matrix_J;

    /* LM 算法部分方法定义 */
    public:
        /* LM 算法初始化，确定阻尼因子 lambda 和相关参数的数值 */
        bool InitializeLM(void);
        /* 判断是否满足终止迭代条件 */
        bool IsConverged(size_t iter);
        bool IsUnconverged(size_t iter);

    /* 线性求解器方法定义 */
    public:
        /* 根据 this->linearSolver 设置的方法求解线性方程组 */
        VectorX<Scalar> SolveLinearFunction(const MatrixX<Scalar> &A, const VectorX<Scalar> &b);
        /* PCG 求解器 */
        VectorX<Scalar> PreconditionedConjugateGradient(const MatrixX<Scalar> &A, const VectorX<Scalar> &b);

    /* 边缘化相关参数定义 */
    public:
        // 先验信息以及其备份
        MatrixX<Scalar> prior_J;
        VectorX<Scalar> prior_r;
        VectorX<Scalar> stored_prior_r;

    /* 边缘化相关方法定义 */
    public:
        /* 设置先验信息 */
        bool SetPrior(const MatrixX<Scalar> &prior_J, const VectorX<Scalar> &prior_r);
        /* 提取先验信息 */
        bool GetPrior(MatrixX<Scalar> &prior_J, VectorX<Scalar> &prior_r);
        /* 对整体的雅可比矩阵进行 QR 分解，同时作用在 this->vector_r 上，最后提取出先验信息 */
        /* 保存在 this->prior_J 和 this->prior_r 中 */
        bool ConstructPrior(const size_t margSize);
    };
}


/* 此处为模板类成员的定义 */
namespace rootVIO {
    /* 构造函数与析构函数 */
    template<typename Scalar> ProblemBase<Scalar>::ProblemBase() {
        this->ResetBase();
    }
    template<typename Scalar> ProblemBase<Scalar>::~ProblemBase() {}


    /* 重置待求解问题中的基类内容 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::ResetBase(void) {
        this->options.maxInvalidStep = 3;
        this->options.maxLambda = 1e32;
        this->options.maxMinCostHold = 3;
        this->options.minCostDownRate = 1e-6;
        this->options.minLambda = 1e-8;
        this->options.initLambda = 1e-6;
        this->options.minNormDeltaX = 1e-6;
        this->options.minPCGCostDownRate = 1e-6;
        this->options.maxTimeCost = INFINITY;
        this->options.minPriorItem = 1e-6;
        this->invalidCount = 0;
        this->sumCost = 0;
        this->timeCost = 0;
        this->normDeltaX = INFINITY;
        this->minCost.first = 0;
        this->minCost.second = INFINITY;
        this->SetDampParameter(9.0, 11.0);
        this->v = 2.0;
        return true;
    }


    /* 设置此问题采用的线性求解器 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::SetLinearSolver(LinearSolver solver) {
        this->linearSolver = solver;
        return true;
    }


    /* 为此问题设置迭代阈值条件 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::SetOptions(const struct Options &options) {
        this->options = options;
        return true;
    }


    /* 为此问题的 LM 算法设置阻尼因子更新策略 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::SetDampPolicy(DampPolicy policy) {
        this->policy = policy;
        return true;
    }


    /* 为此问题的 LM 算法设置阻尼因子相关参数 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::SetDampParameter(Scalar Lup, Scalar Ldown) {
        this->Lup = Lup;
        this->Ldown = Ldown;
        return true;
    }


    /* LM 算法初始化，确定阻尼因子 lambda 和相关参数的数值 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::InitializeLM(void) {
        this->lambda = this->options.initLambda;
        this->linearizedCost = this->sumCost;
        this->v = 2.0;
        this->invalidCount = 0;
        return true;
    }


    /* 判断是否满足终止迭代条件 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::IsConverged(size_t iter) {
        if (this->linearizedCost - this->sumCost < this->linearizedCost * this->options.minCostDownRate &&
            this->linearizedCost > this->sumCost) {
            std::cout << "<Stop> cost down rate up to threshold, finished." << std::endl;
            return true;
        }
        if (this->normDeltaX < this->options.minNormDeltaX) {
            std::cout << "<Stop> norm delta X up to threshold, finished." << std::endl;
            return true;
        }
        // 更新最小误差，并判断是否终止迭代
        if (this->minCost.second > this->sumCost) {
            this->minCost.first = iter;
            this->minCost.second = this->sumCost;
        }
        if (iter - this->minCost.first > this->options.maxMinCostHold) {
            std::cout << "<Stop> min cost holding times up to threshold, finished." << std::endl;
            return true;
        }
        return false;
    }
    template<typename Scalar>
    bool ProblemBase<Scalar>::IsUnconverged(size_t iter) {
        // 如果连续无效迭代步数过多，则认为求解失败
        if (this->invalidCount > this->options.maxInvalidStep) {
            std::cout << "<Stop> invalid step count up to threshold, finished." << std::endl;
            return true;
        }
        // 如果耗时超出阈值，则终止迭代
        if (this->timeCost > this->options.maxTimeCost) {
            std::cout << "<Stop> solving cost time up to threshold, finished." << std::endl;
            return true;
        }
        return false;
    }


    /* 根据 this->linearSolver 设置的方法求解线性方程组 */
    template<typename Scalar>
    VectorX<Scalar> ProblemBase<Scalar>::SolveLinearFunction(const MatrixX<Scalar> &A, const VectorX<Scalar> &b) {
        switch (this->linearSolver) {
            case LinearSolver::PCG_Solver:
                return this->PreconditionedConjugateGradient(A, b);
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


    /* PCG 求解器 */
    template<typename Scalar>
    VectorX<Scalar> ProblemBase<Scalar>::PreconditionedConjugateGradient(const MatrixX<Scalar> &A, const VectorX<Scalar> &b) {
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
        for (size_t i = 0; i < M_inv_diag.rows(); ++i) {
            if (std::isinf(M_inv_diag(i)) == true) {
                M_inv_diag(i) = 0;
            }
        }
        VectorX<Scalar> z0 = M_inv_diag.array() * r0.array();    // solve M * z0 = r0
        // 取得第一个基底，计算基底权重 alpha，并更新 x
        VectorX<Scalar> p(z0);
        VectorX<Scalar> w = A * p;
        double r0z0 = r0.dot(z0);
        double alpha = r0z0 / p.dot(w);
        x += alpha * p;
        VectorX<Scalar> r1 = r0 - alpha * w;
        // 设定迭代终止的阈值
        double threshold = this->options.minPCGCostDownRate * r0.norm();
        // 迭代求解
        int i = 0;
        while (r1.norm() > threshold && i < maxIteration) {
            i++;
            VectorX<Scalar> z1 = M_inv_diag.array() * r1.array();
            double r1z1 = r1.dot(z1);
            double belta = r1z1 / r0z0;
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


    /* 设置先验信息 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::SetPrior(const MatrixX<Scalar> &prior_J, const VectorX<Scalar> &prior_r) {
        if (prior_J.rows() != prior_r.rows()) {
            return false;
        }
        this->prior_J = prior_J;
        this->prior_r = prior_r;
        return true;
    }


    /* 提取先验信息 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::GetPrior(MatrixX<Scalar> &prior_J, VectorX<Scalar> &prior_r) {
        prior_J = this->prior_J;
        prior_r = this->prior_r;
        return true;
    }


    /* 对整体的雅可比矩阵进行 QR 分解，同时作用在 this->vector_r 上，最后提取出先验信息 */
    /* 保存在 this->prior_J 和 this->prior_r 中 */
    template<typename Scalar>
    bool ProblemBase<Scalar>::ConstructPrior(const size_t margSize) {
        // 若整个大雅可比矩阵的行数小于等于被边缘化的行数，则此步不会存在先验信息
        if (this->matrix_J.rows() <= margSize) {
            this->prior_J.resize(0, 0);
            this->prior_r.resize(0);
            return true;
        }
        // 当且仅当大雅可比矩阵的行数大于被边缘化的行数时，才会存在先验信息
        size_t rows = this->matrix_J.rows();
        size_t cols = this->matrix_J.cols();
        Scalar beta = 0;
        Scalar tau = 0;
        size_t changedRows = rows;
        VectorX<Scalar> tempVector1(cols);
        VectorX<Scalar> tempVector2(rows);
        for (size_t col = 0; col < cols; ++col) {
            size_t idx = rows - changedRows;
            if (changedRows != 1) {
                this->matrix_J.col(col).segment(idx, changedRows)
                    .makeHouseholder(tempVector2, tau, beta);
                this->matrix_J.block(idx, col, changedRows, cols - col)
                    .applyHouseholderOnTheLeft(tempVector2, tau, tempVector1.data());
                this->vector_r.segment(idx, changedRows)
                    .applyHouseholderOnTheLeft(tempVector2, tau, tempVector1.data());
            }
            if (std::fabs(this->matrix_J(idx, col)) > this->options.minPriorItem) {
                --changedRows;
                if (changedRows > 0) {
                    this->matrix_J.col(col).segment(idx + 1, changedRows).setZero();
                }
            } else {
                this->matrix_J.col(col).segment(idx, changedRows).setZero();
            }
            if (changedRows == 0) {
                break;
            }
        }
        // std::cout << "ProblemVO<Scalar>::ConstructPrior :\n  matrix J is\n" << this->matrix_J << "\n  vector r is\n" <<
        //     this->vector_r << std::endl;
        size_t priorRows = rows - changedRows - margSize;
        size_t priorCols = cols - margSize;
        this->prior_J = this->matrix_J.block(margSize, margSize, priorRows, priorCols);
        this->prior_r = this->vector_r.segment(margSize, priorRows);
        return true;
    }


}