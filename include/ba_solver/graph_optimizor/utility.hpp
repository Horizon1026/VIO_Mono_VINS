#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <map>
#include <memory>
#include <list>
#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>

// 命名空间为 GraphOptimizor
namespace GraphOptimizor {
    template<typename Scalar> using Matrix15 = Eigen::Matrix<Scalar, 15, 15>;
    template<typename Scalar> using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    template<typename Scalar> using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    template<typename Scalar> using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
    template<typename Scalar> using Vector15 = Eigen::Matrix<Scalar, 15, 1>;
    template<typename Scalar> using Vector9 = Eigen::Matrix<Scalar, 9, 1>;
    template<typename Scalar> using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
    template<typename Scalar> using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    template<typename Scalar> using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
    template<typename Scalar> using Vector1 = Eigen::Matrix<Scalar, 1, 1>;
    template<typename Scalar> using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar> using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar> using Quaternion = Eigen::Quaternion<Scalar>;


    /* 计算三维向量的反对称矩阵 */
    template<typename Scalar>
    Matrix3<Scalar> SkewSymmetricMatrix(const Vector3<Scalar> &v) {
        Matrix3<Scalar> M;
        M << 0, - v.z(), v.y(),
             v.z(), 0, - v.x(),
             - v.y(), v.x(), 0;
        return M;
    }
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetricMatrix(const Eigen::MatrixBase<Derived> &v) {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -v(2), v(1),
               v(2), typename Derived::Scalar(0), -v(0),
               -v(1), v(0), typename Derived::Scalar(0);
        return ans;
    }

    /* 四元数格式转换 */
    template <typename Derived>
    Eigen::Quaternion<typename Derived::Scalar> Positify(const Eigen::QuaternionBase<Derived> &q) {
        return q;
    }

    /* 基于旋转向量计算四元数增量 */
    template <typename Derived>
    Eigen::Quaternion<typename Derived::Scalar> DeltaQ(const Eigen::MatrixBase<Derived> &theta) {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }


    /* 计算四元数的左乘矩阵 */
    template<typename Scalar>
    Matrix4<Scalar> Qleft(const Quaternion<Scalar> &q) {
        Matrix4<Scalar> Q;
        Q.template block<1, 3>(0, 1) = - q.vec().transpose();
        Q.template block<3, 1>(1, 0) = q.vec();
        Q.template block<3, 3>(1, 1) = SkewSymmetricMatrix(q.vec());
        for (size_t i = 0; i < 4; ++i) {
            Q(i, i) = q.w();
        }
        return Q;
    }
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q) {
        Eigen::Quaternion<typename Derived::Scalar> qq = Positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w();
        ans.template block<1, 3>(0, 1) = - qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec();
        ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + SkewSymmetricMatrix(qq.vec());
        return ans;
    }


    /* 计算四元数的右乘矩阵 */
    template<typename Scalar>
    Matrix4<Scalar> Qright(const Quaternion<Scalar> &q) {
        Matrix4<Scalar> Q;
        Q.template block<1, 3>(0, 1) = - q.vec().transpose();
        Q.template block<3, 1>(1, 0) = q.vec();
        Q.template block<3, 3>(1, 1) = - SkewSymmetricMatrix(q.vec());
        for (size_t i = 0; i < 4; ++i) {
            Q(i, i) = q.w();
        }
        return Q;
    }
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p) {
        Eigen::Quaternion<typename Derived::Scalar> pp = Positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w();
        ans.template block<1, 3>(0, 1) = - pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec();
        ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - SkewSymmetricMatrix(pp.vec());
        return ans;
    }


    /* 计算李代数右雅可比的逆 */
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 3, 3> JacobianRInv(const Eigen::MatrixBase<Derived> &v) {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans.setIdentity();
        Scalar_t theta = v.norm();
        if (theta > 1e-5) {
            Eigen::Matrix<typename Derived::Scalar, 3, 3> K = SkewSymmetricMatrix(v.normalized());
            ans += Scalar_t(0.5) * K + (Scalar_t(1) - (Scalar_t(1) + std::cos(theta)) * theta / (Scalar_t(2) * std::sin(theta))) * K * K;
        }
        return ans;
    }


    /* 计算旋转误差的量级，输出单位为 rad 的误差角 */
    template<typename Scalar>
    Scalar ComputeRotationMagnitude(Matrix3<Scalar> &delta_R) {
        Scalar magnitude = std::fabs(std::acos(
            (delta_R.trace() - Scalar(1.0)) / Scalar(2.0)
        ));
        return magnitude;
    }
    template<typename Scalar>
    Scalar ComputeRotationMagnitude(Quaternion<Scalar> &delta_q) {
        Matrix3<Scalar> delta_R(delta_q);
        return ComputeRotationMagnitude(delta_R);
    }
    template<typename Scalar>
    Scalar ComputeRotationMagnitude(Vector3<Scalar> &delta_r) {
        Quaternion<Scalar> delta_q(
            Scalar(1.0), 
            Scalar(0.5) * delta_r(0),
            Scalar(0.5) * delta_r(1),
            Scalar(0.5) * delta_r(2));
        return ComputeRotationMagnitude(delta_q);
    }


    /* 计算位置误差的量级，输出单位等于输入单位 */
    template<typename Scalar>
    Scalar ComputeTranslationMagnitude(Vector3<Scalar> &delta_t) {
        return delta_t.norm();
    }
}