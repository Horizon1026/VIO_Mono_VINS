#pragma once

#include <include/ba_solver/root_vio/utility.hpp>

// 全局命名空间定义为 rootVIO
namespace rootVIO {

    // 图优化问题节点的基类，定义节点的通用方法
    template<typename Scalar>
    class VertexBase {
    private:
        // 是否被固定
        bool fixed = false;

    public:
        /* 构造函数与析构函数 */
        VertexBase() { this->fixed = false; }
        virtual ~VertexBase() = default;
    
    public:
        /* 设置是否固定 */
        void SetFixed(bool fixed = true) { this->fixed = fixed; }
        /* 检查是否被固定 */
        bool IsFixed(void) { return this->fixed; }
    };

    // 相机与 IMU 相对位姿类
    // 存储时，姿态采用四元数存储，但是运算过程采用姿态角表示（三自由度）
    template<typename Scalar>
    class VertexExPose : public VertexBase<Scalar> {
    private:
        // 存储的参数和备份参数
        struct Param {
            Quaternion<Scalar> q_bc;
            Vector3<Scalar> t_bc;
        } param, stored;

    public:
        /* 构造函数与析构函数 */
        VertexExPose(const Quaternion<Scalar> &q_bc, const Vector3<Scalar> &t_bc) {
            this->SetParam(q_bc, t_bc);
            this->BackUp();
        }
        ~VertexExPose() {}

    public:
        /* 设置节点参数 */
        void SetParam(const Quaternion<Scalar> &q_bc, const Vector3<Scalar> &t_bc) {
            this->param.q_bc = q_bc;
            this->param.t_bc = t_bc;
        }

        /* 提取节点参数 */
        Quaternion<Scalar> &Get_q_bc(void) { return this->param.q_bc; }
        Vector3<Scalar> &Get_t_bc(void) { return this->param.t_bc; }

        /* 备份当前参数 */
        void BackUp(void) {
            this->stored = this->param;
        }

        /* 更新基于增量更新参数 */
        void UpdateParameters(Vector6<Scalar> dx) {
            // tx, ty, tz, rx, ry, rz
            Quaternion<Scalar> delta_q(1.0, dx[3] * 0.5, dx[4] * 0.5, dx[5] * 0.5);
            Vector3<Scalar> delta_t(dx[0], dx[1], dx[2]);
            this->param.q_bc = this->param.q_bc * delta_q;
            this->param.q_bc.normalize();
            this->param.t_bc = this->param.t_bc + delta_t;
        }

        /* 读取备份参数 */
        void RollBack(void) {
            this->param = this->stored;
        }
    };

    // 相机节点类，保存一帧相机的位姿信息
    // 存储时，姿态采用四元数存储，但是运算过程采用姿态角表示（三自由度）
    template<typename Scalar>
    class VertexCameraPose : public VertexBase<Scalar> {
    private:
        // 存储的参数和备份参数
        struct Param {
            Quaternion<Scalar> q_wb;
            Vector3<Scalar> t_wb;
        } param, stored;

    public:
        /* 构造函数与析构函数 */
        VertexCameraPose(const Quaternion<Scalar> &q_wb, const Vector3<Scalar> &t_wb) {
            this->SetParam(q_wb, t_wb);
            this->BackUp();
        }
        ~VertexCameraPose() {}

    public:
        /* 设置节点参数 */
        void SetParam(const Quaternion<Scalar> &q_wb, const Vector3<Scalar> &t_wb) {
            this->param.q_wb = q_wb;
            this->param.t_wb = t_wb;
        }

        /* 提取节点参数 */
        Quaternion<Scalar> &Get_q_wb(void) { return this->param.q_wb; }
        Vector3<Scalar> &Get_t_wb(void) { return this->param.t_wb; }

        /* 备份当前参数 */
        void BackUp(void) {
            this->stored = this->param;
        }

        /* 更新基于增量更新参数 */
        void UpdateParameters(Vector6<Scalar> dx) {
            // tx, ty, tz, rx, ry, rz
            Quaternion<Scalar> delta_q(1.0, dx[3] * 0.5, dx[4] * 0.5, dx[5] * 0.5);
            Vector3<Scalar> delta_t(dx[0], dx[1], dx[2]);
            this->param.q_wb = this->param.q_wb * delta_q;
            this->param.q_wb.normalize();
            this->param.t_wb = this->param.t_wb + delta_t;
        }

        /* 读取备份参数 */
        void RollBack(void) {
            this->param = this->stored;
        }
    };


    // 特征点节点类，保存一个特征点的位置信息和逆深度信息
    template<typename Scalar>
    class VertexLandmark : public VertexBase<Scalar> {
    private:
        // 存储的参数和备份参数
        struct Param {
            Vector3<Scalar> p_w_or_invdep;
        } param, stored;

    public:
        /* 构造函数与析构函数 */
        VertexLandmark(const Vector3<Scalar> &p_w) {
            this->SetParam(p_w);
            this->BackUp();
        }
        VertexLandmark(const Scalar &invdep) {
            this->SetParam(invdep);
            this->BackUp();
        }
        ~VertexLandmark() {}

    public:
        /* 设置节点参数 */
        void SetParam(const Vector3<Scalar> &p_w) { this->param.p_w_or_invdep = p_w; }
        void SetParam(const Scalar &invdep) { this->param.p_w_or_invdep(0) = invdep; }

        /* 提取节点参数 */
        Vector3<Scalar> &Get_p_w(void) { return this->param.p_w_or_invdep; }
        Scalar &Get_invdep(void) { return this->param.p_w_or_invdep(0); }

        /* 备份当前参数 */
        void BackUp(void) {
            this->stored = this->param;
        }

        /* 更新基于增量更新参数 */
        void UpdateParameters(VectorX<Scalar> dx) {
            // tx, ty, tz / invdep, 0, 0
            size_t size = dx.rows();
            this->param.p_w_or_invdep.head(size) = this->param.p_w_or_invdep.head(size) + dx;
        }
        void UpdateParameters(Scalar dx) {
            // invdep
            this->param.p_w_or_invdep(0) += dx;
        }

        /* 读取备份参数 */
        void RollBack(void) {
            this->param = this->stored;
        }
    };


    // // 特征点节点类，保存一个特征点的位置信息
    // template<typename Scalar>
    // class VertexLandmarkPosi : public VertexBase<Scalar> {
    // private:
    //     // 存储的参数和备份参数
    //     struct Param {
    //         Vector3<Scalar> p_w;
    //     } param, stored;

    // public:
    //     /* 构造函数与析构函数 */
    //     VertexLandmarkPosi(const Vector3<Scalar> &p_w) {
    //         this->SetParam(p_w);
    //         this->BackUp();
    //     }
    //     ~VertexLandmarkPosi() {}

    // public:
    //     /* 设置节点参数 */
    //     void SetParam(const Vector3<Scalar> &p_w) { this->param.p_w = p_w; }

    //     /* 提取节点参数 */
    //     Vector3<Scalar> &Get_p_w(void) { return this->param.p_w; }

    //     /* 备份当前参数 */
    //     void BackUp(void) {
    //         this->stored = this->param;
    //     }

    //     /* 更新基于增量更新参数 */
    //     void UpdateParameters(Vector3<Scalar> dx) {
    //         // tx, ty, tz
    //         this->param.p_w = this->param.p_w + dx;
    //     }

    //     /* 读取备份参数 */
    //     void RollBack(void) {
    //         this->param = this->stored;
    //     }
    // };


    // // 特征点节点类，保存一个特征点的逆深度信息
    // template<typename Scalar>
    // class VertexLandmarkInvdep : public VertexBase<Scalar> {
    // private:
    //     // 存储的参数和备份参数
    //     struct Param {
    //         Vector1<Scalar> invdep;
    //     } param, stored;

    // public:
    //     /* 构造函数与析构函数 */
    //     VertexLandmarkInvdep(const Vector1<Scalar> &invdep) {
    //         this->SetParam(invdep);
    //         this->BackUp();
    //     }
    //     ~VertexLandmarkInvdep() {}

    // public:
    //     /* 设置节点参数 */
    //     void SetParam(const Vector1<Scalar> &invdep) { this->param.invdep = invdep; }

    //     /* 提取节点参数 */
    //     Vector1<Scalar> &Get_invdep(void) { return this->param.invdep; }

    //     /* 备份当前参数 */
    //     void BackUp(void) {
    //         this->stored = this->param;
    //     }

    //     /* 更新基于增量更新参数 */
    //     void UpdateParameters(Vector1<Scalar> dx) {
    //         // invdep
    //         this->param.invdep = this->param.invdep + dx;
    //     }

    //     /* 读取备份参数 */
    //     void RollBack(void) {
    //         this->param = this->stored;
    //     }
    // };


    // 与关键帧绑定的速度和偏差节点
    template<typename Scalar>
    class VertexVelocityBias : public VertexBase<Scalar> {
    private:
        // 存储的参数和备份参数
        struct Param {
            Vector3<Scalar> v_wb;
            Vector3<Scalar> bias_a;
            Vector3<Scalar> bias_g;
        } param, stored;

    public:
        /* 构造函数与析构函数 */
        VertexVelocityBias(const Vector3<Scalar> &v_wb, const Vector3<Scalar> &bias_a, const Vector3<Scalar> &bias_g) {
            this->SetParam(v_wb, bias_a, bias_g);
            this->BackUp();
        }
        ~VertexVelocityBias() {}

    public:
        /* 设置节点参数 */
        void SetParam(const Vector3<Scalar> &v_wb, const Vector3<Scalar> &bias_a, const Vector3<Scalar> &bias_g) {
            this->param.v_wb = v_wb;
            this->param.bias_a = bias_a;
            this->param.bias_g = bias_g;
        }

        /* 提取节点参数 */
        Vector3<Scalar> &Get_v_wb(void) { return this->param.v_wb; }
        Vector3<Scalar> &Get_bias_a(void) { return this->param.bias_a; }
        Vector3<Scalar> &Get_bias_g(void) { return this->param.bias_g; }

        /* 备份当前参数 */
        void BackUp(void) {
            this->stored = this->param;
        }

        /* 更新基于增量更新参数 */
        void UpdateParameters(Vector9<Scalar> dx) {
            // vx, vy, vz, bax, bay, baz, bgx, bgy, bgz
            this->param.v_wb = this->param.v_wb + dx.template head<3>();
            this->param.bias_a = this->param.bias_a + dx.template segment<3>(3);
            this->param.bias_g = this->param.bias_g + dx.template tail<3>();
        }

        /* 读取备份参数 */
        void RollBack(void) {
            this->param = this->stored;
        }
    };

}