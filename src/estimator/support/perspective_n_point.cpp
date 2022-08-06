#include <include/estimator/support/perspective_n_point.h>
#include <3rd_library/sophus/se3.hpp>

/* 采用 RANSAC 方法代入 PnP 模型来估计相机位姿 */
void PerspectiveNPointClass::EstimateRotationAndTranslation(std::vector<cv::Point3f> &wordPoints,   // 对应世界坐标系中的 3D 点坐标
                                                            std::vector<cv::Point2f> &pixelPoints,  // 像素平面坐标
                                                            Eigen::Matrix3d &CameraMatrix,          // 相机内参矩阵
                                                            Eigen::Matrix3d &R_cw,          // 求解输出的相机姿态
                                                            Eigen::Vector3d &t_cw,          // 求解输出的相机位置
                                                            std::vector<uchar> &status) {   // 判断 3D 和 2D 点对是否为 inliers 或者 outliers
    // 检查输入数据是否符合规范
    if (wordPoints.size() != pixelPoints.size() || wordPoints.size() < 3) {
        return;
    }

    // 转换格式
    std::vector<Eigen::Vector3d> points3D(wordPoints.size());
    for (unsigned int i = 0; i < points3D.size(); i++) {
        points3D[i] = Eigen::Vector3d(wordPoints[i].x, wordPoints[i].y, wordPoints[i].z);
    }
    std::vector<Eigen::Vector3d> points2D = this->GetNormPointsFromPixelPoints(pixelPoints, CameraMatrix);

    // 调用接口函数
    this->EstimateRotationAndTranslation(points3D, points2D, R_cw, t_cw, status);
}


/* 采用 RANSAC 方法代入 PnP 模型来估计相机位姿 */
void PerspectiveNPointClass::EstimateRotationAndTranslation(std::vector<Eigen::Vector3d> &wordPoints,  // 对应世界坐标系中的 3D 点坐标
                                                            std::vector<Eigen::Vector3d> &normPoints,  // 归一化平面坐标
                                                            Eigen::Matrix3d &R_cw,          // 求解输出的相机姿态
                                                            Eigen::Vector3d &t_cw,          // 求解输出的相机位置
                                                            std::vector<uchar> &status) {   // 判断 3D 和 2D 点对是否为 inliers 或者 outliers
    /* 采用 RANSAC 方法来估计相机位姿 */

    // 如果匹配情况向量不存在，则新建并默认没有 outliers，写入 1 表示匹配正常
    if (status.empty()) {
        status.resize(normPoints.size(), 1);
    }

    // 根据匹配情况向量，构建索引映射表，使得在随机生成点的过程中不会命中 outliers
    std::vector<int> map;
    for (unsigned int i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            map.emplace_back(i);
        }
    }
    if (map.size() < 3) {
        return;
    }

    // 在 RANSAC 迭代过程中，每次选择 3 对有效的点对进行 PnP 求解，并根据结果确定哪些是 inliers，这里构建临时变量
    std::vector<Eigen::Vector3d> tempPoints3D(3);
    std::vector<Eigen::Vector3d> tempPoints2D(3);
    std::vector<Eigen::Vector3d> inliersPoints3D;
    std::vector<Eigen::Vector3d> inliersPoints2D;

    // 构造临时变量，计算某一模型 R 和 t 下每一对点的重投影误差，并定义误差阈值
    std::vector<double> tempErrors;
    double errorThreshold = RANSAC_THRESHOLD;

    // 构造临时配对状态变量，以及记录最佳配对状态结果的变量
    std::vector<uchar> tempStatus = status;
    std::vector<uchar> bestStatus = status;

    // 定义最大迭代次数
    int maxEpoch = 150;

    // 保存历史最小的模型误差，以及最佳的 R 和 t（选择函数输入的 R 和 t 作为初值）
    double minError = INFINITY;
    Eigen::Matrix3d bestR = R_cw;
    Eigen::Vector3d bestt = t_cw;

    // 开始迭代，通过 RANSAC 方法估计出最佳的位姿
    while (maxEpoch > 0) {
        maxEpoch--;

        // 利用事先构造的映射表，跳过所有 status 已经是 0 的点对，之后随机生成 3 个索引，从而构造出临时点对
        std::set<int> subIndex;
        while (subIndex.size() < 3) {
            subIndex.insert(rand() % map.size());
        }
        std::set<int>::iterator it;
        int idx = 0;
        for (it = subIndex.begin(); it != subIndex.end(); it++) {
            tempPoints3D[idx] = wordPoints[map[*it]];
            tempPoints2D[idx] = normPoints[map[*it]];
            idx++;
        }

        // 如果选出的三个点共线，则不进行 PnP 迭代优化
        if (this->IsCollineation(tempPoints2D) == true) {
            continue;
        }

        // 用临时点对估计出相机位姿
        Eigen::Matrix3d tempR = bestR;
        Eigen::Vector3d tempt = bestt;
        this->SolvePnP(tempPoints3D, tempPoints2D, tempR, tempt);
    
        // 根据当前的模型 R 和 t，计算出所有点对的模型误差
        tempErrors = this->ComputeReprojectionError(wordPoints, normPoints, tempR, tempt);

        // 设定阈值，判断出 outliers 和 inliers，并计算出 inliers 的平均误差
        double inliersMeanError = 0.0;
        int inliersCount = 0;
        int outliersCount = 0;
        for (unsigned long i = 0; i < tempStatus.size(); i++) {
            if (status[i] == 1 && tempErrors[i] < errorThreshold) {
                inliersMeanError += tempErrors[i];
                inliersCount++;
                tempStatus[i] = 1;
            } else {
                outliersCount++;
                tempStatus[i] = 0;
            }
        }
        inliersMeanError /= double(inliersCount);

        // 如果本次的平均误差比上一次的要更小，则以本次结果取代上一次结果
        if (inliersMeanError < minError) {
            // 保存本次的评价结果
            minError = inliersMeanError;

            // 保存本次的配对结果
            bestStatus = tempStatus;

            // 从所有的 inliers 中，随机选择最多 USED_INLIERS_NUMS 对点对
            std::set<int> randIndex;
            for (unsigned int i = 0; i < USED_INLIERS_NUMS; i++) {
                randIndex.insert(rand() % normPoints.size());
            }
            inliersPoints3D.clear();
            inliersPoints2D.clear();
            int idx = 0;
            std::set<int>::iterator it;
            for (it = randIndex.begin(); it != randIndex.end(); it++) {
                if (tempStatus[*it] == 1) {
                    inliersPoints3D.emplace_back(wordPoints[*it]);
                    inliersPoints2D.emplace_back(normPoints[*it]);
                    idx++;
                }
            }

            // 如果挑选出来的 inliers 数量足够，则由所有的 inliers 计算本质矩阵
            if (inliersPoints3D.size() > 3) {
                this->SolvePnP(inliersPoints3D, inliersPoints2D, tempR, tempt);
                bestR = tempR;
                bestt = tempt;
            } else {
                bestR = tempR;
                bestt = tempt;
            }
        }

        // 如果已经没有 outliers 了，则直接结束
        if (outliersCount == 0) {
            break;
        }
    }

    // 更新 outliers 检测结果
    status = bestStatus;

    // 返回优化结果
    R_cw = bestR;
    t_cw = bestt;
}

/* 利用所有的 3D 点和 2D 点对，估计相机位姿 */
void PerspectiveNPointClass::SolvePnP(std::vector<cv::Point3f> &wordPoints,     // 对应世界坐标系中的 3D 点坐标
                                      std::vector<cv::Point2f> &pixelPoints,    // 像素平面坐标
                                      Eigen::Matrix3d &CameraMatrix,            // 相机内参
                                      Eigen::Matrix3d &R_cw,    // 求解输出的相机姿态
                                      Eigen::Vector3d &t_cw) {  // 求解输出的相机位置
    // 检查输入数据是否符合规范
    if (wordPoints.size() != pixelPoints.size() || wordPoints.size() < 3) {
        return;
    }

    // 转换格式
    std::vector<Eigen::Vector3d> points3D(wordPoints.size());
    for (unsigned int i = 0; i < points3D.size(); i++) {
        points3D[i] = Eigen::Vector3d(wordPoints[i].x, wordPoints[i].y, wordPoints[i].z);
    }
    std::vector<Eigen::Vector3d> points2D = this->GetNormPointsFromPixelPoints(pixelPoints, CameraMatrix);

    // 调用接口函数
    this->SolvePnP(points3D, points2D, R_cw, t_cw);
}


/* 利用所有的 3D 点和 2D 点对，估计相机位姿 */
void PerspectiveNPointClass::SolvePnP(std::vector<Eigen::Vector3d> &wordPoints,     // 对应世界坐标系中的 3D 点坐标
                                      std::vector<Eigen::Vector3d> &normPoints,     // 归一化平面坐标
                                      Eigen::Matrix3d &R_cw,    // 求解输出的相机姿态
                                      Eigen::Vector3d &t_cw) {  // 求解输出的相机位置
    // 设定最大迭代次数
    unsigned int maxEpoch = 100;

    // 记录上一次的总误差
    double cost = 0;
    double last_cost = -1;
    while (maxEpoch--) {
        // 初始化高斯牛顿迭代方程Hx=b
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        // 遍历每一个点，构造高斯牛顿迭代方程
        cost = 0;
        for (unsigned int i = 0; i < normPoints.size(); i++) {

            // 将世界坐标系的 3D 点，旋转平移到相机坐标系，并提前计算Z轴的逆
            Eigen::Vector3d point_c = R_cw * wordPoints[i] + t_cw;
            double Z_inv = 1.0 / point_c[2];
            double Z2_inv = Z_inv * Z_inv;

            // 将相机坐标系中的点，变换到归一化平面，计算此点的重投影误差
            Eigen::Vector2d point = Eigen::Vector2d(point_c[0] / point_c[2], point_c[1] / point_c[2]);
            Eigen::Vector2d error = normPoints[i].block<2, 1>(0, 0) - point;
            cost += error.squaredNorm();

            // 计算雅可比矩阵
            Eigen::Matrix<double, 2, 6> J;
            J << - Z_inv,
                 0,
                 point_c[0] * Z2_inv,
                 point_c[0] * point_c[1] * Z2_inv,
                 - 1.0 - point_c[0] * point_c[0] * Z2_inv,
                 point_c[1] * Z_inv,
                 0,
                 - Z_inv,
                 point_c[1] * Z2_inv,
                 1.0 + point_c[1] * point_c[1] * Z2_inv,
                 - point_c[0] * point_c[1] * Z2_inv,
                 - point_c[0] * Z_inv;

            // 构造高斯牛顿迭代方程
            H += J.transpose() * J;
            b += - J.transpose() * error;
        }

        // 采用 LDLT 分解方法求解
        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);

        // 检查结果是否可以更新
        if (std::isnan(dx[0])) {
            break;
        } else if (last_cost > 0 && cost > last_cost) {
            break;
        }

        //将位姿转化为李代数进行更新
        Sophus::SE3d T(R_cw, t_cw);
        T = Sophus::SE3d::exp(dx) * T;
        R_cw = T.rotationMatrix();
        t_cw = T.translation();

        //如果更新量已经足够小，则不用再继续迭代
        last_cost = cost;
        if (dx.norm() < CONVERGE_THRESHOLD) {
            break;
        }
    }
}


/* 像素坐标变换为归一化坐标 */
std::vector<Eigen::Vector3d> PerspectiveNPointClass::GetNormPointsFromPixelPoints(std::vector<cv::Point2f> &pixelPoints,
                                                                                  Eigen::Matrix3d &CameraMatrix) {
    // 从相机内参矩阵中提取出内参参数
    double fx = CameraMatrix(0, 0);
    double fy = CameraMatrix(1, 1);
    double cx = CameraMatrix(0, 2);
    double cy = CameraMatrix(1, 2);

    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints(pixelPoints.size());
    for (unsigned int i = 0; i < pixelPoints.size(); i++) {
        normPoints[i] = Eigen::Vector3d((pixelPoints[i].x - cx) / fx, (pixelPoints[i].y - cy) / fy, 1.0);
    }
    return normPoints;
}


/* 判断 3 个 2D 点是否共线 */
bool PerspectiveNPointClass::IsCollineation(std::vector<Eigen::Vector3d> &points2D) {
    // 如果输入的点数小于 3，则肯定不会共线
    if (points2D.size() < 3) {
        return false;
    }

    // 只判断前三个点是否共线，通过计算斜率进行比较来判断
    double slope01 = abs((points2D[0](0, 0) - points2D[1](0, 0)) / (points2D[0](1, 0) - points2D[1](1, 0)));
    double slope12 = abs((points2D[1](0, 0) - points2D[2](0, 0)) / (points2D[1](1, 0) - points2D[2](1, 0)));
    if (abs(slope01 - slope12) < 1e-10) {
        return true;
    } else {
        return false;
    }
}


/* 指定位姿的情况下，为每一对点对计算重投影误差 */
std::vector<double> PerspectiveNPointClass::ComputeReprojectionError(const std::vector<Eigen::Vector3d> &points3D,
                                                                     const std::vector<Eigen::Vector3d> &points2D,
                                                                     const Eigen::Matrix3d &R,
                                                                     const Eigen::Vector3d &t) {
    std::vector<double> errors(points3D.size());

    // 遍历每一对点对
    for (unsigned int i = 0; i < points3D.size(); i++) {
        Eigen::Vector3d reprojectionP = R * points3D[i] + t;
        reprojectionP /= reprojectionP(2, 0);
        Eigen::Vector3d residual = points2D[i] - reprojectionP;
        errors[i] = residual.block<2, 1>(0, 0).norm();
    }

    // 返回计算的误差结果
    return errors;
}