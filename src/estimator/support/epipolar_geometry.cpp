#include <include/estimator/support/epipolar_geometry.h>
#include <Eigen/Dense>


/* 带参数的构造函数 */
EpipolarGeometryClass::EpipolarGeometryClass(double threshold, int usedInliers) {
    RANSAC_THRESHOLD = threshold;
    USED_INLIERS_NUMS = usedInliers;
}


/* 根据两对匹配的特征点的像素坐标，估计出两帧之间的相对位姿 */
bool EpipolarGeometryClass::EstimateRotationAndTranslation(std::vector<cv::Point2f> &pixelPoints0,   // 图像 0 中的特征点像素坐标
                                         std::vector<cv::Point2f> &pixelPoints1,   // 图像 1 中的特征点像素坐标
                                         Eigen::Matrix3d &CameraMatrix,            // 相机内参矩阵
                                         std::vector<uchar> &status,               // 图像 0 和图像 1 的特征点匹配情况
                                         Eigen::Matrix3d &R_c1c0,                  // 输出分解出来的相对姿态
                                         Eigen::Vector3d &t_c1c0) {                // 输出分解出来的相对位置
    // 检查输入数据是否符合要求
    if (pixelPoints0.size() != pixelPoints1.size() || pixelPoints0.size() < 5 || pixelPoints1.size() < 5) {
        return false;
    }

    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints0 = this->GetNormPointsFromPixelPoints(pixelPoints0, CameraMatrix);
    std::vector<Eigen::Vector3d> normPoints1 = this->GetNormPointsFromPixelPoints(pixelPoints1, CameraMatrix);

    // 调用接口函数
    return this->EstimateRotationAndTranslation(normPoints0, normPoints1, status, R_c1c0, t_c1c0);
}


/* 根据两对匹配的特征点的像素坐标，估计出两帧之间的相对位姿 */
bool EpipolarGeometryClass::EstimateRotationAndTranslation(std::vector<Eigen::Vector3d> &normPoints0,// 图像 0 中的归一化平面坐标
                                                           std::vector<Eigen::Vector3d> &normPoints1,// 图像 1 中的归一化平面坐标
                                                           std::vector<uchar> &status,               // 图像 0 和图像 1 的特征点匹配情况
                                                           Eigen::Matrix3d &R_c1c0,                  // 输出分解出来的相对姿态
                                                           Eigen::Vector3d &t_c1c0) {                // 输出分解出来的相对位置
    /* 采用 RANSAC 方法估计本质矩阵和位姿 */

    // 如果匹配情况向量不存在，则新建并默认为无 outliers，写入 1 表示对应点对匹配正常
    if (status.empty()) {
        status.resize(normPoints0.size(), 1);
    }

    // 根据匹配情况向量，构建索引映射表，使得在随机生成点的过程中不会命中 outliers
    std::vector<int> map;
    for (unsigned int i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            map.emplace_back(i);
        }
    }
    
    // 在 RANSAC 迭代过程中，每次选择 5 对有效的特征点进行本质矩阵的计算，并根据结果确定哪些是 inliers，这里构建临时变量
    std::vector<Eigen::Vector3d> tempPoints0(5);
    std::vector<Eigen::Vector3d> tempPoints1(5);
    std::vector<Eigen::Vector3d> inliersPoints0;
    std::vector<Eigen::Vector3d> inliersPoints1;
    
    // 构造临时变量，计算某一模型 E 下的每一对特征点的代入误差，并定义误差阈值
    std::vector<double> tempErrors;
    double errorThreshold = RANSAC_THRESHOLD;

    // 构造临时配对状态变量，以及记录最佳配对状态结果的变量
    std::vector<uchar> tempStatus = status;
    std::vector<uchar> bestStatus = status;

    // 定义最大迭代次数
    int maxEpoch = 50;

    // 保存历史最小的模型误差，以及最佳的本质矩阵 E
    double minError = INFINITY;
    Eigen::Matrix3d bestE;

    // 开始迭代，通过 RANSAC 方法估计出最佳的本质矩阵 bestE
    while (maxEpoch > 0) {
        maxEpoch--;

        // 利用事先构造的映射表，跳过所有 status 已经是 0 的特征点对，之后随机生成 5 个索引，从而构造出临时特征点对
        std::set<int> subIndex;
        while (subIndex.size() < 5) {
            subIndex.insert(rand() % map.size());
        }
        std::set<int>::iterator it;
        int idx = 0;
        for (it = subIndex.begin(); it != subIndex.end(); it++) {
            tempPoints0[idx] = normPoints0[map[*it]];
            tempPoints1[idx] = normPoints1[map[*it]];
            idx++;
        }

        // 用临时特征点对计算出本质矩阵 E
        double invalidError;
        Eigen::Matrix3d tempE = this->FindEssentialMatrix(tempPoints0, tempPoints1, invalidError);

        // 根据当前模型 E，计算出所有特征点对的模型误差
        tempErrors = this->ComputeEpipolarGeometryError(normPoints0, normPoints1, tempE);

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

        // 如果本次的平均误差比上一次的要更小，则以本次结果取代上一次的结果
        if (inliersMeanError < minError) {
            // 保存本次的评价结果
            minError = inliersMeanError;

            // 保存本次的配对结果
            bestStatus = tempStatus;

            // 从所有的 inliers 中，随机选择最多 USED_INLIERS_NUMS 对匹配点
            std::set<int> randIndex;
            for (unsigned int i = 0; i < USED_INLIERS_NUMS; i++) {
                randIndex.insert(rand() % normPoints0.size());
            }
            inliersPoints0.clear();
            inliersPoints1.clear();
            int idx = 0;
            std::set<int>::iterator it;
            for (it = randIndex.begin(); it != randIndex.end(); it++) {
                if (tempStatus[*it] == 1) {
                    inliersPoints0.emplace_back(normPoints0[*it]);
                    inliersPoints1.emplace_back(normPoints1[*it]);
                    idx++;
                }
            }

            // 如果挑选出来的 inliers 数量足够，则由所有的 inliers 计算本质矩阵
            if (inliersPoints0.size() > 4) {
                bestE = this->FindEssentialMatrix(inliersPoints0, inliersPoints1, invalidError);
            } else {
                bestE = tempE;
            }
        }

        // 如果已经没有 outliers，则直接结束
        if (outliersCount == 0) {
            break;
        }
    }

    // 更新配对结果
    status = bestStatus;

    // 分解本质矩阵 E，获得位姿
    int wrongNums = this->RecoverPoseFromEssentialMatrix(bestE, inliersPoints0, inliersPoints1, R_c1c0, t_c1c0);
    
    // 根据三角测量有效的点对数量，判断结果是否可靠
    if (3 * wrongNums > inliersPoints0.size()) {
        return false;
    } else {
        return true;
    }
}


/* 根据两对匹配的特征点的坐标，判断出哪些是 inliers */
bool EpipolarGeometryClass::FindInliersWithEssentialMatrix(std::vector<cv::Point2f> &pixelPoints0,   // 图像 0 中的特征点像素坐标
                                                           std::vector<cv::Point2f> &pixelPoints1,   // 图像 1 中的特征点像素坐标
                                                           Eigen::Matrix3d &CameraMatrix,            // 相机内参矩阵
                                                           std::vector<uchar> &status) {             // 图像 0 和图像 1 的特征点匹配情况
    // 检查输入数据是否符合要求
    if (pixelPoints0.size() != pixelPoints1.size() || pixelPoints0.size() < 5 || pixelPoints1.size() < 5) {
        return false;
    }

    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints0 = this->GetNormPointsFromPixelPoints(pixelPoints0, CameraMatrix);
    std::vector<Eigen::Vector3d> normPoints1 = this->GetNormPointsFromPixelPoints(pixelPoints1, CameraMatrix);

    // 调用接口函数
    return this->FindInliersWithEssentialMatrix(normPoints0, normPoints1, status);
}

/* 根据两对匹配的特征点的坐标，判断出哪些是 inliers */
bool EpipolarGeometryClass::FindInliersWithEssentialMatrix(std::vector<Eigen::Vector3d> &normPoints0,// 图像 0 中的归一化平面坐标
                                                           std::vector<Eigen::Vector3d> &normPoints1,// 图像 1 中的归一化平面坐标
                                                           std::vector<uchar> &status) {             // 图像 0 和图像 1 的特征点匹配情况
    /* 采用 RANSAC 方法估计本质矩阵和位姿 */

    // 如果匹配情况向量不存在，则新建并默认为无 outliers，写入 1 表示对应点对匹配正常
    if (status.empty()) {
        status.resize(normPoints0.size(), 1);
    }

    // 根据匹配情况向量，构建索引映射表，使得在随机生成点的过程中不会命中 outliers
    std::vector<int> map;
    for (unsigned long i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            map.emplace_back(i);
        }
    }
    
    // 在 RANSAC 迭代过程中，每次选择 5 对有效的特征点进行本质矩阵的计算，并根据结果确定哪些是 inliers，这里构建临时变量
    std::vector<Eigen::Vector3d> tempPoints0(5);
    std::vector<Eigen::Vector3d> tempPoints1(5);
    
    // 构造临时变量，计算某一模型 E 下的每一对特征点的代入误差，并定义误差阈值
    std::vector<double> tempErrors;
    double errorThreshold = RANSAC_THRESHOLD;

    // 构造临时配对状态变量
    std::vector<uchar> tempStatus = status;
    std::vector<uchar> bestStatus = status;

    // 定义最大迭代次数
    int maxEpoch = 50;

    // 保存历史最小模型误差，以及最佳的本质矩阵 E
    double minError = INFINITY;

    // 开始迭代，通过 RANSAC 方法估计出最佳的本质矩阵 bestE
    while (maxEpoch > 0) {
        maxEpoch--;

        // 利用事先构造的映射表，跳过所有 status 已经是 0 的特征点对，之后随机生成 5 个索引，从而构造出临时特征点对
        std::set<int> subIndex;
        while (subIndex.size() < 5) {
            subIndex.insert(rand() % map.size());
        }
        std::set<int>::iterator it;
        int idx = 0;
        for (it = subIndex.begin(); it != subIndex.end(); it++) {
            tempPoints0[idx] = normPoints0[map[*it]];
            tempPoints1[idx] = normPoints1[map[*it]];
            idx++;
        }

        // 用临时特征点对计算出本质矩阵 E
        double invalidError;
        Eigen::Matrix3d tempE = this->FindEssentialMatrix(tempPoints0, tempPoints1, invalidError);

        // 根据当前模型 E，计算出所有特征点对的模型误差
        tempErrors = this->ComputeEpipolarGeometryError(normPoints0, normPoints1, tempE);

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

        // 如果本次的平均误差比上一次的要更小，则以本次结果取代上一次的结果
        if (inliersMeanError < minError) {
            // 保存本次的评价结果
            minError = inliersMeanError;

            // 保存本次点对匹配情况
            bestStatus = tempStatus;
        }

        // 如果已经没有 outliers，则直接结束
        if (outliersCount == 0) {
            break;
        }
    }

    // 更新点对匹配情况
    status = bestStatus;
    return true;
}

/* 用指定数量的匹配点，估计本质矩阵 */
Eigen::Matrix3d EpipolarGeometryClass::FindEssentialMatrix(std::vector<cv::Point2f> &pixelPoints0,
                                                           std::vector<cv::Point2f> &pixelPoints1,
                                                           Eigen::Matrix3d &CameraMatrix,
                                                           double &error) {
    // 检查输入数据是否符合要求
    if (pixelPoints0.size() != pixelPoints1.size() || pixelPoints0.size() < 5 || pixelPoints1.size() < 5) {
        return Eigen::Matrix3d::Zero();
    }
    
    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints0 = this->GetNormPointsFromPixelPoints(pixelPoints0, CameraMatrix);
    std::vector<Eigen::Vector3d> normPoints1 = this->GetNormPointsFromPixelPoints(pixelPoints1, CameraMatrix);

    // 调用接口函数
    return this->FindEssentialMatrix(normPoints0, normPoints1, error);
}


/* 使用全部的匹配点，估计本质矩阵 */
Eigen::Matrix3d EpipolarGeometryClass::FindEssentialMatrix(std::vector<Eigen::Vector3d> &normPoints0,
                                                           std::vector<Eigen::Vector3d> &normPoints1,
                                                           double &error) {
    // 估计出最佳的本质矩阵，同时分解出其最佳的相对位姿 R 和 t
    std::vector<double> meanErrors;
    std::vector<Eigen::Matrix3d> allE = this->FindAllEssentialMatriceFivePoints(normPoints0, normPoints1, meanErrors);
    Eigen::Matrix3d bestE = Eigen::Matrix3d::Zero();

    // // 采用三角测量深度不为整数的错误点的个数，以及匹配点在某 E 矩阵下的平均误差来综合衡量
    // int wrongNums = normPoints0.size();
    // double meanError = INFINITY;
    // // 遍历所有可能的本质矩阵，找到最好的那个
    // for (unsigned int i = 0; i < allE.size(); i++) {
    //     Eigen::Matrix3d tempR;
    //     Eigen::Vector3d tempt;
    //     int cnt = RecoverPoseFromEssentialMatrix(allE[i], normPoints0, normPoints1, tempR, tempt);
    //     if (wrongNums >= cnt) {
    //         wrongNums = cnt;
    //         if (meanError > meanErrors[i]) {
    //             meanError = meanErrors[i];
    //             bestE = allE[i];
    //             error = meanError;
    //         }
    //     }
    // }

    // 采用匹配点在某 E 矩阵下的平均误差来衡量
    double meanError = INFINITY;
    // 遍历所有可能的本质矩阵，找到最好的那个
    for (unsigned int i = 0; i < allE.size(); i++) {
        if (meanError > meanErrors[i]) {
            meanError = meanErrors[i];
            bestE = allE[i];
            error = meanError;
        }
    }

    // 返回最佳的本质矩阵
    return bestE;
}


/* 从 E 矩阵中分解出 4 种位姿 */
void EpipolarGeometryClass::DecomposeEssentialMatrix(Eigen::Matrix3d &E,    // 输入本质矩阵
                                                     Eigen::Matrix3d &R0,   // 输出可能的两种相对位姿 R_c1_c0
                                                     Eigen::Matrix3d &R1,
                                                     Eigen::Vector3d &t0,   // 输出可能的两种相对位置 t_c1_c0
                                                     Eigen::Vector3d &t1) {
    // 对本质矩阵 E 做 SVD 分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 若 U 或者 V 的行列时小于零，直接取反
    if (U.determinant() < 0) {
        U = U * -1.0;
    }
    if (V.determinant() < 0) {
        V = V * -1.0;
    }

    // 定义旋转矩阵 R(PI / 2) 和反向的旋转矩阵 R(-PI / 2)，用于计算本质矩阵分解结果
    Eigen::Matrix3d R;
    R << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    // 通过解析解计算姿态 R 和位置 t
    R0 = U * R * V.transpose();
    R1 = U * R.transpose() * V.transpose();
    t0 = U.col(2);
    t1 = - t0;
}


/* 基于两幅图像的位姿，对一对匹配的特征点进行三角测量，得到其3D坐标。输入的点坐标是矫正后归一化平面的坐标 */
Eigen::Vector3d EpipolarGeometryClass::TriangulateOnePoint(Eigen::Matrix3d &R0, Eigen::Vector3d &t0,           // 前一帧图像的位姿，R0_cw，t0_cw
                                                           Eigen::Matrix3d &R1, Eigen::Vector3d &t1,           // 后一帧图像的位姿，R1_cw，t1_cw
                                                           Eigen::Vector3d &point0, Eigen::Vector3d &point1) { // 两帧图像中对应点的归一化平面坐标（矫正后）
    // 将位置和姿态捆绑成一个矩阵
    Eigen::Matrix<double, 3, 4> pose0;
    pose0.block<3, 3>(0, 0) = R0;
    pose0.block<3, 1>(0, 3) = t0;
    Eigen::Matrix<double, 3, 4> pose1;
    pose1.block<3, 3>(0, 0) = R1;
    pose1.block<3, 1>(0, 3) = t1;
    
    // 构造三角测量约束方程 Ax=0
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
    A.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
    A.row(2) = point1[0] * pose1.row(2) - pose1.row(0);
    A.row(3) = point1[1] * pose1.row(2) - pose1.row(1);

    // 通过 SVD 方法求解齐次线性方程，直接取出 A 的 SVD 分解的 V 矩阵的最右边一列
    Eigen::Vector4d x = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // 返回结果
    Eigen::Vector3d point;
    point << x[0] / x[3], x[1] / x[3], x[2] / x[3];
    return point;
}


/* 从本质矩阵中恢复出帧间位姿 R_c1c0 和 t_c1c0，并返回不符合这一相对位姿的特征点对的个数 */
int EpipolarGeometryClass::RecoverPoseFromEssentialMatrix(Eigen::Matrix3d &E,                     // 输入本质矩阵
                                                          std::vector<cv::Point2f> &pixelPoints0, // 输入图像 0 的特征点像素坐标
                                                          std::vector<cv::Point2f> &pixelPoints1, // 输入图像 1 的特征点像素坐标
                                                          Eigen::Matrix3d &CameraMatrix,          // 输入相机内参矩阵
                                                          Eigen::Matrix3d &R_c1c0,                // 输出分解出来的相对姿态
                                                          Eigen::Vector3d &t_c1c0) {              // 输出分解出来的相对位置
    // 检查输入数据是否符合要求
    if (pixelPoints0.size() < 1 || pixelPoints1.size() < 1 || pixelPoints0.size() != pixelPoints1.size()) {
        return 0;
    }

    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints0 = this->GetNormPointsFromPixelPoints(pixelPoints0, CameraMatrix);
    std::vector<Eigen::Vector3d> normPoints1 = this->GetNormPointsFromPixelPoints(pixelPoints1, CameraMatrix);

    // 调用接口函数
    return this->RecoverPoseFromEssentialMatrix(E, normPoints0, normPoints1, R_c1c0, t_c1c0);
}


/* 从本质矩阵中恢复出帧间位姿 R_c1c0 和 t_c1c0，并返回不符合这一相对位姿的特征点对的个数 */
int EpipolarGeometryClass::RecoverPoseFromEssentialMatrix(Eigen::Matrix3d &E,                          // 输入本质矩阵
                                                          std::vector<Eigen::Vector3d> &normPoints0,   // 输入图像 0 的特征点像素坐标
                                                          std::vector<Eigen::Vector3d> &normPoints1,   // 输入图像 1 的特征点像素坐标
                                                          Eigen::Matrix3d &R_c1c0,                     // 输出分解出来的相对姿态
                                                          Eigen::Vector3d &t_c1c0) {                   // 输出分解出来的相对位置
    // 分解本质矩阵，分解所得为 R_c1c0 和 t_c1c0
    Eigen::Matrix3d R0, R1;
    Eigen::Vector3d t0, t1;
    this->DecomposeEssentialMatrix(E, R0, R1, t0, t1);

    // 把四种组合情况分别代入归一化平面点对，三角测量得到3D坐标，判断深度以选择最佳位姿
    std::vector<std::pair<int, std::pair<Eigen::Matrix3d, Eigen::Vector3d>>> test(4);
    test[0] = std::make_pair(0, std::pair<Eigen::Matrix3d, Eigen::Vector3d>(R0, t0));
    test[1] = std::make_pair(0, std::pair<Eigen::Matrix3d, Eigen::Vector3d>(R1, t0));
    test[2] = std::make_pair(0, std::pair<Eigen::Matrix3d, Eigen::Vector3d>(R0, t1));
    test[3] = std::make_pair(0, std::pair<Eigen::Matrix3d, Eigen::Vector3d>(R1, t1));
    Eigen::Matrix3d R_c0c0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_c0c0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pos;
    for (unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < normPoints0.size(); j++) {
            //三角测量出此特征点基于前一帧相机坐标系和后一帧的 3D 位置，如果其 Z 轴为正，则认为可信
            pos = this->TriangulateOnePoint(R_c0c0, t_c0c0, test[i].second.first, test[i].second.second, normPoints0[j], normPoints1[j]);
            // P_c0
            if (pos[2] > 0) {
                // P_c1 = R_c1c0 * P_c0 + t_c1c0
                pos = test[i].second.first * pos + test[i].second.second;
                if (pos[2] > 0) {
                    test[i].first++;
                }
            }
        }
    }

    // 从四个测试模型中选择可信点最多的一个作为结果
    int maxScore = -1;
    int bestIndex = -1;
    int sumScore = 0;
    for (unsigned int i = 0; i < 4; i++) {
        sumScore += test[i].first;
        if (test[i].first > maxScore) {
            maxScore = test[i].first;
            bestIndex = i;
        }
    }
    R_c1c0 = test[bestIndex].second.first;
    t_c1c0 = test[bestIndex].second.second;

    // 返回不符合分解出来的位姿的特征点对数
    return sumScore - maxScore;
}

/* 用五点法估计所有可能的本质矩阵 */
std::vector<Eigen::Matrix3d> EpipolarGeometryClass::FindAllEssentialMatriceFivePoints(std::vector<cv::Point2f> &pixelPoints0,
                                                                                       std::vector<cv::Point2f> &pixelPoints1,
                                                                                       Eigen::Matrix3d &CameraMatrix,
                                                                                       std::vector<double> &meanErrors) {
    // 检查输入数据是否符合要求
    if (pixelPoints0.size() != pixelPoints1.size() || pixelPoints0.size() < 5 || pixelPoints1.size() < 5) {
        return std::vector<Eigen::Matrix3d>();
    }

    // 计算归一化平面坐标
    std::vector<Eigen::Vector3d> normPoints0 = this->GetNormPointsFromPixelPoints(pixelPoints0, CameraMatrix);
    std::vector<Eigen::Vector3d> normPoints1 = this->GetNormPointsFromPixelPoints(pixelPoints1, CameraMatrix);

    // 调用接口函数
    return this->FindAllEssentialMatriceFivePoints(normPoints0, normPoints1, meanErrors);
}


/* 用五点法估计所有可能的本质矩阵 */
std::vector<Eigen::Matrix3d> EpipolarGeometryClass::FindAllEssentialMatriceFivePoints(std::vector<Eigen::Vector3d> &normPoints0,
                                                                                       std::vector<Eigen::Vector3d> &normPoints1,
                                                                                       std::vector<double> &meanErrors) {
    std::vector<Eigen::Matrix3d> allE;
    
    // 利用归一化平面坐标的对极约束，构造方程组 Qx = 0
    cv::Mat Q(normPoints0.size(), 9, CV_64F);
    for (unsigned int i = 0; i < normPoints0.size(); i++) {
        double u0 = normPoints0[i](0, 0);
        double v0 = normPoints0[i](1, 0);
        double u1 = normPoints1[i](0, 0);
        double v1 = normPoints1[i](1, 0);
        Q.at<double>(i, 0) = u1 * u0;
        Q.at<double>(i, 1) = u1 * v0;
        Q.at<double>(i, 2) = u1;
        Q.at<double>(i, 3) = v1 * u0;
        Q.at<double>(i, 4) = v1 * v0;
        Q.at<double>(i, 5) = v1;
        Q.at<double>(i, 6) = u0;
        Q.at<double>(i, 7) = v0;
        Q.at<double>(i, 8) = 1.0;
    }

    /* 因为一共只有 5 个方程，但是有 9 个变量，因此单单这个方程 Ax=0 的解的零空间就有 4 个自由度，可以写成 E = xX + yY + zZ + wW，大写为矩阵，小写为系数 */
    /* 但是因为矩阵 E 的性质，增加了一项约束： E * E.transpose() * E - 0.5 * trace(E * E.transpose()) * E = 0 */
    /* 将 E = xX + yY + zZ + wW 代入到 E * E.transpose() * E - 0.5 * trace(E * E.transpose()) * E = 0 中，可以得到 9 个三元三次方程 */
    /* 从而可以建立等效方程 A * c = 0，其中 A 矩阵是 10 * 20， c 向量是 x，y，z 的 20 种组合 */
    /* 但是三元三次方程不好求解，因此可以考虑先求解出一项 z，再带回去求解 x 和 y。 */

    // 求解出 E 的解空间 [X|Y|Z|W]，并由于 E 本身尺度不唯一，所以指定系数 w = 1
    cv::Mat U, W, Vt;
    cv::SVD::compute(Q, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat EE = cv::Mat(Vt.t()).colRange(5, 9) * 1.0;

    // 构造出 A 矩阵（10 * 20），并进行高斯-约旦消元
    cv::Mat A(10, 20, CV_64F);
    EE = EE.t();
    this->GetCoeffMatrixFivePoints(EE.ptr<double>(), A.ptr<double>());
    EE = EE.t();
    A = A.colRange(0, 10).inv() * A.colRange(10, 20);

    // 构造以 z 的多项式为系数的线性方程组 B * b = 0
    double b[3 * 13];
    cv::Mat B(3, 13, CV_64F, b);
    for (int i = 0; i < 3; i++) {
        cv::Mat arow1 = A.row(i * 2 + 4) * 1.0;
        cv::Mat arow2 = A.row(i * 2 + 5) * 1.0;
        cv::Mat row1(1, 13, CV_64F, cv::Scalar(0.0));
        cv::Mat row2(1, 13, CV_64F, cv::Scalar(0.0));
        row1.colRange(1, 4)  = arow1.colRange(0, 3) * 1.0;
        row1.colRange(5, 8)  = arow1.colRange(3, 6) * 1.0;
        row1.colRange(9, 13) = arow1.colRange(6, 10) * 1.0;
        row2.colRange(0, 3)  = arow2.colRange(0, 3) * 1.0;
        row2.colRange(4, 7)  = arow2.colRange(3, 6) * 1.0;
        row2.colRange(8, 12) = arow2.colRange(6, 10) * 1.0;
        B.row(i) = row1 - row2;
    }

    // 构造以 z 为自变量的 11 阶多项式方程
    double c[11];
    cv::Mat coeffs(1, 11, CV_64F, c);
    c[10] = (b[0]*b[17]*b[34]+b[26]*b[4]*b[21]-b[26]*b[17]*b[8]-b[13]*b[4]*b[34]-b[0]*b[21]*b[30]+b[13]*b[30]*b[8]);
    c[9] = (b[26]*b[4]*b[22]+b[14]*b[30]*b[8]+b[13]*b[31]*b[8]+b[1]*b[17]*b[34]-b[13]*b[5]*b[34]+b[26]*b[5]*b[21]-b[0]*b[21]*b[31]-b[26]*b[17]*b[9]-b[1]*b[21]*b[30]+b[27]*b[4]*b[21]+b[0]*b[17]*b[35]-b[0]*b[22]*b[30]+b[13]*b[30]*b[9]+b[0]*b[18]*b[34]-b[27]*b[17]*b[8]-b[14]*b[4]*b[34]-b[13]*b[4]*b[35]-b[26]*b[18]*b[8]);
    c[8] = (b[14]*b[30]*b[9]+b[14]*b[31]*b[8]+b[13]*b[31]*b[9]-b[13]*b[4]*b[36]-b[13]*b[5]*b[35]+b[15]*b[30]*b[8]-b[13]*b[6]*b[34]+b[13]*b[30]*b[10]+b[13]*b[32]*b[8]-b[14]*b[4]*b[35]-b[14]*b[5]*b[34]+b[26]*b[4]*b[23]+b[26]*b[5]*b[22]+b[26]*b[6]*b[21]-b[26]*b[17]*b[10]-b[15]*b[4]*b[34]-b[26]*b[18]*b[9]-b[26]*b[19]*b[8]+b[27]*b[4]*b[22]+b[27]*b[5]*b[21]-b[27]*b[17]*b[9]-b[27]*b[18]*b[8]-b[1]*b[21]*b[31]-b[0]*b[23]*b[30]-b[0]*b[21]*b[32]+b[28]*b[4]*b[21]-b[28]*b[17]*b[8]+b[2]*b[17]*b[34]+b[0]*b[18]*b[35]-b[0]*b[22]*b[31]+b[0]*b[17]*b[36]+b[0]*b[19]*b[34]-b[1]*b[22]*b[30]+b[1]*b[18]*b[34]+b[1]*b[17]*b[35]-b[2]*b[21]*b[30]);
    c[7] = (b[14]*b[30]*b[10]+b[14]*b[32]*b[8]-b[3]*b[21]*b[30]+b[3]*b[17]*b[34]+b[13]*b[32]*b[9]+b[13]*b[33]*b[8]-b[13]*b[4]*b[37]-b[13]*b[5]*b[36]+b[15]*b[30]*b[9]+b[15]*b[31]*b[8]-b[16]*b[4]*b[34]-b[13]*b[6]*b[35]-b[13]*b[7]*b[34]+b[13]*b[30]*b[11]+b[13]*b[31]*b[10]+b[14]*b[31]*b[9]-b[14]*b[4]*b[36]-b[14]*b[5]*b[35]-b[14]*b[6]*b[34]+b[16]*b[30]*b[8]-b[26]*b[20]*b[8]+b[26]*b[4]*b[24]+b[26]*b[5]*b[23]+b[26]*b[6]*b[22]+b[26]*b[7]*b[21]-b[26]*b[17]*b[11]-b[15]*b[4]*b[35]-b[15]*b[5]*b[34]-b[26]*b[18]*b[10]-b[26]*b[19]*b[9]+b[27]*b[4]*b[23]+b[27]*b[5]*b[22]+b[27]*b[6]*b[21]-b[27]*b[17]*b[10]-b[27]*b[18]*b[9]-b[27]*b[19]*b[8]+b[0]*b[17]*b[37]-b[0]*b[23]*b[31]-b[0]*b[24]*b[30]-b[0]*b[21]*b[33]-b[29]*b[17]*b[8]+b[28]*b[4]*b[22]+b[28]*b[5]*b[21]-b[28]*b[17]*b[9]-b[28]*b[18]*b[8]+b[29]*b[4]*b[21]+b[1]*b[19]*b[34]-b[2]*b[21]*b[31]+b[0]*b[20]*b[34]+b[0]*b[19]*b[35]+b[0]*b[18]*b[36]-b[0]*b[22]*b[32]-b[1]*b[23]*b[30]-b[1]*b[21]*b[32]+b[1]*b[18]*b[35]-b[1]*b[22]*b[31]-b[2]*b[22]*b[30]+b[2]*b[17]*b[35]+b[1]*b[17]*b[36]+b[2]*b[18]*b[34]);
    c[6] = (-b[14]*b[6]*b[35]-b[14]*b[7]*b[34]-b[3]*b[22]*b[30]-b[3]*b[21]*b[31]+b[3]*b[17]*b[35]+b[3]*b[18]*b[34]+b[13]*b[32]*b[10]+b[13]*b[33]*b[9]-b[13]*b[4]*b[38]-b[13]*b[5]*b[37]-b[15]*b[6]*b[34]+b[15]*b[30]*b[10]+b[15]*b[32]*b[8]-b[16]*b[4]*b[35]-b[13]*b[6]*b[36]-b[13]*b[7]*b[35]+b[13]*b[31]*b[11]+b[13]*b[30]*b[12]+b[14]*b[32]*b[9]+b[14]*b[33]*b[8]-b[14]*b[4]*b[37]-b[14]*b[5]*b[36]+b[16]*b[30]*b[9]+b[16]*b[31]*b[8]-b[26]*b[20]*b[9]+b[26]*b[4]*b[25]+b[26]*b[5]*b[24]+b[26]*b[6]*b[23]+b[26]*b[7]*b[22]-b[26]*b[17]*b[12]+b[14]*b[30]*b[11]+b[14]*b[31]*b[10]+b[15]*b[31]*b[9]-b[15]*b[4]*b[36]-b[15]*b[5]*b[35]-b[26]*b[18]*b[11]-b[26]*b[19]*b[10]-b[27]*b[20]*b[8]+b[27]*b[4]*b[24]+b[27]*b[5]*b[23]+b[27]*b[6]*b[22]+b[27]*b[7]*b[21]-b[27]*b[17]*b[11]-b[27]*b[18]*b[10]-b[27]*b[19]*b[9]-b[16]*b[5]*b[34]-b[29]*b[17]*b[9]-b[29]*b[18]*b[8]+b[28]*b[4]*b[23]+b[28]*b[5]*b[22]+b[28]*b[6]*b[21]-b[28]*b[17]*b[10]-b[28]*b[18]*b[9]-b[28]*b[19]*b[8]+b[29]*b[4]*b[22]+b[29]*b[5]*b[21]-b[2]*b[23]*b[30]+b[2]*b[18]*b[35]-b[1]*b[22]*b[32]-b[2]*b[21]*b[32]+b[2]*b[19]*b[34]+b[0]*b[19]*b[36]-b[0]*b[22]*b[33]+b[0]*b[20]*b[35]-b[0]*b[23]*b[32]-b[0]*b[25]*b[30]+b[0]*b[17]*b[38]+b[0]*b[18]*b[37]-b[0]*b[24]*b[31]+b[1]*b[17]*b[37]-b[1]*b[23]*b[31]-b[1]*b[24]*b[30]-b[1]*b[21]*b[33]+b[1]*b[20]*b[34]+b[1]*b[19]*b[35]+b[1]*b[18]*b[36]+b[2]*b[17]*b[36]-b[2]*b[22]*b[31]);
    c[5] = (-b[14]*b[6]*b[36]-b[14]*b[7]*b[35]+b[14]*b[31]*b[11]-b[3]*b[23]*b[30]-b[3]*b[21]*b[32]+b[3]*b[18]*b[35]-b[3]*b[22]*b[31]+b[3]*b[17]*b[36]+b[3]*b[19]*b[34]+b[13]*b[32]*b[11]+b[13]*b[33]*b[10]-b[13]*b[5]*b[38]-b[15]*b[6]*b[35]-b[15]*b[7]*b[34]+b[15]*b[30]*b[11]+b[15]*b[31]*b[10]+b[16]*b[31]*b[9]-b[13]*b[6]*b[37]-b[13]*b[7]*b[36]+b[13]*b[31]*b[12]+b[14]*b[32]*b[10]+b[14]*b[33]*b[9]-b[14]*b[4]*b[38]-b[14]*b[5]*b[37]-b[16]*b[6]*b[34]+b[16]*b[30]*b[10]+b[16]*b[32]*b[8]-b[26]*b[20]*b[10]+b[26]*b[5]*b[25]+b[26]*b[6]*b[24]+b[26]*b[7]*b[23]+b[14]*b[30]*b[12]+b[15]*b[32]*b[9]+b[15]*b[33]*b[8]-b[15]*b[4]*b[37]-b[15]*b[5]*b[36]+b[29]*b[5]*b[22]+b[29]*b[6]*b[21]-b[26]*b[18]*b[12]-b[26]*b[19]*b[11]-b[27]*b[20]*b[9]+b[27]*b[4]*b[25]+b[27]*b[5]*b[24]+b[27]*b[6]*b[23]+b[27]*b[7]*b[22]-b[27]*b[17]*b[12]-b[27]*b[18]*b[11]-b[27]*b[19]*b[10]-b[28]*b[20]*b[8]-b[16]*b[4]*b[36]-b[16]*b[5]*b[35]-b[29]*b[17]*b[10]-b[29]*b[18]*b[9]-b[29]*b[19]*b[8]+b[28]*b[4]*b[24]+b[28]*b[5]*b[23]+b[28]*b[6]*b[22]+b[28]*b[7]*b[21]-b[28]*b[17]*b[11]-b[28]*b[18]*b[10]-b[28]*b[19]*b[9]+b[29]*b[4]*b[23]-b[2]*b[22]*b[32]-b[2]*b[21]*b[33]-b[1]*b[24]*b[31]+b[0]*b[18]*b[38]-b[0]*b[24]*b[32]+b[0]*b[19]*b[37]+b[0]*b[20]*b[36]-b[0]*b[25]*b[31]-b[0]*b[23]*b[33]+b[1]*b[19]*b[36]-b[1]*b[22]*b[33]+b[1]*b[20]*b[35]+b[2]*b[19]*b[35]-b[2]*b[24]*b[30]-b[2]*b[23]*b[31]+b[2]*b[20]*b[34]+b[2]*b[17]*b[37]-b[1]*b[25]*b[30]+b[1]*b[18]*b[37]+b[1]*b[17]*b[38]-b[1]*b[23]*b[32]+b[2]*b[18]*b[36]);
    c[4] = (-b[14]*b[6]*b[37]-b[14]*b[7]*b[36]+b[14]*b[31]*b[12]+b[3]*b[17]*b[37]-b[3]*b[23]*b[31]-b[3]*b[24]*b[30]-b[3]*b[21]*b[33]+b[3]*b[20]*b[34]+b[3]*b[19]*b[35]+b[3]*b[18]*b[36]-b[3]*b[22]*b[32]+b[13]*b[32]*b[12]+b[13]*b[33]*b[11]-b[15]*b[6]*b[36]-b[15]*b[7]*b[35]+b[15]*b[31]*b[11]+b[15]*b[30]*b[12]+b[16]*b[32]*b[9]+b[16]*b[33]*b[8]-b[13]*b[6]*b[38]-b[13]*b[7]*b[37]+b[14]*b[32]*b[11]+b[14]*b[33]*b[10]-b[14]*b[5]*b[38]-b[16]*b[6]*b[35]-b[16]*b[7]*b[34]+b[16]*b[30]*b[11]+b[16]*b[31]*b[10]-b[26]*b[19]*b[12]-b[26]*b[20]*b[11]+b[26]*b[6]*b[25]+b[26]*b[7]*b[24]+b[15]*b[32]*b[10]+b[15]*b[33]*b[9]-b[15]*b[4]*b[38]-b[15]*b[5]*b[37]+b[29]*b[5]*b[23]+b[29]*b[6]*b[22]+b[29]*b[7]*b[21]-b[27]*b[20]*b[10]+b[27]*b[5]*b[25]+b[27]*b[6]*b[24]+b[27]*b[7]*b[23]-b[27]*b[18]*b[12]-b[27]*b[19]*b[11]-b[28]*b[20]*b[9]-b[16]*b[4]*b[37]-b[16]*b[5]*b[36]+b[0]*b[19]*b[38]-b[0]*b[24]*b[33]+b[0]*b[20]*b[37]-b[29]*b[17]*b[11]-b[29]*b[18]*b[10]-b[29]*b[19]*b[9]+b[28]*b[4]*b[25]+b[28]*b[5]*b[24]+b[28]*b[6]*b[23]+b[28]*b[7]*b[22]-b[28]*b[17]*b[12]-b[28]*b[18]*b[11]-b[28]*b[19]*b[10]-b[29]*b[20]*b[8]+b[29]*b[4]*b[24]+b[2]*b[18]*b[37]-b[0]*b[25]*b[32]+b[1]*b[18]*b[38]-b[1]*b[24]*b[32]+b[1]*b[19]*b[37]+b[1]*b[20]*b[36]-b[1]*b[25]*b[31]+b[2]*b[17]*b[38]+b[2]*b[19]*b[36]-b[2]*b[24]*b[31]-b[2]*b[22]*b[33]-b[2]*b[23]*b[32]+b[2]*b[20]*b[35]-b[1]*b[23]*b[33]-b[2]*b[25]*b[30]);
    c[3] = (-b[14]*b[6]*b[38]-b[14]*b[7]*b[37]+b[3]*b[19]*b[36]-b[3]*b[22]*b[33]+b[3]*b[20]*b[35]-b[3]*b[23]*b[32]-b[3]*b[25]*b[30]+b[3]*b[17]*b[38]+b[3]*b[18]*b[37]-b[3]*b[24]*b[31]-b[15]*b[6]*b[37]-b[15]*b[7]*b[36]+b[15]*b[31]*b[12]+b[16]*b[32]*b[10]+b[16]*b[33]*b[9]+b[13]*b[33]*b[12]-b[13]*b[7]*b[38]+b[14]*b[32]*b[12]+b[14]*b[33]*b[11]-b[16]*b[6]*b[36]-b[16]*b[7]*b[35]+b[16]*b[31]*b[11]+b[16]*b[30]*b[12]+b[15]*b[32]*b[11]+b[15]*b[33]*b[10]-b[15]*b[5]*b[38]+b[29]*b[5]*b[24]+b[29]*b[6]*b[23]-b[26]*b[20]*b[12]+b[26]*b[7]*b[25]-b[27]*b[19]*b[12]-b[27]*b[20]*b[11]+b[27]*b[6]*b[25]+b[27]*b[7]*b[24]-b[28]*b[20]*b[10]-b[16]*b[4]*b[38]-b[16]*b[5]*b[37]+b[29]*b[7]*b[22]-b[29]*b[17]*b[12]-b[29]*b[18]*b[11]-b[29]*b[19]*b[10]+b[28]*b[5]*b[25]+b[28]*b[6]*b[24]+b[28]*b[7]*b[23]-b[28]*b[18]*b[12]-b[28]*b[19]*b[11]-b[29]*b[20]*b[9]+b[29]*b[4]*b[25]-b[2]*b[24]*b[32]+b[0]*b[20]*b[38]-b[0]*b[25]*b[33]+b[1]*b[19]*b[38]-b[1]*b[24]*b[33]+b[1]*b[20]*b[37]-b[2]*b[25]*b[31]+b[2]*b[20]*b[36]-b[1]*b[25]*b[32]+b[2]*b[19]*b[37]+b[2]*b[18]*b[38]-b[2]*b[23]*b[33]);
    c[2] = (b[3]*b[18]*b[38]-b[3]*b[24]*b[32]+b[3]*b[19]*b[37]+b[3]*b[20]*b[36]-b[3]*b[25]*b[31]-b[3]*b[23]*b[33]-b[15]*b[6]*b[38]-b[15]*b[7]*b[37]+b[16]*b[32]*b[11]+b[16]*b[33]*b[10]-b[16]*b[5]*b[38]-b[16]*b[6]*b[37]-b[16]*b[7]*b[36]+b[16]*b[31]*b[12]+b[14]*b[33]*b[12]-b[14]*b[7]*b[38]+b[15]*b[32]*b[12]+b[15]*b[33]*b[11]+b[29]*b[5]*b[25]+b[29]*b[6]*b[24]-b[27]*b[20]*b[12]+b[27]*b[7]*b[25]-b[28]*b[19]*b[12]-b[28]*b[20]*b[11]+b[29]*b[7]*b[23]-b[29]*b[18]*b[12]-b[29]*b[19]*b[11]+b[28]*b[6]*b[25]+b[28]*b[7]*b[24]-b[29]*b[20]*b[10]+b[2]*b[19]*b[38]-b[1]*b[25]*b[33]+b[2]*b[20]*b[37]-b[2]*b[24]*b[33]-b[2]*b[25]*b[32]+b[1]*b[20]*b[38]);
    c[1] = (b[29]*b[7]*b[24]-b[29]*b[20]*b[11]+b[2]*b[20]*b[38]-b[2]*b[25]*b[33]-b[28]*b[20]*b[12]+b[28]*b[7]*b[25]-b[29]*b[19]*b[12]-b[3]*b[24]*b[33]+b[15]*b[33]*b[12]+b[3]*b[19]*b[38]-b[16]*b[6]*b[38]+b[3]*b[20]*b[37]+b[16]*b[32]*b[12]+b[29]*b[6]*b[25]-b[16]*b[7]*b[37]-b[3]*b[25]*b[32]-b[15]*b[7]*b[38]+b[16]*b[33]*b[11]);
    c[0] = -b[29]*b[20]*b[12]+b[29]*b[7]*b[25]+b[16]*b[33]*b[12]-b[16]*b[7]*b[38]+b[3]*b[20]*b[38]-b[3]*b[25]*b[33];

    // 求解 11 阶多项式方程的根 roots （复数根）
    std::vector<cv::Complex<double>> roots;
    cv::solvePoly(coeffs, roots);

    // 因为 11 阶多项式方程的根有很多，所以对应的每一个结果都需要代入，计算出一组 x，y，z参数
    std::vector<double> xs, ys, zs;

    // 将解出来的所有的根 roots 都代入进行检查，选择出最合适的结果
    for (unsigned long i = 0; i < roots.size(); i++) {
        // 如果当前遍历到的根的虚部比较大，则不认为有效
        if (fabs(roots[i].im) > 1e-10) continue;

        // 取根的实部作为 z 参数，并计算其二次方、三次方和四次方
        double z1 = roots[i].re;
        double z2 = z1 * z1;
        double z3 = z2 * z1;
        double z4 = z3 * z1;

        // 将 z 的结果代入到向量为 [xy, x, y, 1] 的线性方程组中，但只构造向量 [x, y, 1] 对应的线性方程组的参数矩阵
        double bz[3][3];
        for (int j = 0; j < 3; j++) {
            const double * br = b + j * 13;
            bz[j][0] = br[0] * z3 + br[1] * z2 + br[2] * z1 + br[3];
            bz[j][1] = br[4] * z3 + br[5] * z2 + br[6] * z1 + br[7];
            bz[j][2] = br[8] * z4 + br[9] * z3 + br[10] * z2 + br[11] * z1 + br[12];
        }

        // 求解此方程组，得到向量 [x, y, 1]
        cv::Mat Bz(3, 3, CV_64F, bz);
        cv::Mat xy1;
        cv::SVD::solveZ(Bz, xy1);

        // 如果求解出来的应该是 1 的元素（下面作为分母进行归一化）接近于 0，则不认为有效
        if (fabs(xy1.at<double>(2)) < 1e-10) continue;
        xs.emplace_back(xy1.at<double>(0) / xy1.at<double>(2));
        ys.emplace_back(xy1.at<double>(1) / xy1.at<double>(2));
        zs.emplace_back(z1);

        // 利用求解出来的参数 x，y，z，w，在解空间中计算出 E 矩阵
        cv::Mat Evec = EE.col(0) * xs.back() + EE.col(1) * ys.back() + EE.col(2) * zs.back() + EE.col(3);
        Evec /= cv::norm(Evec);

        // 将结果拷贝到 ematrix 中，这里用指针 e 指向了 ematrix
        Eigen::Matrix3d tempE;
        tempE << Evec.at<double>(0, 0), Evec.at<double>(3, 0), Evec.at<double>(6, 0),
                 Evec.at<double>(1, 0), Evec.at<double>(4, 0), Evec.at<double>(7, 0),
                 Evec.at<double>(2, 0), Evec.at<double>(5, 0), Evec.at<double>(8, 0);
        allE.emplace_back(tempE);
    }

    // 对每一个本质矩阵，都计算其平均模型误差
    meanErrors.clear();
    for (auto &E : allE) {
        std::vector<double> errors = this->ComputeEpipolarGeometryError(normPoints0, normPoints1, E);
        double error = 0.0;
        for (unsigned int i = 0; i < errors.size(); i++) {
            error += errors[i];
        }
        error /= errors.size();
        meanErrors.emplace_back(error);
    }

    // 返回所有的本质矩阵
    return allE;
}



/* 计算对极约束误差 */
std::vector<double> EpipolarGeometryClass::ComputeEpipolarGeometryError(std::vector<Eigen::Vector3d> &normPoints0,
                                                                        std::vector<Eigen::Vector3d> &normPoints1,
                                                                        Eigen::Matrix3d &E) {
    // 构造计算所得误差结果向量，计算每一对匹配点对在本质矩阵 E 下的误差
    std::vector<double> errors(normPoints0.size());

    // 遍历所有的特征点对，计算出对应的误差 error
    for (unsigned long i = 0; i < normPoints0.size(); i++) {
        // 建立引用
        double &error = errors[i];
        Eigen::Vector3d &point0 = normPoints0[i];
        Eigen::Vector3d &point1 = normPoints1[i];

        // 计算对极约束的误差
        Eigen::Matrix<double, 3, 1> E_p0 = E * point0;
        Eigen::Matrix<double, 1, 3> p1T_E = point1.transpose() * E;
        double p1T_E_p0 = p1T_E * point0;
        error = (p1T_E_p0 * p1T_E_p0) / (E_p0(0, 0) * E_p0(0, 0) + E_p0(1, 0) * E_p0(1, 0) + p1T_E(0, 0) * p1T_E(0, 0) + p1T_E(0, 1) * p1T_E(0, 1));
    }

    // 返回结果
    return errors;
}


/* 像素坐标变换为归一化坐标 */
std::vector<Eigen::Vector3d> EpipolarGeometryClass::GetNormPointsFromPixelPoints(std::vector<cv::Point2f> &pixelPoints,
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


/* 五点法计算系数多项式 */
void EpipolarGeometryClass::GetCoeffMatrixFivePoints(double *e, double *A) {
    double ep2[36], ep3[36];
    for (int i = 0; i < 36; i++) {
        ep2[i] = e[i] * e[i];
        ep3[i] = ep2[i] * e[i];
    }

    A[0]=e[33]*e[28]*e[32]-e[33]*e[31]*e[29]+e[30]*e[34]*e[29]-e[30]*e[28]*e[35]-e[27]*e[32]*e[34]+e[27]*e[31]*e[35];
    A[146]=.5000000000*e[6]*ep2[8]-.5000000000*e[6]*ep2[5]+.5000000000*ep3[6]+.5000000000*e[6]*ep2[7]-.5000000000*e[6]*ep2[4]+e[0]*e[2]*e[8]+e[3]*e[4]*e[7]+e[3]*e[5]*e[8]+e[0]*e[1]*e[7]-.5000000000*e[6]*ep2[1]-.5000000000*e[6]*ep2[2]+.5000000000*ep2[0]*e[6]+.5000000000*ep2[3]*e[6];
    A[1]=e[30]*e[34]*e[2]+e[33]*e[1]*e[32]-e[3]*e[28]*e[35]+e[0]*e[31]*e[35]+e[3]*e[34]*e[29]-e[30]*e[1]*e[35]+e[27]*e[31]*e[8]-e[27]*e[32]*e[7]-e[30]*e[28]*e[8]-e[33]*e[31]*e[2]-e[0]*e[32]*e[34]+e[6]*e[28]*e[32]-e[33]*e[4]*e[29]+e[33]*e[28]*e[5]+e[30]*e[7]*e[29]+e[27]*e[4]*e[35]-e[27]*e[5]*e[34]-e[6]*e[31]*e[29];
    A[147]=e[9]*e[27]*e[15]+e[9]*e[29]*e[17]+e[9]*e[11]*e[35]+e[9]*e[28]*e[16]+e[9]*e[10]*e[34]+e[27]*e[11]*e[17]+e[27]*e[10]*e[16]+e[12]*e[30]*e[15]+e[12]*e[32]*e[17]+e[12]*e[14]*e[35]+e[12]*e[31]*e[16]+e[12]*e[13]*e[34]+e[30]*e[14]*e[17]+e[30]*e[13]*e[16]+e[15]*e[35]*e[17]+e[15]*e[34]*e[16]-1.*e[15]*e[28]*e[10]-1.*e[15]*e[31]*e[13]-1.*e[15]*e[32]*e[14]-1.*e[15]*e[29]*e[11]+.5000000000*ep2[9]*e[33]+.5000000000*e[33]*ep2[16]-.5000000000*e[33]*ep2[11]+.5000000000*e[33]*ep2[12]+1.500000000*e[33]*ep2[15]+.5000000000*e[33]*ep2[17]-.5000000000*e[33]*ep2[10]-.5000000000*e[33]*ep2[14]-.5000000000*e[33]*ep2[13];
    A[2]=-e[33]*e[22]*e[29]-e[33]*e[31]*e[20]-e[27]*e[32]*e[25]+e[27]*e[22]*e[35]-e[27]*e[23]*e[34]+e[27]*e[31]*e[26]+e[33]*e[28]*e[23]-e[21]*e[28]*e[35]+e[30]*e[25]*e[29]+e[24]*e[28]*e[32]-e[24]*e[31]*e[29]+e[18]*e[31]*e[35]-e[30]*e[28]*e[26]-e[30]*e[19]*e[35]+e[21]*e[34]*e[29]+e[33]*e[19]*e[32]-e[18]*e[32]*e[34]+e[30]*e[34]*e[20];
    A[144]=e[18]*e[2]*e[17]+e[3]*e[21]*e[15]+e[3]*e[12]*e[24]+e[3]*e[23]*e[17]+e[3]*e[14]*e[26]+e[3]*e[22]*e[16]+e[3]*e[13]*e[25]+3.*e[6]*e[24]*e[15]+e[6]*e[26]*e[17]+e[6]*e[25]*e[16]+e[0]*e[20]*e[17]+e[0]*e[11]*e[26]+e[0]*e[19]*e[16]+e[0]*e[10]*e[25]+e[15]*e[26]*e[8]-1.*e[15]*e[20]*e[2]-1.*e[15]*e[19]*e[1]-1.*e[15]*e[22]*e[4]+e[15]*e[25]*e[7]-1.*e[15]*e[23]*e[5]+e[12]*e[21]*e[6]+e[12]*e[22]*e[7]+e[12]*e[4]*e[25]+e[12]*e[23]*e[8]+e[12]*e[5]*e[26]-1.*e[24]*e[11]*e[2]-1.*e[24]*e[10]*e[1]-1.*e[24]*e[13]*e[4]+e[24]*e[16]*e[7]-1.*e[24]*e[14]*e[5]+e[24]*e[17]*e[8]+e[21]*e[13]*e[7]+e[21]*e[4]*e[16]+e[21]*e[14]*e[8]+e[21]*e[5]*e[17]-1.*e[6]*e[23]*e[14]-1.*e[6]*e[20]*e[11]-1.*e[6]*e[19]*e[10]-1.*e[6]*e[22]*e[13]+e[9]*e[18]*e[6]+e[9]*e[0]*e[24]+e[9]*e[19]*e[7]+e[9]*e[1]*e[25]+e[9]*e[20]*e[8]+e[9]*e[2]*e[26]+e[18]*e[0]*e[15]+e[18]*e[10]*e[7]+e[18]*e[1]*e[16]+e[18]*e[11]*e[8];
    A[3]=e[33]*e[10]*e[32]+e[33]*e[28]*e[14]-e[33]*e[13]*e[29]-e[33]*e[31]*e[11]+e[9]*e[31]*e[35]-e[9]*e[32]*e[34]+e[27]*e[13]*e[35]-e[27]*e[32]*e[16]+e[27]*e[31]*e[17]-e[27]*e[14]*e[34]+e[12]*e[34]*e[29]-e[12]*e[28]*e[35]+e[30]*e[34]*e[11]+e[30]*e[16]*e[29]-e[30]*e[10]*e[35]-e[30]*e[28]*e[17]+e[15]*e[28]*e[32]-e[15]*e[31]*e[29];
    A[145]=e[0]*e[27]*e[6]+e[0]*e[28]*e[7]+e[0]*e[1]*e[34]+e[0]*e[29]*e[8]+e[0]*e[2]*e[35]+e[6]*e[34]*e[7]-1.*e[6]*e[32]*e[5]+e[6]*e[30]*e[3]+e[6]*e[35]*e[8]-1.*e[6]*e[29]*e[2]-1.*e[6]*e[28]*e[1]-1.*e[6]*e[31]*e[4]+e[27]*e[1]*e[7]+e[27]*e[2]*e[8]+e[3]*e[31]*e[7]+e[3]*e[4]*e[34]+e[3]*e[32]*e[8]+e[3]*e[5]*e[35]+e[30]*e[4]*e[7]+e[30]*e[5]*e[8]+.5000000000*ep2[0]*e[33]+1.500000000*e[33]*ep2[6]-.5000000000*e[33]*ep2[4]-.5000000000*e[33]*ep2[5]-.5000000000*e[33]*ep2[1]+.5000000000*e[33]*ep2[7]+.5000000000*e[33]*ep2[3]-.5000000000*e[33]*ep2[2]+.5000000000*e[33]*ep2[8];
    A[4]=-e[0]*e[23]*e[16]+e[9]*e[4]*e[26]+e[9]*e[22]*e[8]-e[9]*e[5]*e[25]-e[9]*e[23]*e[7]+e[18]*e[4]*e[17]+e[18]*e[13]*e[8]-e[18]*e[5]*e[16]-e[18]*e[14]*e[7]+e[3]*e[16]*e[20]+e[3]*e[25]*e[11]-e[3]*e[10]*e[26]-e[3]*e[19]*e[17]+e[12]*e[7]*e[20]+e[12]*e[25]*e[2]-e[12]*e[1]*e[26]-e[12]*e[19]*e[8]+e[21]*e[7]*e[11]+e[21]*e[16]*e[2]-e[21]*e[1]*e[17]-e[21]*e[10]*e[8]+e[6]*e[10]*e[23]+e[6]*e[19]*e[14]-e[6]*e[13]*e[20]-e[6]*e[22]*e[11]+e[15]*e[1]*e[23]+e[15]*e[19]*e[5]-e[15]*e[4]*e[20]-e[15]*e[22]*e[2]+e[24]*e[1]*e[14]+e[24]*e[10]*e[5]-e[24]*e[4]*e[11]-e[24]*e[13]*e[2]+e[0]*e[13]*e[26]+e[0]*e[22]*e[17]-e[0]*e[14]*e[25];
    A[150]=e[18]*e[19]*e[25]+.5000000000*ep3[24]-.5000000000*e[24]*ep2[23]+e[18]*e[20]*e[26]+e[21]*e[22]*e[25]+e[21]*e[23]*e[26]-.5000000000*e[24]*ep2[19]+.5000000000*ep2[21]*e[24]+.5000000000*e[24]*ep2[26]-.5000000000*e[24]*ep2[20]+.5000000000*ep2[18]*e[24]-.5000000000*e[24]*ep2[22]+.5000000000*e[24]*ep2[25];
    A[5]=-e[3]*e[1]*e[35]-e[0]*e[32]*e[7]+e[27]*e[4]*e[8]+e[33]*e[1]*e[5]-e[33]*e[4]*e[2]+e[0]*e[4]*e[35]+e[3]*e[34]*e[2]-e[30]*e[1]*e[8]+e[30]*e[7]*e[2]-e[6]*e[4]*e[29]+e[3]*e[7]*e[29]+e[6]*e[1]*e[32]-e[0]*e[5]*e[34]-e[3]*e[28]*e[8]+e[0]*e[31]*e[8]+e[6]*e[28]*e[5]-e[6]*e[31]*e[2]-e[27]*e[5]*e[7];
    A[151]=e[33]*e[16]*e[7]-1.*e[33]*e[14]*e[5]+e[33]*e[17]*e[8]+e[30]*e[13]*e[7]+e[30]*e[4]*e[16]+e[30]*e[14]*e[8]+e[30]*e[5]*e[17]+e[6]*e[27]*e[9]-1.*e[6]*e[28]*e[10]-1.*e[6]*e[31]*e[13]-1.*e[6]*e[32]*e[14]-1.*e[6]*e[29]*e[11]+e[9]*e[28]*e[7]+e[9]*e[1]*e[34]+e[9]*e[29]*e[8]+e[9]*e[2]*e[35]+e[27]*e[10]*e[7]+e[27]*e[1]*e[16]+e[27]*e[11]*e[8]+e[27]*e[2]*e[17]+e[3]*e[30]*e[15]+e[3]*e[12]*e[33]+e[3]*e[32]*e[17]+e[3]*e[14]*e[35]+e[3]*e[31]*e[16]+e[3]*e[13]*e[34]+3.*e[6]*e[33]*e[15]+e[6]*e[35]*e[17]+e[6]*e[34]*e[16]+e[0]*e[27]*e[15]+e[0]*e[9]*e[33]+e[0]*e[29]*e[17]+e[0]*e[11]*e[35]+e[0]*e[28]*e[16]+e[0]*e[10]*e[34]+e[15]*e[34]*e[7]-1.*e[15]*e[32]*e[5]+e[15]*e[35]*e[8]-1.*e[15]*e[29]*e[2]-1.*e[15]*e[28]*e[1]-1.*e[15]*e[31]*e[4]+e[12]*e[30]*e[6]+e[12]*e[31]*e[7]+e[12]*e[4]*e[34]+e[12]*e[32]*e[8]+e[12]*e[5]*e[35]-1.*e[33]*e[11]*e[2]-1.*e[33]*e[10]*e[1]-1.*e[33]*e[13]*e[4];
    A[6]=e[6]*e[1]*e[5]-e[6]*e[4]*e[2]+e[3]*e[7]*e[2]+e[0]*e[4]*e[8]-e[0]*e[5]*e[7]-e[3]*e[1]*e[8];
    A[148]=.5000000000*ep3[15]+e[9]*e[10]*e[16]-.5000000000*e[15]*ep2[11]+e[9]*e[11]*e[17]+.5000000000*ep2[12]*e[15]+.5000000000*e[15]*ep2[16]+.5000000000*e[15]*ep2[17]-.5000000000*e[15]*ep2[13]+.5000000000*ep2[9]*e[15]+e[12]*e[14]*e[17]-.5000000000*e[15]*ep2[10]-.5000000000*e[15]*ep2[14]+e[12]*e[13]*e[16];
    A[7]=e[15]*e[28]*e[14]-e[15]*e[13]*e[29]-e[15]*e[31]*e[11]+e[33]*e[10]*e[14]-e[33]*e[13]*e[11]+e[9]*e[13]*e[35]-e[9]*e[32]*e[16]+e[9]*e[31]*e[17]-e[9]*e[14]*e[34]+e[27]*e[13]*e[17]-e[27]*e[14]*e[16]+e[12]*e[34]*e[11]+e[12]*e[16]*e[29]-e[12]*e[10]*e[35]-e[12]*e[28]*e[17]+e[30]*e[16]*e[11]-e[30]*e[10]*e[17]+e[15]*e[10]*e[32];
    A[149]=e[18]*e[27]*e[24]+e[18]*e[28]*e[25]+e[18]*e[19]*e[34]+e[18]*e[29]*e[26]+e[18]*e[20]*e[35]+e[27]*e[19]*e[25]+e[27]*e[20]*e[26]+e[21]*e[30]*e[24]+e[21]*e[31]*e[25]+e[21]*e[22]*e[34]+e[21]*e[32]*e[26]+e[21]*e[23]*e[35]+e[30]*e[22]*e[25]+e[30]*e[23]*e[26]+e[24]*e[34]*e[25]+e[24]*e[35]*e[26]-1.*e[24]*e[29]*e[20]-1.*e[24]*e[31]*e[22]-1.*e[24]*e[32]*e[23]-1.*e[24]*e[28]*e[19]+1.500000000*e[33]*ep2[24]+.5000000000*e[33]*ep2[25]+.5000000000*e[33]*ep2[26]-.5000000000*e[33]*ep2[23]-.5000000000*e[33]*ep2[19]-.5000000000*e[33]*ep2[20]-.5000000000*e[33]*ep2[22]+.5000000000*ep2[18]*e[33]+.5000000000*ep2[21]*e[33];
    A[9]=e[21]*e[25]*e[29]-e[27]*e[23]*e[25]+e[24]*e[19]*e[32]-e[21]*e[28]*e[26]-e[21]*e[19]*e[35]+e[18]*e[31]*e[26]-e[30]*e[19]*e[26]-e[24]*e[31]*e[20]+e[24]*e[28]*e[23]+e[27]*e[22]*e[26]+e[30]*e[25]*e[20]-e[33]*e[22]*e[20]+e[33]*e[19]*e[23]+e[21]*e[34]*e[20]-e[18]*e[23]*e[34]-e[24]*e[22]*e[29]-e[18]*e[32]*e[25]+e[18]*e[22]*e[35];
    A[155]=e[12]*e[14]*e[8]+e[12]*e[5]*e[17]+e[15]*e[16]*e[7]+e[15]*e[17]*e[8]+e[0]*e[11]*e[17]+e[0]*e[9]*e[15]+e[0]*e[10]*e[16]+e[3]*e[14]*e[17]+e[3]*e[13]*e[16]+e[9]*e[10]*e[7]+e[9]*e[1]*e[16]+e[9]*e[11]*e[8]+e[9]*e[2]*e[17]-1.*e[15]*e[11]*e[2]-1.*e[15]*e[10]*e[1]-1.*e[15]*e[13]*e[4]-1.*e[15]*e[14]*e[5]+e[12]*e[3]*e[15]+e[12]*e[13]*e[7]+e[12]*e[4]*e[16]+.5000000000*ep2[12]*e[6]+1.500000000*ep2[15]*e[6]+.5000000000*e[6]*ep2[17]+.5000000000*e[6]*ep2[16]+.5000000000*e[6]*ep2[9]-.5000000000*e[6]*ep2[11]-.5000000000*e[6]*ep2[10]-.5000000000*e[6]*ep2[14]-.5000000000*e[6]*ep2[13];
    A[8]=-e[9]*e[14]*e[16]-e[12]*e[10]*e[17]+e[9]*e[13]*e[17]-e[15]*e[13]*e[11]+e[15]*e[10]*e[14]+e[12]*e[16]*e[11];
    A[154]=e[21]*e[14]*e[17]+e[21]*e[13]*e[16]+e[15]*e[26]*e[17]+e[15]*e[25]*e[16]-1.*e[15]*e[23]*e[14]-1.*e[15]*e[20]*e[11]-1.*e[15]*e[19]*e[10]-1.*e[15]*e[22]*e[13]+e[9]*e[20]*e[17]+e[9]*e[11]*e[26]+e[9]*e[19]*e[16]+e[9]*e[10]*e[25]+.5000000000*ep2[12]*e[24]+1.500000000*e[24]*ep2[15]+.5000000000*e[24]*ep2[17]+.5000000000*e[24]*ep2[16]+.5000000000*ep2[9]*e[24]-.5000000000*e[24]*ep2[11]-.5000000000*e[24]*ep2[10]-.5000000000*e[24]*ep2[14]-.5000000000*e[24]*ep2[13]+e[18]*e[11]*e[17]+e[18]*e[9]*e[15]+e[18]*e[10]*e[16]+e[12]*e[21]*e[15]+e[12]*e[23]*e[17]+e[12]*e[14]*e[26]+e[12]*e[22]*e[16]+e[12]*e[13]*e[25];
    A[11]=-e[9]*e[5]*e[34]+e[9]*e[31]*e[8]-e[9]*e[32]*e[7]+e[27]*e[4]*e[17]+e[27]*e[13]*e[8]-e[27]*e[5]*e[16]-e[27]*e[14]*e[7]+e[0]*e[13]*e[35]-e[0]*e[32]*e[16]+e[0]*e[31]*e[17]-e[0]*e[14]*e[34]+e[9]*e[4]*e[35]+e[6]*e[10]*e[32]+e[6]*e[28]*e[14]-e[6]*e[13]*e[29]-e[6]*e[31]*e[11]+e[15]*e[1]*e[32]+e[3]*e[34]*e[11]+e[3]*e[16]*e[29]-e[3]*e[10]*e[35]-e[3]*e[28]*e[17]-e[12]*e[1]*e[35]+e[12]*e[7]*e[29]+e[12]*e[34]*e[2]-e[12]*e[28]*e[8]+e[15]*e[28]*e[5]-e[15]*e[4]*e[29]-e[15]*e[31]*e[2]+e[33]*e[1]*e[14]+e[33]*e[10]*e[5]-e[33]*e[4]*e[11]-e[33]*e[13]*e[2]+e[30]*e[7]*e[11]+e[30]*e[16]*e[2]-e[30]*e[1]*e[17]-e[30]*e[10]*e[8];
    A[153]=e[21]*e[31]*e[7]+e[21]*e[4]*e[34]+e[21]*e[32]*e[8]+e[21]*e[5]*e[35]+e[30]*e[22]*e[7]+e[30]*e[4]*e[25]+e[30]*e[23]*e[8]+e[30]*e[5]*e[26]+3.*e[24]*e[33]*e[6]+e[24]*e[34]*e[7]+e[24]*e[35]*e[8]+e[33]*e[25]*e[7]+e[33]*e[26]*e[8]+e[0]*e[27]*e[24]+e[0]*e[18]*e[33]+e[0]*e[28]*e[25]+e[0]*e[19]*e[34]+e[0]*e[29]*e[26]+e[0]*e[20]*e[35]+e[18]*e[27]*e[6]+e[18]*e[28]*e[7]+e[18]*e[1]*e[34]+e[18]*e[29]*e[8]+e[18]*e[2]*e[35]+e[27]*e[19]*e[7]+e[27]*e[1]*e[25]+e[27]*e[20]*e[8]+e[27]*e[2]*e[26]+e[3]*e[30]*e[24]+e[3]*e[21]*e[33]+e[3]*e[31]*e[25]+e[3]*e[22]*e[34]+e[3]*e[32]*e[26]+e[3]*e[23]*e[35]+e[6]*e[30]*e[21]-1.*e[6]*e[29]*e[20]+e[6]*e[35]*e[26]-1.*e[6]*e[31]*e[22]-1.*e[6]*e[32]*e[23]-1.*e[6]*e[28]*e[19]+e[6]*e[34]*e[25]-1.*e[24]*e[32]*e[5]-1.*e[24]*e[29]*e[2]-1.*e[24]*e[28]*e[1]-1.*e[24]*e[31]*e[4]-1.*e[33]*e[20]*e[2]-1.*e[33]*e[19]*e[1]-1.*e[33]*e[22]*e[4]-1.*e[33]*e[23]*e[5];
    A[10]=e[21]*e[25]*e[20]-e[21]*e[19]*e[26]+e[18]*e[22]*e[26]-e[18]*e[23]*e[25]-e[24]*e[22]*e[20]+e[24]*e[19]*e[23];
    A[152]=e[3]*e[4]*e[25]+e[3]*e[23]*e[8]+e[3]*e[5]*e[26]+e[21]*e[4]*e[7]+e[21]*e[5]*e[8]+e[6]*e[25]*e[7]+e[6]*e[26]*e[8]+e[0]*e[19]*e[7]+e[0]*e[1]*e[25]+e[0]*e[20]*e[8]+e[0]*e[2]*e[26]-1.*e[6]*e[20]*e[2]-1.*e[6]*e[19]*e[1]-1.*e[6]*e[22]*e[4]-1.*e[6]*e[23]*e[5]+e[18]*e[1]*e[7]+e[18]*e[0]*e[6]+e[18]*e[2]*e[8]+e[3]*e[21]*e[6]+e[3]*e[22]*e[7]-.5000000000*e[24]*ep2[4]+.5000000000*e[24]*ep2[0]+1.500000000*e[24]*ep2[6]-.5000000000*e[24]*ep2[5]-.5000000000*e[24]*ep2[1]+.5000000000*e[24]*ep2[7]+.5000000000*e[24]*ep2[3]-.5000000000*e[24]*ep2[2]+.5000000000*e[24]*ep2[8];
    A[13]=e[6]*e[28]*e[23]-e[6]*e[22]*e[29]-e[6]*e[31]*e[20]-e[3]*e[19]*e[35]+e[3]*e[34]*e[20]+e[3]*e[25]*e[29]-e[21]*e[1]*e[35]+e[21]*e[7]*e[29]+e[21]*e[34]*e[2]+e[24]*e[1]*e[32]+e[24]*e[28]*e[5]-e[24]*e[4]*e[29]-e[24]*e[31]*e[2]+e[33]*e[1]*e[23]+e[33]*e[19]*e[5]-e[33]*e[4]*e[20]-e[33]*e[22]*e[2]-e[21]*e[28]*e[8]+e[30]*e[7]*e[20]+e[30]*e[25]*e[2]-e[30]*e[1]*e[26]+e[18]*e[4]*e[35]-e[18]*e[5]*e[34]+e[18]*e[31]*e[8]-e[18]*e[32]*e[7]+e[27]*e[4]*e[26]+e[27]*e[22]*e[8]-e[27]*e[5]*e[25]-e[27]*e[23]*e[7]-e[3]*e[28]*e[26]-e[0]*e[32]*e[25]+e[0]*e[22]*e[35]-e[0]*e[23]*e[34]+e[0]*e[31]*e[26]-e[30]*e[19]*e[8]+e[6]*e[19]*e[32];
    A[159]=.5000000000*ep2[18]*e[6]+.5000000000*ep2[21]*e[6]+1.500000000*ep2[24]*e[6]+.5000000000*e[6]*ep2[26]-.5000000000*e[6]*ep2[23]-.5000000000*e[6]*ep2[19]-.5000000000*e[6]*ep2[20]-.5000000000*e[6]*ep2[22]+.5000000000*e[6]*ep2[25]+e[21]*e[3]*e[24]+e[18]*e[20]*e[8]+e[21]*e[4]*e[25]+e[18]*e[19]*e[7]+e[18]*e[1]*e[25]+e[21]*e[22]*e[7]+e[21]*e[23]*e[8]+e[18]*e[0]*e[24]+e[18]*e[2]*e[26]+e[21]*e[5]*e[26]+e[24]*e[26]*e[8]-1.*e[24]*e[20]*e[2]-1.*e[24]*e[19]*e[1]-1.*e[24]*e[22]*e[4]+e[24]*e[25]*e[7]-1.*e[24]*e[23]*e[5]+e[0]*e[19]*e[25]+e[0]*e[20]*e[26]+e[3]*e[22]*e[25]+e[3]*e[23]*e[26];
    A[12]=e[18]*e[4]*e[8]+e[3]*e[7]*e[20]+e[3]*e[25]*e[2]-e[3]*e[1]*e[26]-e[18]*e[5]*e[7]+e[6]*e[1]*e[23]+e[6]*e[19]*e[5]-e[6]*e[4]*e[20]-e[6]*e[22]*e[2]+e[21]*e[7]*e[2]-e[21]*e[1]*e[8]+e[24]*e[1]*e[5]-e[24]*e[4]*e[2]-e[3]*e[19]*e[8]+e[0]*e[4]*e[26]+e[0]*e[22]*e[8]-e[0]*e[5]*e[25]-e[0]*e[23]*e[7];
    A[158]=e[9]*e[1]*e[7]+e[9]*e[0]*e[6]+e[9]*e[2]*e[8]+e[3]*e[12]*e[6]+e[3]*e[13]*e[7]+e[3]*e[4]*e[16]+e[3]*e[14]*e[8]+e[3]*e[5]*e[17]+e[12]*e[4]*e[7]+e[12]*e[5]*e[8]+e[6]*e[16]*e[7]+e[6]*e[17]*e[8]-1.*e[6]*e[11]*e[2]-1.*e[6]*e[10]*e[1]-1.*e[6]*e[13]*e[4]-1.*e[6]*e[14]*e[5]+e[0]*e[10]*e[7]+e[0]*e[1]*e[16]+e[0]*e[11]*e[8]+e[0]*e[2]*e[17]+.5000000000*ep2[3]*e[15]+1.500000000*e[15]*ep2[6]+.5000000000*e[15]*ep2[7]+.5000000000*e[15]*ep2[8]+.5000000000*ep2[0]*e[15]-.5000000000*e[15]*ep2[4]-.5000000000*e[15]*ep2[5]-.5000000000*e[15]*ep2[1]-.5000000000*e[15]*ep2[2];
    A[15]=-e[15]*e[13]*e[2]-e[6]*e[13]*e[11]-e[15]*e[4]*e[11]+e[12]*e[16]*e[2]-e[3]*e[10]*e[17]+e[3]*e[16]*e[11]+e[0]*e[13]*e[17]-e[0]*e[14]*e[16]+e[15]*e[1]*e[14]-e[12]*e[10]*e[8]+e[9]*e[4]*e[17]+e[9]*e[13]*e[8]-e[9]*e[5]*e[16]-e[9]*e[14]*e[7]+e[15]*e[10]*e[5]+e[12]*e[7]*e[11]+e[6]*e[10]*e[14]-e[12]*e[1]*e[17];
    A[157]=e[12]*e[30]*e[24]+e[12]*e[21]*e[33]+e[12]*e[31]*e[25]+e[12]*e[22]*e[34]+e[12]*e[32]*e[26]+e[12]*e[23]*e[35]+e[9]*e[27]*e[24]+e[9]*e[18]*e[33]+e[9]*e[28]*e[25]+e[9]*e[19]*e[34]+e[9]*e[29]*e[26]+e[9]*e[20]*e[35]+e[21]*e[30]*e[15]+e[21]*e[32]*e[17]+e[21]*e[14]*e[35]+e[21]*e[31]*e[16]+e[21]*e[13]*e[34]+e[30]*e[23]*e[17]+e[30]*e[14]*e[26]+e[30]*e[22]*e[16]+e[30]*e[13]*e[25]+e[15]*e[27]*e[18]+3.*e[15]*e[33]*e[24]-1.*e[15]*e[29]*e[20]+e[15]*e[35]*e[26]-1.*e[15]*e[31]*e[22]-1.*e[15]*e[32]*e[23]-1.*e[15]*e[28]*e[19]+e[15]*e[34]*e[25]+e[18]*e[29]*e[17]+e[18]*e[11]*e[35]+e[18]*e[28]*e[16]+e[18]*e[10]*e[34]+e[27]*e[20]*e[17]+e[27]*e[11]*e[26]+e[27]*e[19]*e[16]+e[27]*e[10]*e[25]-1.*e[24]*e[28]*e[10]-1.*e[24]*e[31]*e[13]-1.*e[24]*e[32]*e[14]+e[24]*e[34]*e[16]+e[24]*e[35]*e[17]-1.*e[24]*e[29]*e[11]-1.*e[33]*e[23]*e[14]+e[33]*e[25]*e[16]+e[33]*e[26]*e[17]-1.*e[33]*e[20]*e[11]-1.*e[33]*e[19]*e[10]-1.*e[33]*e[22]*e[13];
    A[14]=e[18]*e[13]*e[17]+e[9]*e[13]*e[26]+e[9]*e[22]*e[17]-e[9]*e[14]*e[25]-e[18]*e[14]*e[16]-e[15]*e[13]*e[20]-e[15]*e[22]*e[11]+e[12]*e[16]*e[20]+e[12]*e[25]*e[11]-e[12]*e[10]*e[26]-e[12]*e[19]*e[17]+e[21]*e[16]*e[11]-e[21]*e[10]*e[17]-e[9]*e[23]*e[16]+e[24]*e[10]*e[14]-e[24]*e[13]*e[11]+e[15]*e[10]*e[23]+e[15]*e[19]*e[14];
    A[156]=e[21]*e[12]*e[24]+e[21]*e[23]*e[17]+e[21]*e[14]*e[26]+e[21]*e[22]*e[16]+e[21]*e[13]*e[25]+e[24]*e[26]*e[17]+e[24]*e[25]*e[16]+e[9]*e[19]*e[25]+e[9]*e[18]*e[24]+e[9]*e[20]*e[26]+e[12]*e[22]*e[25]+e[12]*e[23]*e[26]+e[18]*e[20]*e[17]+e[18]*e[11]*e[26]+e[18]*e[19]*e[16]+e[18]*e[10]*e[25]-1.*e[24]*e[23]*e[14]-1.*e[24]*e[20]*e[11]-1.*e[24]*e[19]*e[10]-1.*e[24]*e[22]*e[13]+.5000000000*ep2[21]*e[15]+1.500000000*ep2[24]*e[15]+.5000000000*e[15]*ep2[25]+.5000000000*e[15]*ep2[26]+.5000000000*e[15]*ep2[18]-.5000000000*e[15]*ep2[23]-.5000000000*e[15]*ep2[19]-.5000000000*e[15]*ep2[20]-.5000000000*e[15]*ep2[22];
    A[18]=e[6]*e[1]*e[14]+e[15]*e[1]*e[5]-e[0]*e[5]*e[16]-e[0]*e[14]*e[7]+e[0]*e[13]*e[8]-e[15]*e[4]*e[2]+e[12]*e[7]*e[2]+e[6]*e[10]*e[5]+e[3]*e[7]*e[11]-e[6]*e[4]*e[11]+e[3]*e[16]*e[2]-e[6]*e[13]*e[2]-e[3]*e[1]*e[17]-e[9]*e[5]*e[7]-e[3]*e[10]*e[8]-e[12]*e[1]*e[8]+e[0]*e[4]*e[17]+e[9]*e[4]*e[8];
    A[128]=-.5000000000*e[14]*ep2[16]-.5000000000*e[14]*ep2[10]-.5000000000*e[14]*ep2[9]+e[11]*e[9]*e[12]+.5000000000*ep3[14]+e[17]*e[13]*e[16]+.5000000000*e[14]*ep2[12]+e[11]*e[10]*e[13]-.5000000000*e[14]*ep2[15]+.5000000000*e[14]*ep2[17]+e[17]*e[12]*e[15]+.5000000000*ep2[11]*e[14]+.5000000000*e[14]*ep2[13];
    A[19]=-e[21]*e[19]*e[8]+e[18]*e[4]*e[26]-e[18]*e[5]*e[25]-e[18]*e[23]*e[7]+e[21]*e[25]*e[2]-e[21]*e[1]*e[26]+e[6]*e[19]*e[23]+e[18]*e[22]*e[8]-e[0]*e[23]*e[25]-e[6]*e[22]*e[20]+e[24]*e[1]*e[23]+e[24]*e[19]*e[5]-e[24]*e[4]*e[20]-e[24]*e[22]*e[2]+e[3]*e[25]*e[20]-e[3]*e[19]*e[26]+e[0]*e[22]*e[26]+e[21]*e[7]*e[20];
    A[129]=.5000000000*ep2[20]*e[32]+1.500000000*e[32]*ep2[23]+.5000000000*e[32]*ep2[22]+.5000000000*e[32]*ep2[21]+.5000000000*e[32]*ep2[26]-.5000000000*e[32]*ep2[18]-.5000000000*e[32]*ep2[19]-.5000000000*e[32]*ep2[24]-.5000000000*e[32]*ep2[25]+e[20]*e[27]*e[21]+e[20]*e[18]*e[30]+e[20]*e[28]*e[22]+e[20]*e[19]*e[31]+e[20]*e[29]*e[23]+e[29]*e[19]*e[22]+e[29]*e[18]*e[21]+e[23]*e[30]*e[21]+e[23]*e[31]*e[22]+e[26]*e[30]*e[24]+e[26]*e[21]*e[33]+e[26]*e[31]*e[25]+e[26]*e[22]*e[34]+e[26]*e[23]*e[35]+e[35]*e[22]*e[25]+e[35]*e[21]*e[24]-1.*e[23]*e[27]*e[18]-1.*e[23]*e[33]*e[24]-1.*e[23]*e[28]*e[19]-1.*e[23]*e[34]*e[25];
    A[16]=-e[9]*e[23]*e[25]-e[21]*e[10]*e[26]-e[21]*e[19]*e[17]-e[18]*e[23]*e[16]+e[18]*e[13]*e[26]+e[12]*e[25]*e[20]-e[12]*e[19]*e[26]-e[15]*e[22]*e[20]+e[21]*e[16]*e[20]+e[21]*e[25]*e[11]+e[24]*e[10]*e[23]+e[24]*e[19]*e[14]-e[24]*e[13]*e[20]-e[24]*e[22]*e[11]+e[18]*e[22]*e[17]-e[18]*e[14]*e[25]+e[9]*e[22]*e[26]+e[15]*e[19]*e[23];
    A[130]=.5000000000*e[23]*ep2[21]+e[20]*e[19]*e[22]+e[20]*e[18]*e[21]+.5000000000*ep3[23]+e[26]*e[22]*e[25]+.5000000000*e[23]*ep2[26]-.5000000000*e[23]*ep2[18]+.5000000000*e[23]*ep2[22]-.5000000000*e[23]*ep2[19]+e[26]*e[21]*e[24]+.5000000000*ep2[20]*e[23]-.5000000000*e[23]*ep2[24]-.5000000000*e[23]*ep2[25];
    A[17]=e[18]*e[13]*e[35]-e[18]*e[32]*e[16]+e[18]*e[31]*e[17]-e[18]*e[14]*e[34]+e[27]*e[13]*e[26]+e[27]*e[22]*e[17]-e[27]*e[14]*e[25]-e[27]*e[23]*e[16]-e[9]*e[32]*e[25]+e[9]*e[22]*e[35]-e[9]*e[23]*e[34]+e[9]*e[31]*e[26]+e[15]*e[19]*e[32]+e[15]*e[28]*e[23]-e[15]*e[22]*e[29]-e[15]*e[31]*e[20]+e[24]*e[10]*e[32]+e[24]*e[28]*e[14]-e[24]*e[13]*e[29]-e[24]*e[31]*e[11]+e[33]*e[10]*e[23]+e[33]*e[19]*e[14]-e[33]*e[13]*e[20]-e[33]*e[22]*e[11]+e[21]*e[16]*e[29]-e[21]*e[10]*e[35]-e[21]*e[28]*e[17]+e[30]*e[16]*e[20]+e[30]*e[25]*e[11]-e[30]*e[10]*e[26]-e[30]*e[19]*e[17]-e[12]*e[28]*e[26]-e[12]*e[19]*e[35]+e[12]*e[34]*e[20]+e[12]*e[25]*e[29]+e[21]*e[34]*e[11];
    A[131]=-1.*e[32]*e[10]*e[1]+e[32]*e[13]*e[4]-1.*e[32]*e[16]*e[7]-1.*e[32]*e[15]*e[6]-1.*e[32]*e[9]*e[0]+e[32]*e[12]*e[3]+e[17]*e[30]*e[6]+e[17]*e[3]*e[33]+e[17]*e[31]*e[7]+e[17]*e[4]*e[34]+e[17]*e[5]*e[35]-1.*e[5]*e[27]*e[9]-1.*e[5]*e[28]*e[10]-1.*e[5]*e[33]*e[15]-1.*e[5]*e[34]*e[16]+e[5]*e[29]*e[11]+e[35]*e[12]*e[6]+e[35]*e[3]*e[15]+e[35]*e[13]*e[7]+e[35]*e[4]*e[16]+e[11]*e[27]*e[3]+e[11]*e[0]*e[30]+e[11]*e[28]*e[4]+e[11]*e[1]*e[31]+e[29]*e[9]*e[3]+e[29]*e[0]*e[12]+e[29]*e[10]*e[4]+e[29]*e[1]*e[13]+e[5]*e[30]*e[12]+3.*e[5]*e[32]*e[14]+e[5]*e[31]*e[13]+e[8]*e[30]*e[15]+e[8]*e[12]*e[33]+e[8]*e[32]*e[17]+e[8]*e[14]*e[35]+e[8]*e[31]*e[16]+e[8]*e[13]*e[34]+e[2]*e[27]*e[12]+e[2]*e[9]*e[30]+e[2]*e[29]*e[14]+e[2]*e[11]*e[32]+e[2]*e[28]*e[13]+e[2]*e[10]*e[31]-1.*e[14]*e[27]*e[0]-1.*e[14]*e[34]*e[7]-1.*e[14]*e[33]*e[6]+e[14]*e[30]*e[3]-1.*e[14]*e[28]*e[1]+e[14]*e[31]*e[4];
    A[22]=.5000000000*e[18]*ep2[29]+.5000000000*e[18]*ep2[28]+.5000000000*e[18]*ep2[30]+.5000000000*e[18]*ep2[33]-.5000000000*e[18]*ep2[32]-.5000000000*e[18]*ep2[31]-.5000000000*e[18]*ep2[34]-.5000000000*e[18]*ep2[35]+1.500000000*e[18]*ep2[27]+e[27]*e[28]*e[19]+e[27]*e[29]*e[20]+e[21]*e[27]*e[30]+e[21]*e[29]*e[32]+e[21]*e[28]*e[31]+e[30]*e[28]*e[22]+e[30]*e[19]*e[31]+e[30]*e[29]*e[23]+e[30]*e[20]*e[32]+e[24]*e[27]*e[33]+e[24]*e[29]*e[35]+e[24]*e[28]*e[34]+e[33]*e[28]*e[25]+e[33]*e[19]*e[34]+e[33]*e[29]*e[26]+e[33]*e[20]*e[35]-1.*e[27]*e[35]*e[26]-1.*e[27]*e[31]*e[22]-1.*e[27]*e[32]*e[23]-1.*e[27]*e[34]*e[25];
    A[132]=e[20]*e[1]*e[4]+e[20]*e[0]*e[3]+e[20]*e[2]*e[5]+e[5]*e[21]*e[3]+e[5]*e[22]*e[4]+e[8]*e[21]*e[6]+e[8]*e[3]*e[24]+e[8]*e[22]*e[7]+e[8]*e[4]*e[25]+e[8]*e[5]*e[26]+e[26]*e[4]*e[7]+e[26]*e[3]*e[6]+e[2]*e[18]*e[3]+e[2]*e[0]*e[21]+e[2]*e[19]*e[4]+e[2]*e[1]*e[22]-1.*e[5]*e[19]*e[1]-1.*e[5]*e[18]*e[0]-1.*e[5]*e[25]*e[7]-1.*e[5]*e[24]*e[6]+.5000000000*e[23]*ep2[4]-.5000000000*e[23]*ep2[0]-.5000000000*e[23]*ep2[6]+1.500000000*e[23]*ep2[5]-.5000000000*e[23]*ep2[1]-.5000000000*e[23]*ep2[7]+.5000000000*e[23]*ep2[3]+.5000000000*e[23]*ep2[2]+.5000000000*e[23]*ep2[8];
    A[23]=1.500000000*e[9]*ep2[27]+.5000000000*e[9]*ep2[29]+.5000000000*e[9]*ep2[28]-.5000000000*e[9]*ep2[32]-.5000000000*e[9]*ep2[31]+.5000000000*e[9]*ep2[33]+.5000000000*e[9]*ep2[30]-.5000000000*e[9]*ep2[34]-.5000000000*e[9]*ep2[35]+e[33]*e[27]*e[15]+e[33]*e[29]*e[17]+e[33]*e[11]*e[35]+e[33]*e[28]*e[16]+e[33]*e[10]*e[34]+e[27]*e[29]*e[11]+e[27]*e[28]*e[10]+e[27]*e[30]*e[12]-1.*e[27]*e[31]*e[13]-1.*e[27]*e[32]*e[14]-1.*e[27]*e[34]*e[16]-1.*e[27]*e[35]*e[17]+e[30]*e[29]*e[14]+e[30]*e[11]*e[32]+e[30]*e[28]*e[13]+e[30]*e[10]*e[31]+e[12]*e[29]*e[32]+e[12]*e[28]*e[31]+e[15]*e[29]*e[35]+e[15]*e[28]*e[34];
    A[133]=-1.*e[32]*e[24]*e[6]+e[8]*e[30]*e[24]+e[8]*e[21]*e[33]+e[8]*e[31]*e[25]+e[8]*e[22]*e[34]+e[26]*e[30]*e[6]+e[26]*e[3]*e[33]+e[26]*e[31]*e[7]+e[26]*e[4]*e[34]+e[26]*e[32]*e[8]+e[26]*e[5]*e[35]+e[35]*e[21]*e[6]+e[35]*e[3]*e[24]+e[35]*e[22]*e[7]+e[35]*e[4]*e[25]+e[35]*e[23]*e[8]+e[2]*e[27]*e[21]+e[2]*e[18]*e[30]+e[2]*e[28]*e[22]+e[2]*e[19]*e[31]+e[2]*e[29]*e[23]+e[2]*e[20]*e[32]+e[20]*e[27]*e[3]+e[20]*e[0]*e[30]+e[20]*e[28]*e[4]+e[20]*e[1]*e[31]+e[20]*e[29]*e[5]+e[29]*e[18]*e[3]+e[29]*e[0]*e[21]+e[29]*e[19]*e[4]+e[29]*e[1]*e[22]+e[5]*e[30]*e[21]+e[5]*e[31]*e[22]+3.*e[5]*e[32]*e[23]-1.*e[5]*e[27]*e[18]-1.*e[5]*e[33]*e[24]-1.*e[5]*e[28]*e[19]-1.*e[5]*e[34]*e[25]-1.*e[23]*e[27]*e[0]-1.*e[23]*e[34]*e[7]-1.*e[23]*e[33]*e[6]+e[23]*e[30]*e[3]-1.*e[23]*e[28]*e[1]+e[23]*e[31]*e[4]+e[32]*e[21]*e[3]-1.*e[32]*e[19]*e[1]+e[32]*e[22]*e[4]-1.*e[32]*e[18]*e[0]-1.*e[32]*e[25]*e[7];
    A[20]=.5000000000*e[27]*ep2[33]-.5000000000*e[27]*ep2[32]-.5000000000*e[27]*ep2[31]-.5000000000*e[27]*ep2[34]-.5000000000*e[27]*ep2[35]+e[33]*e[29]*e[35]+.5000000000*e[27]*ep2[29]+e[30]*e[29]*e[32]+e[30]*e[28]*e[31]+e[33]*e[28]*e[34]+.5000000000*e[27]*ep2[28]+.5000000000*e[27]*ep2[30]+.5000000000*ep3[27];
    A[134]=e[14]*e[21]*e[12]+e[14]*e[22]*e[13]+e[17]*e[21]*e[15]+e[17]*e[12]*e[24]+e[17]*e[14]*e[26]+e[17]*e[22]*e[16]+e[17]*e[13]*e[25]+e[26]*e[12]*e[15]+e[26]*e[13]*e[16]-1.*e[14]*e[24]*e[15]-1.*e[14]*e[25]*e[16]-1.*e[14]*e[18]*e[9]-1.*e[14]*e[19]*e[10]+e[11]*e[18]*e[12]+e[11]*e[9]*e[21]+e[11]*e[19]*e[13]+e[11]*e[10]*e[22]+e[20]*e[11]*e[14]+e[20]*e[9]*e[12]+e[20]*e[10]*e[13]+1.500000000*e[23]*ep2[14]+.5000000000*e[23]*ep2[12]+.5000000000*e[23]*ep2[13]+.5000000000*e[23]*ep2[17]+.5000000000*ep2[11]*e[23]-.5000000000*e[23]*ep2[16]-.5000000000*e[23]*ep2[9]-.5000000000*e[23]*ep2[15]-.5000000000*e[23]*ep2[10];
    A[21]=1.500000000*e[0]*ep2[27]+.5000000000*e[0]*ep2[29]+.5000000000*e[0]*ep2[28]+.5000000000*e[0]*ep2[30]-.5000000000*e[0]*ep2[32]-.5000000000*e[0]*ep2[31]+.5000000000*e[0]*ep2[33]-.5000000000*e[0]*ep2[34]-.5000000000*e[0]*ep2[35]-1.*e[27]*e[31]*e[4]+e[3]*e[27]*e[30]+e[3]*e[29]*e[32]+e[3]*e[28]*e[31]+e[30]*e[28]*e[4]+e[30]*e[1]*e[31]+e[30]*e[29]*e[5]+e[30]*e[2]*e[32]+e[6]*e[27]*e[33]+e[6]*e[29]*e[35]+e[6]*e[28]*e[34]+e[27]*e[28]*e[1]+e[27]*e[29]*e[2]+e[33]*e[28]*e[7]+e[33]*e[1]*e[34]+e[33]*e[29]*e[8]+e[33]*e[2]*e[35]-1.*e[27]*e[34]*e[7]-1.*e[27]*e[32]*e[5]-1.*e[27]*e[35]*e[8];
    A[135]=e[14]*e[12]*e[3]+e[14]*e[13]*e[4]+e[17]*e[12]*e[6]+e[17]*e[3]*e[15]+e[17]*e[13]*e[7]+e[17]*e[4]*e[16]+e[17]*e[14]*e[8]+e[8]*e[12]*e[15]+e[8]*e[13]*e[16]+e[2]*e[11]*e[14]+e[2]*e[9]*e[12]+e[2]*e[10]*e[13]+e[11]*e[9]*e[3]+e[11]*e[0]*e[12]+e[11]*e[10]*e[4]+e[11]*e[1]*e[13]-1.*e[14]*e[10]*e[1]-1.*e[14]*e[16]*e[7]-1.*e[14]*e[15]*e[6]-1.*e[14]*e[9]*e[0]-.5000000000*e[5]*ep2[16]-.5000000000*e[5]*ep2[9]+.5000000000*e[5]*ep2[11]+.5000000000*e[5]*ep2[12]-.5000000000*e[5]*ep2[15]-.5000000000*e[5]*ep2[10]+.5000000000*e[5]*ep2[13]+1.500000000*ep2[14]*e[5]+.5000000000*e[5]*ep2[17];
    A[27]=1.500000000*e[27]*ep2[9]-.5000000000*e[27]*ep2[16]+.5000000000*e[27]*ep2[11]+.5000000000*e[27]*ep2[12]+.5000000000*e[27]*ep2[15]-.5000000000*e[27]*ep2[17]+.5000000000*e[27]*ep2[10]-.5000000000*e[27]*ep2[14]-.5000000000*e[27]*ep2[13]+e[12]*e[10]*e[31]+e[30]*e[11]*e[14]+e[30]*e[10]*e[13]+e[15]*e[9]*e[33]+e[15]*e[29]*e[17]+e[15]*e[11]*e[35]+e[15]*e[28]*e[16]+e[15]*e[10]*e[34]+e[33]*e[11]*e[17]+e[33]*e[10]*e[16]-1.*e[9]*e[31]*e[13]-1.*e[9]*e[32]*e[14]-1.*e[9]*e[34]*e[16]-1.*e[9]*e[35]*e[17]+e[9]*e[29]*e[11]+e[9]*e[28]*e[10]+e[12]*e[9]*e[30]+e[12]*e[29]*e[14]+e[12]*e[11]*e[32]+e[12]*e[28]*e[13];
    A[137]=e[29]*e[18]*e[12]+e[29]*e[9]*e[21]+e[29]*e[19]*e[13]+e[29]*e[10]*e[22]+e[17]*e[30]*e[24]+e[17]*e[21]*e[33]+e[17]*e[31]*e[25]+e[17]*e[22]*e[34]+e[17]*e[32]*e[26]+e[17]*e[23]*e[35]-1.*e[23]*e[27]*e[9]-1.*e[23]*e[28]*e[10]-1.*e[23]*e[33]*e[15]-1.*e[23]*e[34]*e[16]-1.*e[32]*e[24]*e[15]-1.*e[32]*e[25]*e[16]-1.*e[32]*e[18]*e[9]-1.*e[32]*e[19]*e[10]+e[26]*e[30]*e[15]+e[26]*e[12]*e[33]+e[26]*e[31]*e[16]+e[26]*e[13]*e[34]+e[35]*e[21]*e[15]+e[35]*e[12]*e[24]+e[35]*e[22]*e[16]+e[35]*e[13]*e[25]+e[14]*e[30]*e[21]+e[14]*e[31]*e[22]+3.*e[14]*e[32]*e[23]+e[11]*e[27]*e[21]+e[11]*e[18]*e[30]+e[11]*e[28]*e[22]+e[11]*e[19]*e[31]+e[11]*e[29]*e[23]+e[11]*e[20]*e[32]+e[23]*e[30]*e[12]+e[23]*e[31]*e[13]+e[32]*e[21]*e[12]+e[32]*e[22]*e[13]-1.*e[14]*e[27]*e[18]-1.*e[14]*e[33]*e[24]+e[14]*e[29]*e[20]+e[14]*e[35]*e[26]-1.*e[14]*e[28]*e[19]-1.*e[14]*e[34]*e[25]+e[20]*e[27]*e[12]+e[20]*e[9]*e[30]+e[20]*e[28]*e[13]+e[20]*e[10]*e[31];
    A[26]=.5000000000*e[0]*ep2[1]+.5000000000*e[0]*ep2[2]+e[6]*e[2]*e[8]+e[6]*e[1]*e[7]+.5000000000*e[0]*ep2[3]+e[3]*e[1]*e[4]+.5000000000*e[0]*ep2[6]+e[3]*e[2]*e[5]-.5000000000*e[0]*ep2[5]-.5000000000*e[0]*ep2[8]+.5000000000*ep3[0]-.5000000000*e[0]*ep2[7]-.5000000000*e[0]*ep2[4];
    A[136]=1.500000000*ep2[23]*e[14]+.5000000000*e[14]*ep2[26]-.5000000000*e[14]*ep2[18]-.5000000000*e[14]*ep2[19]+.5000000000*e[14]*ep2[20]+.5000000000*e[14]*ep2[22]-.5000000000*e[14]*ep2[24]+.5000000000*e[14]*ep2[21]-.5000000000*e[14]*ep2[25]+e[23]*e[21]*e[12]+e[23]*e[22]*e[13]+e[26]*e[21]*e[15]+e[26]*e[12]*e[24]+e[26]*e[23]*e[17]+e[26]*e[22]*e[16]+e[26]*e[13]*e[25]+e[17]*e[22]*e[25]+e[17]*e[21]*e[24]+e[11]*e[19]*e[22]+e[11]*e[18]*e[21]+e[11]*e[20]*e[23]+e[20]*e[18]*e[12]+e[20]*e[9]*e[21]+e[20]*e[19]*e[13]+e[20]*e[10]*e[22]-1.*e[23]*e[24]*e[15]-1.*e[23]*e[25]*e[16]-1.*e[23]*e[18]*e[9]-1.*e[23]*e[19]*e[10];
    A[25]=1.500000000*e[27]*ep2[0]-.5000000000*e[27]*ep2[4]+.5000000000*e[27]*ep2[6]-.5000000000*e[27]*ep2[5]+.5000000000*e[27]*ep2[1]-.5000000000*e[27]*ep2[7]+.5000000000*e[27]*ep2[3]+.5000000000*e[27]*ep2[2]-.5000000000*e[27]*ep2[8]+e[0]*e[33]*e[6]+e[0]*e[30]*e[3]-1.*e[0]*e[35]*e[8]-1.*e[0]*e[31]*e[4]+e[3]*e[28]*e[4]+e[3]*e[1]*e[31]+e[3]*e[29]*e[5]+e[3]*e[2]*e[32]+e[30]*e[1]*e[4]+e[30]*e[2]*e[5]+e[6]*e[28]*e[7]+e[6]*e[1]*e[34]+e[6]*e[29]*e[8]+e[6]*e[2]*e[35]+e[33]*e[1]*e[7]+e[33]*e[2]*e[8]+e[0]*e[28]*e[1]+e[0]*e[29]*e[2]-1.*e[0]*e[34]*e[7]-1.*e[0]*e[32]*e[5];
    A[139]=e[8]*e[22]*e[25]+e[8]*e[21]*e[24]+e[20]*e[18]*e[3]+e[20]*e[0]*e[21]+e[20]*e[19]*e[4]+e[20]*e[1]*e[22]+e[20]*e[2]*e[23]+e[23]*e[21]*e[3]+e[23]*e[22]*e[4]+e[23]*e[26]*e[8]-1.*e[23]*e[19]*e[1]-1.*e[23]*e[18]*e[0]-1.*e[23]*e[25]*e[7]-1.*e[23]*e[24]*e[6]+e[2]*e[19]*e[22]+e[2]*e[18]*e[21]+e[26]*e[21]*e[6]+e[26]*e[3]*e[24]+e[26]*e[22]*e[7]+e[26]*e[4]*e[25]+.5000000000*ep2[20]*e[5]+1.500000000*ep2[23]*e[5]+.5000000000*e[5]*ep2[22]+.5000000000*e[5]*ep2[21]+.5000000000*e[5]*ep2[26]-.5000000000*e[5]*ep2[18]-.5000000000*e[5]*ep2[19]-.5000000000*e[5]*ep2[24]-.5000000000*e[5]*ep2[25];
    A[24]=e[24]*e[11]*e[8]+e[24]*e[2]*e[17]+3.*e[9]*e[18]*e[0]+e[9]*e[19]*e[1]+e[9]*e[20]*e[2]+e[18]*e[10]*e[1]+e[18]*e[11]*e[2]+e[3]*e[18]*e[12]+e[3]*e[9]*e[21]+e[3]*e[20]*e[14]+e[3]*e[11]*e[23]+e[3]*e[19]*e[13]+e[3]*e[10]*e[22]+e[6]*e[18]*e[15]+e[6]*e[9]*e[24]+e[6]*e[20]*e[17]+e[6]*e[11]*e[26]+e[6]*e[19]*e[16]+e[6]*e[10]*e[25]+e[0]*e[20]*e[11]+e[0]*e[19]*e[10]-1.*e[9]*e[26]*e[8]-1.*e[9]*e[22]*e[4]-1.*e[9]*e[25]*e[7]-1.*e[9]*e[23]*e[5]+e[12]*e[0]*e[21]+e[12]*e[19]*e[4]+e[12]*e[1]*e[22]+e[12]*e[20]*e[5]+e[12]*e[2]*e[23]-1.*e[18]*e[13]*e[4]-1.*e[18]*e[16]*e[7]-1.*e[18]*e[14]*e[5]-1.*e[18]*e[17]*e[8]+e[21]*e[10]*e[4]+e[21]*e[1]*e[13]+e[21]*e[11]*e[5]+e[21]*e[2]*e[14]+e[15]*e[0]*e[24]+e[15]*e[19]*e[7]+e[15]*e[1]*e[25]+e[15]*e[20]*e[8]+e[15]*e[2]*e[26]-1.*e[0]*e[23]*e[14]-1.*e[0]*e[25]*e[16]-1.*e[0]*e[26]*e[17]-1.*e[0]*e[22]*e[13]+e[24]*e[10]*e[7]+e[24]*e[1]*e[16];
    A[138]=e[11]*e[1]*e[4]+e[11]*e[0]*e[3]+e[11]*e[2]*e[5]+e[5]*e[12]*e[3]+e[5]*e[13]*e[4]+e[8]*e[12]*e[6]+e[8]*e[3]*e[15]+e[8]*e[13]*e[7]+e[8]*e[4]*e[16]+e[8]*e[5]*e[17]+e[17]*e[4]*e[7]+e[17]*e[3]*e[6]-1.*e[5]*e[10]*e[1]-1.*e[5]*e[16]*e[7]-1.*e[5]*e[15]*e[6]-1.*e[5]*e[9]*e[0]+e[2]*e[9]*e[3]+e[2]*e[0]*e[12]+e[2]*e[10]*e[4]+e[2]*e[1]*e[13]+.5000000000*ep2[2]*e[14]-.5000000000*e[14]*ep2[0]-.5000000000*e[14]*ep2[6]-.5000000000*e[14]*ep2[1]-.5000000000*e[14]*ep2[7]+1.500000000*e[14]*ep2[5]+.5000000000*e[14]*ep2[4]+.5000000000*e[14]*ep2[3]+.5000000000*e[14]*ep2[8];
    A[31]=e[3]*e[27]*e[12]+e[3]*e[9]*e[30]+e[3]*e[29]*e[14]+e[3]*e[11]*e[32]+e[3]*e[28]*e[13]+e[3]*e[10]*e[31]+e[6]*e[27]*e[15]+e[6]*e[9]*e[33]+e[6]*e[29]*e[17]+e[6]*e[11]*e[35]+e[6]*e[28]*e[16]+e[6]*e[10]*e[34]+3.*e[0]*e[27]*e[9]+e[0]*e[29]*e[11]+e[0]*e[28]*e[10]-1.*e[9]*e[34]*e[7]-1.*e[9]*e[32]*e[5]-1.*e[9]*e[35]*e[8]+e[9]*e[29]*e[2]+e[9]*e[28]*e[1]-1.*e[9]*e[31]*e[4]+e[12]*e[0]*e[30]+e[12]*e[28]*e[4]+e[12]*e[1]*e[31]+e[12]*e[29]*e[5]+e[12]*e[2]*e[32]+e[27]*e[11]*e[2]+e[27]*e[10]*e[1]-1.*e[27]*e[13]*e[4]-1.*e[27]*e[16]*e[7]-1.*e[27]*e[14]*e[5]-1.*e[27]*e[17]*e[8]+e[30]*e[10]*e[4]+e[30]*e[1]*e[13]+e[30]*e[11]*e[5]+e[30]*e[2]*e[14]+e[15]*e[0]*e[33]+e[15]*e[28]*e[7]+e[15]*e[1]*e[34]+e[15]*e[29]*e[8]+e[15]*e[2]*e[35]-1.*e[0]*e[31]*e[13]-1.*e[0]*e[32]*e[14]-1.*e[0]*e[34]*e[16]-1.*e[0]*e[35]*e[17]+e[33]*e[10]*e[7]+e[33]*e[1]*e[16]+e[33]*e[11]*e[8]+e[33]*e[2]*e[17];
    A[141]=.5000000000*ep2[30]*e[6]+.5000000000*e[6]*ep2[27]-.5000000000*e[6]*ep2[32]-.5000000000*e[6]*ep2[28]-.5000000000*e[6]*ep2[29]-.5000000000*e[6]*ep2[31]+1.500000000*e[6]*ep2[33]+.5000000000*e[6]*ep2[34]+.5000000000*e[6]*ep2[35]+e[0]*e[27]*e[33]+e[0]*e[29]*e[35]+e[0]*e[28]*e[34]+e[3]*e[30]*e[33]+e[3]*e[32]*e[35]+e[3]*e[31]*e[34]+e[30]*e[31]*e[7]+e[30]*e[4]*e[34]+e[30]*e[32]*e[8]+e[30]*e[5]*e[35]+e[27]*e[28]*e[7]+e[27]*e[1]*e[34]+e[27]*e[29]*e[8]+e[27]*e[2]*e[35]+e[33]*e[34]*e[7]+e[33]*e[35]*e[8]-1.*e[33]*e[32]*e[5]-1.*e[33]*e[29]*e[2]-1.*e[33]*e[28]*e[1]-1.*e[33]*e[31]*e[4];
    A[30]=e[24]*e[20]*e[26]+e[21]*e[19]*e[22]-.5000000000*e[18]*ep2[22]-.5000000000*e[18]*ep2[25]+.5000000000*ep3[18]+.5000000000*e[18]*ep2[21]+e[21]*e[20]*e[23]+.5000000000*e[18]*ep2[20]+.5000000000*e[18]*ep2[19]+.5000000000*e[18]*ep2[24]+e[24]*e[19]*e[25]-.5000000000*e[18]*ep2[23]-.5000000000*e[18]*ep2[26];
    A[140]=.5000000000*e[33]*ep2[35]+.5000000000*ep3[33]+.5000000000*ep2[27]*e[33]+.5000000000*ep2[30]*e[33]-.5000000000*e[33]*ep2[29]+.5000000000*e[33]*ep2[34]-.5000000000*e[33]*ep2[32]-.5000000000*e[33]*ep2[28]+e[30]*e[32]*e[35]-.5000000000*e[33]*ep2[31]+e[27]*e[29]*e[35]+e[27]*e[28]*e[34]+e[30]*e[31]*e[34];
    A[29]=1.500000000*e[27]*ep2[18]+.5000000000*e[27]*ep2[19]+.5000000000*e[27]*ep2[20]+.5000000000*e[27]*ep2[21]+.5000000000*e[27]*ep2[24]-.5000000000*e[27]*ep2[26]-.5000000000*e[27]*ep2[23]-.5000000000*e[27]*ep2[22]-.5000000000*e[27]*ep2[25]+e[33]*e[20]*e[26]-1.*e[18]*e[35]*e[26]-1.*e[18]*e[31]*e[22]-1.*e[18]*e[32]*e[23]-1.*e[18]*e[34]*e[25]+e[18]*e[28]*e[19]+e[18]*e[29]*e[20]+e[21]*e[18]*e[30]+e[21]*e[28]*e[22]+e[21]*e[19]*e[31]+e[21]*e[29]*e[23]+e[21]*e[20]*e[32]+e[30]*e[19]*e[22]+e[30]*e[20]*e[23]+e[24]*e[18]*e[33]+e[24]*e[28]*e[25]+e[24]*e[19]*e[34]+e[24]*e[29]*e[26]+e[24]*e[20]*e[35]+e[33]*e[19]*e[25];
    A[143]=e[9]*e[27]*e[33]+e[9]*e[29]*e[35]+e[9]*e[28]*e[34]+e[33]*e[35]*e[17]+e[33]*e[34]*e[16]+e[27]*e[29]*e[17]+e[27]*e[11]*e[35]+e[27]*e[28]*e[16]+e[27]*e[10]*e[34]+e[33]*e[30]*e[12]-1.*e[33]*e[28]*e[10]-1.*e[33]*e[31]*e[13]-1.*e[33]*e[32]*e[14]-1.*e[33]*e[29]*e[11]+e[30]*e[32]*e[17]+e[30]*e[14]*e[35]+e[30]*e[31]*e[16]+e[30]*e[13]*e[34]+e[12]*e[32]*e[35]+e[12]*e[31]*e[34]+.5000000000*e[15]*ep2[27]-.5000000000*e[15]*ep2[32]-.5000000000*e[15]*ep2[28]-.5000000000*e[15]*ep2[29]-.5000000000*e[15]*ep2[31]+1.500000000*e[15]*ep2[33]+.5000000000*e[15]*ep2[30]+.5000000000*e[15]*ep2[34]+.5000000000*e[15]*ep2[35];
    A[28]=.5000000000*e[9]*ep2[12]-.5000000000*e[9]*ep2[16]+.5000000000*e[9]*ep2[10]-.5000000000*e[9]*ep2[17]-.5000000000*e[9]*ep2[13]+e[15]*e[10]*e[16]+e[12]*e[11]*e[14]+.5000000000*e[9]*ep2[11]+.5000000000*e[9]*ep2[15]-.5000000000*e[9]*ep2[14]+e[15]*e[11]*e[17]+.5000000000*ep3[9]+e[12]*e[10]*e[13];
    A[142]=e[18]*e[27]*e[33]+e[18]*e[29]*e[35]+e[18]*e[28]*e[34]+e[27]*e[28]*e[25]+e[27]*e[19]*e[34]+e[27]*e[29]*e[26]+e[27]*e[20]*e[35]+e[21]*e[30]*e[33]+e[21]*e[32]*e[35]+e[21]*e[31]*e[34]+e[30]*e[31]*e[25]+e[30]*e[22]*e[34]+e[30]*e[32]*e[26]+e[30]*e[23]*e[35]+e[33]*e[34]*e[25]+e[33]*e[35]*e[26]-1.*e[33]*e[29]*e[20]-1.*e[33]*e[31]*e[22]-1.*e[33]*e[32]*e[23]-1.*e[33]*e[28]*e[19]+.5000000000*ep2[27]*e[24]+.5000000000*ep2[30]*e[24]+1.500000000*e[24]*ep2[33]+.5000000000*e[24]*ep2[35]+.5000000000*e[24]*ep2[34]-.5000000000*e[24]*ep2[32]-.5000000000*e[24]*ep2[28]-.5000000000*e[24]*ep2[29]-.5000000000*e[24]*ep2[31];
    A[36]=.5000000000*e[9]*ep2[21]+.5000000000*e[9]*ep2[24]+.5000000000*e[9]*ep2[19]+1.500000000*e[9]*ep2[18]+.5000000000*e[9]*ep2[20]-.5000000000*e[9]*ep2[26]-.5000000000*e[9]*ep2[23]-.5000000000*e[9]*ep2[22]-.5000000000*e[9]*ep2[25]+e[21]*e[18]*e[12]+e[21]*e[20]*e[14]+e[21]*e[11]*e[23]+e[21]*e[19]*e[13]+e[21]*e[10]*e[22]+e[24]*e[18]*e[15]+e[24]*e[20]*e[17]+e[24]*e[11]*e[26]+e[24]*e[19]*e[16]+e[24]*e[10]*e[25]+e[15]*e[19]*e[25]+e[15]*e[20]*e[26]+e[12]*e[19]*e[22]+e[12]*e[20]*e[23]+e[18]*e[20]*e[11]+e[18]*e[19]*e[10]-1.*e[18]*e[23]*e[14]-1.*e[18]*e[25]*e[16]-1.*e[18]*e[26]*e[17]-1.*e[18]*e[22]*e[13];
    A[182]=.5000000000*ep2[29]*e[26]+.5000000000*ep2[32]*e[26]+.5000000000*e[26]*ep2[33]+1.500000000*e[26]*ep2[35]+.5000000000*e[26]*ep2[34]-.5000000000*e[26]*ep2[27]-.5000000000*e[26]*ep2[28]-.5000000000*e[26]*ep2[31]-.5000000000*e[26]*ep2[30]+e[20]*e[27]*e[33]+e[20]*e[29]*e[35]+e[20]*e[28]*e[34]+e[29]*e[27]*e[24]+e[29]*e[18]*e[33]+e[29]*e[28]*e[25]+e[29]*e[19]*e[34]+e[23]*e[30]*e[33]+e[23]*e[32]*e[35]+e[23]*e[31]*e[34]+e[32]*e[30]*e[24]+e[32]*e[21]*e[33]+e[32]*e[31]*e[25]+e[32]*e[22]*e[34]+e[35]*e[33]*e[24]+e[35]*e[34]*e[25]-1.*e[35]*e[27]*e[18]-1.*e[35]*e[30]*e[21]-1.*e[35]*e[31]*e[22]-1.*e[35]*e[28]*e[19];
    A[37]=e[12]*e[19]*e[31]+e[12]*e[29]*e[23]+e[12]*e[20]*e[32]+3.*e[9]*e[27]*e[18]+e[9]*e[28]*e[19]+e[9]*e[29]*e[20]+e[21]*e[9]*e[30]+e[21]*e[29]*e[14]+e[21]*e[11]*e[32]+e[21]*e[28]*e[13]+e[21]*e[10]*e[31]+e[30]*e[20]*e[14]+e[30]*e[11]*e[23]+e[30]*e[19]*e[13]+e[30]*e[10]*e[22]+e[9]*e[33]*e[24]-1.*e[9]*e[35]*e[26]-1.*e[9]*e[31]*e[22]-1.*e[9]*e[32]*e[23]-1.*e[9]*e[34]*e[25]+e[18]*e[29]*e[11]+e[18]*e[28]*e[10]+e[27]*e[20]*e[11]+e[27]*e[19]*e[10]+e[15]*e[27]*e[24]+e[15]*e[18]*e[33]+e[15]*e[28]*e[25]+e[15]*e[19]*e[34]+e[15]*e[29]*e[26]+e[15]*e[20]*e[35]-1.*e[18]*e[31]*e[13]-1.*e[18]*e[32]*e[14]-1.*e[18]*e[34]*e[16]-1.*e[18]*e[35]*e[17]-1.*e[27]*e[23]*e[14]-1.*e[27]*e[25]*e[16]-1.*e[27]*e[26]*e[17]-1.*e[27]*e[22]*e[13]+e[24]*e[29]*e[17]+e[24]*e[11]*e[35]+e[24]*e[28]*e[16]+e[24]*e[10]*e[34]+e[33]*e[20]*e[17]+e[33]*e[11]*e[26]+e[33]*e[19]*e[16]+e[33]*e[10]*e[25]+e[12]*e[27]*e[21]+e[12]*e[18]*e[30]+e[12]*e[28]*e[22];
    A[183]=-.5000000000*e[17]*ep2[27]+.5000000000*e[17]*ep2[32]-.5000000000*e[17]*ep2[28]+.5000000000*e[17]*ep2[29]-.5000000000*e[17]*ep2[31]+.5000000000*e[17]*ep2[33]-.5000000000*e[17]*ep2[30]+.5000000000*e[17]*ep2[34]+1.500000000*e[17]*ep2[35]+e[32]*e[30]*e[15]+e[32]*e[12]*e[33]+e[32]*e[31]*e[16]+e[32]*e[13]*e[34]+e[14]*e[30]*e[33]+e[14]*e[31]*e[34]+e[11]*e[27]*e[33]+e[11]*e[29]*e[35]+e[11]*e[28]*e[34]+e[35]*e[33]*e[15]+e[35]*e[34]*e[16]+e[29]*e[27]*e[15]+e[29]*e[9]*e[33]+e[29]*e[28]*e[16]+e[29]*e[10]*e[34]-1.*e[35]*e[27]*e[9]-1.*e[35]*e[30]*e[12]-1.*e[35]*e[28]*e[10]-1.*e[35]*e[31]*e[13]+e[35]*e[32]*e[14];
    A[38]=.5000000000*e[9]*ep2[1]+1.500000000*e[9]*ep2[0]+.5000000000*e[9]*ep2[2]+.5000000000*e[9]*ep2[3]+.5000000000*e[9]*ep2[6]-.5000000000*e[9]*ep2[4]-.5000000000*e[9]*ep2[5]-.5000000000*e[9]*ep2[7]-.5000000000*e[9]*ep2[8]+e[6]*e[0]*e[15]+e[6]*e[10]*e[7]+e[6]*e[1]*e[16]+e[6]*e[11]*e[8]+e[6]*e[2]*e[17]+e[15]*e[1]*e[7]+e[15]*e[2]*e[8]+e[0]*e[11]*e[2]+e[0]*e[10]*e[1]-1.*e[0]*e[13]*e[4]-1.*e[0]*e[16]*e[7]-1.*e[0]*e[14]*e[5]-1.*e[0]*e[17]*e[8]+e[3]*e[0]*e[12]+e[3]*e[10]*e[4]+e[3]*e[1]*e[13]+e[3]*e[11]*e[5]+e[3]*e[2]*e[14]+e[12]*e[1]*e[4]+e[12]*e[2]*e[5];
    A[180]=.5000000000*e[35]*ep2[33]+.5000000000*e[35]*ep2[34]-.5000000000*e[35]*ep2[27]-.5000000000*e[35]*ep2[28]-.5000000000*e[35]*ep2[31]-.5000000000*e[35]*ep2[30]+e[32]*e[31]*e[34]+.5000000000*ep2[29]*e[35]+.5000000000*ep2[32]*e[35]+e[29]*e[28]*e[34]+e[32]*e[30]*e[33]+.5000000000*ep3[35]+e[29]*e[27]*e[33];
    A[39]=.5000000000*e[0]*ep2[19]+.5000000000*e[0]*ep2[20]+.5000000000*e[0]*ep2[24]-.5000000000*e[0]*ep2[26]-.5000000000*e[0]*ep2[23]-.5000000000*e[0]*ep2[22]-.5000000000*e[0]*ep2[25]+1.500000000*ep2[18]*e[0]+.5000000000*e[0]*ep2[21]+e[18]*e[19]*e[1]+e[18]*e[20]*e[2]+e[21]*e[18]*e[3]+e[21]*e[19]*e[4]+e[21]*e[1]*e[22]+e[21]*e[20]*e[5]+e[21]*e[2]*e[23]-1.*e[18]*e[26]*e[8]-1.*e[18]*e[22]*e[4]-1.*e[18]*e[25]*e[7]-1.*e[18]*e[23]*e[5]+e[18]*e[24]*e[6]+e[3]*e[19]*e[22]+e[3]*e[20]*e[23]+e[24]*e[19]*e[7]+e[24]*e[1]*e[25]+e[24]*e[20]*e[8]+e[24]*e[2]*e[26]+e[6]*e[19]*e[25]+e[6]*e[20]*e[26];
    A[181]=.5000000000*ep2[32]*e[8]-.5000000000*e[8]*ep2[27]-.5000000000*e[8]*ep2[28]+.5000000000*e[8]*ep2[29]-.5000000000*e[8]*ep2[31]+.5000000000*e[8]*ep2[33]-.5000000000*e[8]*ep2[30]+.5000000000*e[8]*ep2[34]+1.500000000*e[8]*ep2[35]+e[2]*e[27]*e[33]+e[2]*e[29]*e[35]+e[2]*e[28]*e[34]+e[5]*e[30]*e[33]+e[5]*e[32]*e[35]+e[5]*e[31]*e[34]+e[32]*e[30]*e[6]+e[32]*e[3]*e[33]+e[32]*e[31]*e[7]+e[32]*e[4]*e[34]+e[29]*e[27]*e[6]+e[29]*e[0]*e[33]+e[29]*e[28]*e[7]+e[29]*e[1]*e[34]+e[35]*e[33]*e[6]+e[35]*e[34]*e[7]-1.*e[35]*e[27]*e[0]-1.*e[35]*e[30]*e[3]-1.*e[35]*e[28]*e[1]-1.*e[35]*e[31]*e[4];
    A[32]=-.5000000000*e[18]*ep2[4]+1.500000000*e[18]*ep2[0]+.5000000000*e[18]*ep2[6]-.5000000000*e[18]*ep2[5]+.5000000000*e[18]*ep2[1]-.5000000000*e[18]*ep2[7]+.5000000000*e[18]*ep2[3]+.5000000000*e[18]*ep2[2]-.5000000000*e[18]*ep2[8]+e[3]*e[0]*e[21]+e[3]*e[19]*e[4]+e[3]*e[1]*e[22]+e[3]*e[20]*e[5]+e[3]*e[2]*e[23]+e[21]*e[1]*e[4]+e[21]*e[2]*e[5]+e[6]*e[0]*e[24]+e[6]*e[19]*e[7]+e[6]*e[1]*e[25]+e[6]*e[20]*e[8]+e[6]*e[2]*e[26]+e[24]*e[1]*e[7]+e[24]*e[2]*e[8]+e[0]*e[19]*e[1]+e[0]*e[20]*e[2]-1.*e[0]*e[26]*e[8]-1.*e[0]*e[22]*e[4]-1.*e[0]*e[25]*e[7]-1.*e[0]*e[23]*e[5];
    A[178]=e[10]*e[1]*e[7]+e[10]*e[0]*e[6]+e[10]*e[2]*e[8]+e[4]*e[12]*e[6]+e[4]*e[3]*e[15]+e[4]*e[13]*e[7]+e[4]*e[14]*e[8]+e[4]*e[5]*e[17]+e[13]*e[3]*e[6]+e[13]*e[5]*e[8]+e[7]*e[15]*e[6]+e[7]*e[17]*e[8]-1.*e[7]*e[11]*e[2]-1.*e[7]*e[9]*e[0]-1.*e[7]*e[14]*e[5]-1.*e[7]*e[12]*e[3]+e[1]*e[9]*e[6]+e[1]*e[0]*e[15]+e[1]*e[11]*e[8]+e[1]*e[2]*e[17]+1.500000000*e[16]*ep2[7]+.5000000000*e[16]*ep2[6]+.5000000000*e[16]*ep2[8]+.5000000000*ep2[1]*e[16]-.5000000000*e[16]*ep2[0]-.5000000000*e[16]*ep2[5]-.5000000000*e[16]*ep2[3]-.5000000000*e[16]*ep2[2]+.5000000000*ep2[4]*e[16];
    A[33]=e[0]*e[30]*e[21]-1.*e[0]*e[35]*e[26]-1.*e[0]*e[31]*e[22]-1.*e[0]*e[32]*e[23]-1.*e[0]*e[34]*e[25]-1.*e[18]*e[34]*e[7]-1.*e[18]*e[32]*e[5]-1.*e[18]*e[35]*e[8]-1.*e[18]*e[31]*e[4]-1.*e[27]*e[26]*e[8]-1.*e[27]*e[22]*e[4]-1.*e[27]*e[25]*e[7]-1.*e[27]*e[23]*e[5]+e[6]*e[28]*e[25]+e[6]*e[19]*e[34]+e[6]*e[29]*e[26]+e[6]*e[20]*e[35]+e[21]*e[28]*e[4]+e[21]*e[1]*e[31]+e[21]*e[29]*e[5]+e[21]*e[2]*e[32]+e[30]*e[19]*e[4]+e[30]*e[1]*e[22]+e[30]*e[20]*e[5]+e[30]*e[2]*e[23]+e[24]*e[27]*e[6]+e[24]*e[0]*e[33]+e[24]*e[28]*e[7]+e[24]*e[1]*e[34]+e[24]*e[29]*e[8]+e[24]*e[2]*e[35]+e[33]*e[18]*e[6]+e[33]*e[19]*e[7]+e[33]*e[1]*e[25]+e[33]*e[20]*e[8]+e[33]*e[2]*e[26]+3.*e[0]*e[27]*e[18]+e[0]*e[28]*e[19]+e[0]*e[29]*e[20]+e[18]*e[28]*e[1]+e[18]*e[29]*e[2]+e[27]*e[19]*e[1]+e[27]*e[20]*e[2]+e[3]*e[27]*e[21]+e[3]*e[18]*e[30]+e[3]*e[28]*e[22]+e[3]*e[19]*e[31]+e[3]*e[29]*e[23]+e[3]*e[20]*e[32];
    A[179]=e[19]*e[18]*e[6]+e[19]*e[0]*e[24]+e[19]*e[1]*e[25]+e[19]*e[20]*e[8]+e[19]*e[2]*e[26]+e[22]*e[21]*e[6]+e[22]*e[3]*e[24]+e[22]*e[4]*e[25]+e[22]*e[23]*e[8]+e[22]*e[5]*e[26]-1.*e[25]*e[21]*e[3]+e[25]*e[26]*e[8]-1.*e[25]*e[20]*e[2]-1.*e[25]*e[18]*e[0]-1.*e[25]*e[23]*e[5]+e[25]*e[24]*e[6]+e[1]*e[18]*e[24]+e[1]*e[20]*e[26]+e[4]*e[21]*e[24]+e[4]*e[23]*e[26]+.5000000000*ep2[19]*e[7]+.5000000000*ep2[22]*e[7]+1.500000000*ep2[25]*e[7]+.5000000000*e[7]*ep2[26]-.5000000000*e[7]*ep2[18]-.5000000000*e[7]*ep2[23]-.5000000000*e[7]*ep2[20]+.5000000000*e[7]*ep2[24]-.5000000000*e[7]*ep2[21];
    A[34]=.5000000000*e[18]*ep2[11]+1.500000000*e[18]*ep2[9]+.5000000000*e[18]*ep2[10]+.5000000000*e[18]*ep2[12]+.5000000000*e[18]*ep2[15]-.5000000000*e[18]*ep2[16]-.5000000000*e[18]*ep2[17]-.5000000000*e[18]*ep2[14]-.5000000000*e[18]*ep2[13]+e[12]*e[9]*e[21]+e[12]*e[20]*e[14]+e[12]*e[11]*e[23]+e[12]*e[19]*e[13]+e[12]*e[10]*e[22]+e[21]*e[11]*e[14]+e[21]*e[10]*e[13]+e[15]*e[9]*e[24]+e[15]*e[20]*e[17]+e[15]*e[11]*e[26]+e[15]*e[19]*e[16]+e[15]*e[10]*e[25]+e[24]*e[11]*e[17]+e[24]*e[10]*e[16]-1.*e[9]*e[23]*e[14]-1.*e[9]*e[25]*e[16]-1.*e[9]*e[26]*e[17]+e[9]*e[20]*e[11]+e[9]*e[19]*e[10]-1.*e[9]*e[22]*e[13];
    A[176]=e[13]*e[21]*e[24]+e[13]*e[23]*e[26]+e[19]*e[18]*e[15]+e[19]*e[9]*e[24]+e[19]*e[20]*e[17]+e[19]*e[11]*e[26]-1.*e[25]*e[23]*e[14]-1.*e[25]*e[20]*e[11]-1.*e[25]*e[18]*e[9]-1.*e[25]*e[21]*e[12]+e[22]*e[21]*e[15]+e[22]*e[12]*e[24]+e[22]*e[23]*e[17]+e[22]*e[14]*e[26]+e[22]*e[13]*e[25]+e[25]*e[24]*e[15]+e[25]*e[26]*e[17]+e[10]*e[19]*e[25]+e[10]*e[18]*e[24]+e[10]*e[20]*e[26]-.5000000000*e[16]*ep2[18]-.5000000000*e[16]*ep2[23]+.5000000000*e[16]*ep2[19]-.5000000000*e[16]*ep2[20]-.5000000000*e[16]*ep2[21]+.5000000000*ep2[22]*e[16]+1.500000000*ep2[25]*e[16]+.5000000000*e[16]*ep2[24]+.5000000000*e[16]*ep2[26];
    A[35]=.5000000000*e[0]*ep2[12]+.5000000000*e[0]*ep2[15]+.5000000000*e[0]*ep2[11]+1.500000000*e[0]*ep2[9]+.5000000000*e[0]*ep2[10]-.5000000000*e[0]*ep2[16]-.5000000000*e[0]*ep2[17]-.5000000000*e[0]*ep2[14]-.5000000000*e[0]*ep2[13]+e[12]*e[9]*e[3]+e[12]*e[10]*e[4]+e[12]*e[1]*e[13]+e[12]*e[11]*e[5]+e[12]*e[2]*e[14]+e[15]*e[9]*e[6]+e[15]*e[10]*e[7]+e[15]*e[1]*e[16]+e[15]*e[11]*e[8]+e[15]*e[2]*e[17]+e[6]*e[11]*e[17]+e[6]*e[10]*e[16]+e[3]*e[11]*e[14]+e[3]*e[10]*e[13]+e[9]*e[10]*e[1]+e[9]*e[11]*e[2]-1.*e[9]*e[13]*e[4]-1.*e[9]*e[16]*e[7]-1.*e[9]*e[14]*e[5]-1.*e[9]*e[17]*e[8];
    A[177]=e[19]*e[11]*e[35]+e[28]*e[18]*e[15]+e[28]*e[9]*e[24]+e[28]*e[20]*e[17]+e[28]*e[11]*e[26]-1.*e[25]*e[27]*e[9]-1.*e[25]*e[30]*e[12]-1.*e[25]*e[32]*e[14]+e[25]*e[33]*e[15]+e[25]*e[35]*e[17]-1.*e[25]*e[29]*e[11]-1.*e[34]*e[23]*e[14]+e[34]*e[24]*e[15]+e[34]*e[26]*e[17]-1.*e[34]*e[20]*e[11]-1.*e[34]*e[18]*e[9]-1.*e[34]*e[21]*e[12]+e[13]*e[30]*e[24]+e[13]*e[21]*e[33]+e[13]*e[31]*e[25]+e[13]*e[22]*e[34]+e[13]*e[32]*e[26]+e[13]*e[23]*e[35]+e[10]*e[27]*e[24]+e[10]*e[18]*e[33]+e[10]*e[28]*e[25]+e[10]*e[19]*e[34]+e[10]*e[29]*e[26]+e[10]*e[20]*e[35]+e[22]*e[30]*e[15]+e[22]*e[12]*e[33]+e[22]*e[32]*e[17]+e[22]*e[14]*e[35]+e[22]*e[31]*e[16]+e[31]*e[21]*e[15]+e[31]*e[12]*e[24]+e[31]*e[23]*e[17]+e[31]*e[14]*e[26]-1.*e[16]*e[27]*e[18]+e[16]*e[33]*e[24]-1.*e[16]*e[30]*e[21]-1.*e[16]*e[29]*e[20]+e[16]*e[35]*e[26]-1.*e[16]*e[32]*e[23]+e[16]*e[28]*e[19]+3.*e[16]*e[34]*e[25]+e[19]*e[27]*e[15]+e[19]*e[9]*e[33]+e[19]*e[29]*e[17];
    A[45]=e[4]*e[27]*e[3]+e[4]*e[0]*e[30]+e[4]*e[29]*e[5]+e[4]*e[2]*e[32]+e[31]*e[0]*e[3]+e[31]*e[2]*e[5]+e[7]*e[27]*e[6]+e[7]*e[0]*e[33]+e[7]*e[29]*e[8]+e[7]*e[2]*e[35]+e[34]*e[0]*e[6]+e[34]*e[2]*e[8]+e[1]*e[27]*e[0]+e[1]*e[29]*e[2]+e[1]*e[34]*e[7]-1.*e[1]*e[32]*e[5]-1.*e[1]*e[33]*e[6]-1.*e[1]*e[30]*e[3]-1.*e[1]*e[35]*e[8]+e[1]*e[31]*e[4]+1.500000000*e[28]*ep2[1]+.5000000000*e[28]*ep2[4]+.5000000000*e[28]*ep2[0]-.5000000000*e[28]*ep2[6]-.5000000000*e[28]*ep2[5]+.5000000000*e[28]*ep2[7]-.5000000000*e[28]*ep2[3]+.5000000000*e[28]*ep2[2]-.5000000000*e[28]*ep2[8];
    A[191]=-1.*e[35]*e[10]*e[1]-1.*e[35]*e[13]*e[4]+e[35]*e[16]*e[7]+e[35]*e[15]*e[6]-1.*e[35]*e[9]*e[0]-1.*e[35]*e[12]*e[3]+e[32]*e[12]*e[6]+e[32]*e[3]*e[15]+e[32]*e[13]*e[7]+e[32]*e[4]*e[16]-1.*e[8]*e[27]*e[9]-1.*e[8]*e[30]*e[12]-1.*e[8]*e[28]*e[10]-1.*e[8]*e[31]*e[13]+e[8]*e[29]*e[11]+e[11]*e[27]*e[6]+e[11]*e[0]*e[33]+e[11]*e[28]*e[7]+e[11]*e[1]*e[34]+e[29]*e[9]*e[6]+e[29]*e[0]*e[15]+e[29]*e[10]*e[7]+e[29]*e[1]*e[16]+e[5]*e[30]*e[15]+e[5]*e[12]*e[33]+e[5]*e[32]*e[17]+e[5]*e[14]*e[35]+e[5]*e[31]*e[16]+e[5]*e[13]*e[34]+e[8]*e[33]*e[15]+3.*e[8]*e[35]*e[17]+e[8]*e[34]*e[16]+e[2]*e[27]*e[15]+e[2]*e[9]*e[33]+e[2]*e[29]*e[17]+e[2]*e[11]*e[35]+e[2]*e[28]*e[16]+e[2]*e[10]*e[34]-1.*e[17]*e[27]*e[0]+e[17]*e[34]*e[7]+e[17]*e[33]*e[6]-1.*e[17]*e[30]*e[3]-1.*e[17]*e[28]*e[1]-1.*e[17]*e[31]*e[4]+e[14]*e[30]*e[6]+e[14]*e[3]*e[33]+e[14]*e[31]*e[7]+e[14]*e[4]*e[34]+e[14]*e[32]*e[8];
    A[44]=e[19]*e[11]*e[2]+e[4]*e[18]*e[12]+e[4]*e[9]*e[21]+e[4]*e[20]*e[14]+e[4]*e[11]*e[23]+e[4]*e[19]*e[13]+e[4]*e[10]*e[22]+e[7]*e[18]*e[15]+e[7]*e[9]*e[24]+e[7]*e[20]*e[17]+e[7]*e[11]*e[26]+e[7]*e[19]*e[16]+e[7]*e[10]*e[25]+e[1]*e[18]*e[9]+e[1]*e[20]*e[11]-1.*e[10]*e[21]*e[3]-1.*e[10]*e[26]*e[8]-1.*e[10]*e[23]*e[5]-1.*e[10]*e[24]*e[6]+e[13]*e[18]*e[3]+e[13]*e[0]*e[21]+e[13]*e[1]*e[22]+e[13]*e[20]*e[5]+e[13]*e[2]*e[23]-1.*e[19]*e[15]*e[6]-1.*e[19]*e[14]*e[5]-1.*e[19]*e[12]*e[3]-1.*e[19]*e[17]*e[8]+e[22]*e[9]*e[3]+e[22]*e[0]*e[12]+e[22]*e[11]*e[5]+e[22]*e[2]*e[14]+e[16]*e[18]*e[6]+e[16]*e[0]*e[24]+e[16]*e[1]*e[25]+e[16]*e[20]*e[8]+e[16]*e[2]*e[26]-1.*e[1]*e[23]*e[14]-1.*e[1]*e[24]*e[15]-1.*e[1]*e[26]*e[17]-1.*e[1]*e[21]*e[12]+e[25]*e[9]*e[6]+e[25]*e[0]*e[15]+e[25]*e[11]*e[8]+e[25]*e[2]*e[17]+e[10]*e[18]*e[0]+3.*e[10]*e[19]*e[1]+e[10]*e[20]*e[2]+e[19]*e[9]*e[0];
    A[190]=.5000000000*ep2[23]*e[26]+.5000000000*e[26]*ep2[25]+.5000000000*ep2[20]*e[26]-.5000000000*e[26]*ep2[18]+.5000000000*ep3[26]+.5000000000*e[26]*ep2[24]+e[20]*e[19]*e[25]-.5000000000*e[26]*ep2[19]-.5000000000*e[26]*ep2[21]+e[20]*e[18]*e[24]-.5000000000*e[26]*ep2[22]+e[23]*e[21]*e[24]+e[23]*e[22]*e[25];
    A[47]=e[16]*e[9]*e[33]+e[16]*e[29]*e[17]+e[16]*e[11]*e[35]+e[16]*e[10]*e[34]+e[34]*e[11]*e[17]+e[34]*e[9]*e[15]-1.*e[10]*e[30]*e[12]-1.*e[10]*e[32]*e[14]-1.*e[10]*e[33]*e[15]-1.*e[10]*e[35]*e[17]+e[10]*e[27]*e[9]+e[10]*e[29]*e[11]+e[13]*e[27]*e[12]+e[13]*e[9]*e[30]+e[13]*e[29]*e[14]+e[13]*e[11]*e[32]+e[13]*e[10]*e[31]+e[31]*e[11]*e[14]+e[31]*e[9]*e[12]+e[16]*e[27]*e[15]+1.500000000*e[28]*ep2[10]+.5000000000*e[28]*ep2[16]+.5000000000*e[28]*ep2[9]+.5000000000*e[28]*ep2[11]-.5000000000*e[28]*ep2[12]-.5000000000*e[28]*ep2[15]-.5000000000*e[28]*ep2[17]-.5000000000*e[28]*ep2[14]+.5000000000*e[28]*ep2[13];
    A[189]=.5000000000*ep2[20]*e[35]+.5000000000*ep2[23]*e[35]+1.500000000*e[35]*ep2[26]+.5000000000*e[35]*ep2[25]+.5000000000*e[35]*ep2[24]-.5000000000*e[35]*ep2[18]-.5000000000*e[35]*ep2[19]-.5000000000*e[35]*ep2[22]-.5000000000*e[35]*ep2[21]+e[20]*e[27]*e[24]+e[20]*e[18]*e[33]+e[20]*e[28]*e[25]+e[20]*e[19]*e[34]+e[20]*e[29]*e[26]+e[29]*e[19]*e[25]+e[29]*e[18]*e[24]+e[23]*e[30]*e[24]+e[23]*e[21]*e[33]+e[23]*e[31]*e[25]+e[23]*e[22]*e[34]+e[23]*e[32]*e[26]+e[32]*e[22]*e[25]+e[32]*e[21]*e[24]+e[26]*e[33]*e[24]+e[26]*e[34]*e[25]-1.*e[26]*e[27]*e[18]-1.*e[26]*e[30]*e[21]-1.*e[26]*e[31]*e[22]-1.*e[26]*e[28]*e[19];
    A[46]=e[4]*e[2]*e[5]+.5000000000*e[1]*ep2[0]-.5000000000*e[1]*ep2[6]+e[7]*e[0]*e[6]+.5000000000*e[1]*ep2[7]+.5000000000*e[1]*ep2[4]-.5000000000*e[1]*ep2[8]+.5000000000*e[1]*ep2[2]-.5000000000*e[1]*ep2[3]+.5000000000*ep3[1]+e[7]*e[2]*e[8]-.5000000000*e[1]*ep2[5]+e[4]*e[0]*e[3];
    A[188]=-.5000000000*e[17]*ep2[13]-.5000000000*e[17]*ep2[9]+.5000000000*e[17]*ep2[16]+.5000000000*e[17]*ep2[15]+.5000000000*ep3[17]-.5000000000*e[17]*ep2[10]+e[14]*e[13]*e[16]+e[14]*e[12]*e[15]+.5000000000*ep2[14]*e[17]+e[11]*e[10]*e[16]-.5000000000*e[17]*ep2[12]+.5000000000*ep2[11]*e[17]+e[11]*e[9]*e[15];
    A[41]=e[4]*e[27]*e[30]+e[4]*e[29]*e[32]+e[4]*e[28]*e[31]+e[31]*e[27]*e[3]+e[31]*e[0]*e[30]+e[31]*e[29]*e[5]+e[31]*e[2]*e[32]+e[7]*e[27]*e[33]+e[7]*e[29]*e[35]+e[7]*e[28]*e[34]+e[28]*e[27]*e[0]+e[28]*e[29]*e[2]+e[34]*e[27]*e[6]+e[34]*e[0]*e[33]+e[34]*e[29]*e[8]+e[34]*e[2]*e[35]-1.*e[28]*e[32]*e[5]-1.*e[28]*e[33]*e[6]-1.*e[28]*e[30]*e[3]-1.*e[28]*e[35]*e[8]+.5000000000*e[1]*ep2[27]+.5000000000*e[1]*ep2[29]+1.500000000*e[1]*ep2[28]+.5000000000*e[1]*ep2[31]-.5000000000*e[1]*ep2[32]-.5000000000*e[1]*ep2[33]-.5000000000*e[1]*ep2[30]+.5000000000*e[1]*ep2[34]-.5000000000*e[1]*ep2[35];
    A[187]=.5000000000*ep2[11]*e[35]+.5000000000*e[35]*ep2[16]-.5000000000*e[35]*ep2[9]-.5000000000*e[35]*ep2[12]+.5000000000*e[35]*ep2[15]+1.500000000*e[35]*ep2[17]-.5000000000*e[35]*ep2[10]+.5000000000*e[35]*ep2[14]-.5000000000*e[35]*ep2[13]+e[11]*e[27]*e[15]+e[11]*e[9]*e[33]+e[11]*e[29]*e[17]+e[11]*e[28]*e[16]+e[11]*e[10]*e[34]+e[29]*e[9]*e[15]+e[29]*e[10]*e[16]+e[14]*e[30]*e[15]+e[14]*e[12]*e[33]+e[14]*e[32]*e[17]+e[14]*e[31]*e[16]+e[14]*e[13]*e[34]+e[32]*e[12]*e[15]+e[32]*e[13]*e[16]+e[17]*e[33]*e[15]+e[17]*e[34]*e[16]-1.*e[17]*e[27]*e[9]-1.*e[17]*e[30]*e[12]-1.*e[17]*e[28]*e[10]-1.*e[17]*e[31]*e[13];
    A[40]=e[34]*e[27]*e[33]+e[34]*e[29]*e[35]-.5000000000*e[28]*ep2[30]-.5000000000*e[28]*ep2[35]+.5000000000*ep3[28]+.5000000000*e[28]*ep2[27]+.5000000000*e[28]*ep2[29]+e[31]*e[27]*e[30]+e[31]*e[29]*e[32]-.5000000000*e[28]*ep2[32]-.5000000000*e[28]*ep2[33]+.5000000000*e[28]*ep2[31]+.5000000000*e[28]*ep2[34];
    A[186]=.5000000000*ep2[5]*e[8]+e[2]*e[0]*e[6]+.5000000000*ep2[2]*e[8]+.5000000000*ep3[8]-.5000000000*e[8]*ep2[0]+e[5]*e[4]*e[7]+e[5]*e[3]*e[6]+.5000000000*e[8]*ep2[7]+e[2]*e[1]*e[7]-.5000000000*e[8]*ep2[1]-.5000000000*e[8]*ep2[4]-.5000000000*e[8]*ep2[3]+.5000000000*e[8]*ep2[6];
    A[43]=e[28]*e[27]*e[9]+e[28]*e[29]*e[11]-1.*e[28]*e[30]*e[12]+e[28]*e[31]*e[13]-1.*e[28]*e[32]*e[14]-1.*e[28]*e[33]*e[15]-1.*e[28]*e[35]*e[17]+e[31]*e[27]*e[12]+e[31]*e[9]*e[30]+e[31]*e[29]*e[14]+e[31]*e[11]*e[32]+e[13]*e[27]*e[30]+e[13]*e[29]*e[32]+e[16]*e[27]*e[33]+e[16]*e[29]*e[35]+e[34]*e[27]*e[15]+e[34]*e[9]*e[33]+e[34]*e[29]*e[17]+e[34]*e[11]*e[35]+e[34]*e[28]*e[16]+.5000000000*e[10]*ep2[27]+.5000000000*e[10]*ep2[29]+1.500000000*e[10]*ep2[28]-.5000000000*e[10]*ep2[32]+.5000000000*e[10]*ep2[31]-.5000000000*e[10]*ep2[33]-.5000000000*e[10]*ep2[30]+.5000000000*e[10]*ep2[34]-.5000000000*e[10]*ep2[35];
    A[185]=-.5000000000*e[35]*ep2[1]+.5000000000*e[35]*ep2[7]-.5000000000*e[35]*ep2[3]+.5000000000*ep2[2]*e[35]+1.500000000*e[35]*ep2[8]-.5000000000*e[35]*ep2[4]-.5000000000*e[35]*ep2[0]+.5000000000*e[35]*ep2[6]+.5000000000*e[35]*ep2[5]+e[2]*e[27]*e[6]+e[2]*e[0]*e[33]+e[2]*e[28]*e[7]+e[2]*e[1]*e[34]+e[2]*e[29]*e[8]-1.*e[8]*e[27]*e[0]+e[8]*e[34]*e[7]+e[8]*e[32]*e[5]+e[8]*e[33]*e[6]-1.*e[8]*e[30]*e[3]-1.*e[8]*e[28]*e[1]-1.*e[8]*e[31]*e[4]+e[29]*e[1]*e[7]+e[29]*e[0]*e[6]+e[5]*e[30]*e[6]+e[5]*e[3]*e[33]+e[5]*e[31]*e[7]+e[5]*e[4]*e[34]+e[32]*e[4]*e[7]+e[32]*e[3]*e[6];
    A[42]=e[28]*e[27]*e[18]+e[28]*e[29]*e[20]+e[22]*e[27]*e[30]+e[22]*e[29]*e[32]+e[22]*e[28]*e[31]+e[31]*e[27]*e[21]+e[31]*e[18]*e[30]+e[31]*e[29]*e[23]+e[31]*e[20]*e[32]+e[25]*e[27]*e[33]+e[25]*e[29]*e[35]+e[25]*e[28]*e[34]+e[34]*e[27]*e[24]+e[34]*e[18]*e[33]+e[34]*e[29]*e[26]+e[34]*e[20]*e[35]-1.*e[28]*e[33]*e[24]-1.*e[28]*e[30]*e[21]-1.*e[28]*e[35]*e[26]-1.*e[28]*e[32]*e[23]-.5000000000*e[19]*ep2[33]-.5000000000*e[19]*ep2[30]-.5000000000*e[19]*ep2[35]+.5000000000*e[19]*ep2[27]+.5000000000*e[19]*ep2[29]+1.500000000*e[19]*ep2[28]+.5000000000*e[19]*ep2[31]+.5000000000*e[19]*ep2[34]-.5000000000*e[19]*ep2[32];
    A[184]=e[23]*e[3]*e[15]-1.*e[17]*e[19]*e[1]-1.*e[17]*e[22]*e[4]-1.*e[17]*e[18]*e[0]+e[17]*e[25]*e[7]+e[17]*e[24]*e[6]+e[14]*e[21]*e[6]+e[14]*e[3]*e[24]+e[14]*e[22]*e[7]+e[14]*e[4]*e[25]+e[14]*e[23]*e[8]-1.*e[26]*e[10]*e[1]-1.*e[26]*e[13]*e[4]+e[26]*e[16]*e[7]+e[26]*e[15]*e[6]-1.*e[26]*e[9]*e[0]-1.*e[26]*e[12]*e[3]+e[23]*e[12]*e[6]+e[11]*e[18]*e[6]+e[11]*e[0]*e[24]+e[11]*e[19]*e[7]+e[11]*e[1]*e[25]+e[11]*e[20]*e[8]+e[11]*e[2]*e[26]+e[20]*e[9]*e[6]+e[20]*e[0]*e[15]+e[20]*e[10]*e[7]+e[20]*e[1]*e[16]+e[20]*e[2]*e[17]+e[5]*e[21]*e[15]+e[5]*e[12]*e[24]+e[5]*e[23]*e[17]+e[5]*e[14]*e[26]+e[5]*e[22]*e[16]+e[5]*e[13]*e[25]+e[8]*e[24]*e[15]+3.*e[8]*e[26]*e[17]+e[8]*e[25]*e[16]+e[2]*e[18]*e[15]+e[2]*e[9]*e[24]+e[2]*e[19]*e[16]+e[2]*e[10]*e[25]-1.*e[17]*e[21]*e[3]+e[23]*e[4]*e[16]+e[23]*e[13]*e[7]-1.*e[8]*e[18]*e[9]-1.*e[8]*e[21]*e[12]-1.*e[8]*e[19]*e[10]-1.*e[8]*e[22]*e[13];
    A[54]=e[13]*e[18]*e[12]+e[13]*e[9]*e[21]+e[13]*e[20]*e[14]+e[13]*e[11]*e[23]+e[13]*e[10]*e[22]+e[22]*e[11]*e[14]+e[22]*e[9]*e[12]+e[16]*e[18]*e[15]+e[16]*e[9]*e[24]+e[16]*e[20]*e[17]+e[16]*e[11]*e[26]+e[16]*e[10]*e[25]+e[25]*e[11]*e[17]+e[25]*e[9]*e[15]-1.*e[10]*e[23]*e[14]-1.*e[10]*e[24]*e[15]-1.*e[10]*e[26]*e[17]+e[10]*e[20]*e[11]+e[10]*e[18]*e[9]-1.*e[10]*e[21]*e[12]+.5000000000*e[19]*ep2[11]+.5000000000*e[19]*ep2[9]+1.500000000*e[19]*ep2[10]+.5000000000*e[19]*ep2[13]+.5000000000*e[19]*ep2[16]-.5000000000*e[19]*ep2[12]-.5000000000*e[19]*ep2[15]-.5000000000*e[19]*ep2[17]-.5000000000*e[19]*ep2[14];
    A[164]=e[10]*e[18]*e[6]+e[10]*e[0]*e[24]+e[10]*e[19]*e[7]+e[10]*e[1]*e[25]+e[10]*e[20]*e[8]+e[10]*e[2]*e[26]+e[19]*e[9]*e[6]+e[19]*e[0]*e[15]+e[19]*e[1]*e[16]+e[19]*e[11]*e[8]+e[19]*e[2]*e[17]+e[4]*e[21]*e[15]+e[4]*e[12]*e[24]+e[4]*e[23]*e[17]+e[4]*e[14]*e[26]+e[4]*e[22]*e[16]+e[4]*e[13]*e[25]+e[7]*e[24]*e[15]+e[7]*e[26]*e[17]+3.*e[7]*e[25]*e[16]+e[1]*e[18]*e[15]+e[1]*e[9]*e[24]+e[1]*e[20]*e[17]+e[1]*e[11]*e[26]-1.*e[16]*e[21]*e[3]+e[16]*e[26]*e[8]-1.*e[16]*e[20]*e[2]-1.*e[16]*e[18]*e[0]-1.*e[16]*e[23]*e[5]+e[16]*e[24]*e[6]+e[13]*e[21]*e[6]+e[13]*e[3]*e[24]+e[13]*e[22]*e[7]+e[13]*e[23]*e[8]+e[13]*e[5]*e[26]-1.*e[25]*e[11]*e[2]+e[25]*e[15]*e[6]-1.*e[25]*e[9]*e[0]-1.*e[25]*e[14]*e[5]-1.*e[25]*e[12]*e[3]+e[25]*e[17]*e[8]+e[22]*e[12]*e[6]+e[22]*e[3]*e[15]+e[22]*e[14]*e[8]+e[22]*e[5]*e[17]-1.*e[7]*e[23]*e[14]-1.*e[7]*e[20]*e[11]-1.*e[7]*e[18]*e[9]-1.*e[7]*e[21]*e[12];
    A[55]=e[13]*e[9]*e[3]+e[13]*e[0]*e[12]+e[13]*e[10]*e[4]+e[13]*e[11]*e[5]+e[13]*e[2]*e[14]+e[16]*e[9]*e[6]+e[16]*e[0]*e[15]+e[16]*e[10]*e[7]+e[16]*e[11]*e[8]+e[16]*e[2]*e[17]+e[7]*e[11]*e[17]+e[7]*e[9]*e[15]+e[4]*e[11]*e[14]+e[4]*e[9]*e[12]+e[10]*e[9]*e[0]+e[10]*e[11]*e[2]-1.*e[10]*e[15]*e[6]-1.*e[10]*e[14]*e[5]-1.*e[10]*e[12]*e[3]-1.*e[10]*e[17]*e[8]+.5000000000*e[1]*ep2[11]+.5000000000*e[1]*ep2[9]+1.500000000*e[1]*ep2[10]-.5000000000*e[1]*ep2[12]-.5000000000*e[1]*ep2[15]-.5000000000*e[1]*ep2[17]-.5000000000*e[1]*ep2[14]+.5000000000*e[1]*ep2[13]+.5000000000*e[1]*ep2[16];
    A[165]=e[1]*e[27]*e[6]+e[1]*e[0]*e[33]+e[1]*e[28]*e[7]+e[1]*e[29]*e[8]+e[1]*e[2]*e[35]-1.*e[7]*e[27]*e[0]-1.*e[7]*e[32]*e[5]+e[7]*e[33]*e[6]-1.*e[7]*e[30]*e[3]+e[7]*e[35]*e[8]-1.*e[7]*e[29]*e[2]+e[7]*e[31]*e[4]+e[28]*e[0]*e[6]+e[28]*e[2]*e[8]+e[4]*e[30]*e[6]+e[4]*e[3]*e[33]+e[4]*e[32]*e[8]+e[4]*e[5]*e[35]+e[31]*e[3]*e[6]+e[31]*e[5]*e[8]+.5000000000*ep2[1]*e[34]+1.500000000*e[34]*ep2[7]+.5000000000*e[34]*ep2[4]-.5000000000*e[34]*ep2[0]+.5000000000*e[34]*ep2[6]-.5000000000*e[34]*ep2[5]-.5000000000*e[34]*ep2[3]-.5000000000*e[34]*ep2[2]+.5000000000*e[34]*ep2[8];
    A[52]=e[4]*e[18]*e[3]+e[4]*e[0]*e[21]+e[4]*e[1]*e[22]+e[4]*e[20]*e[5]+e[4]*e[2]*e[23]+e[22]*e[0]*e[3]+e[22]*e[2]*e[5]+e[7]*e[18]*e[6]+e[7]*e[0]*e[24]+e[7]*e[1]*e[25]+e[7]*e[20]*e[8]+e[7]*e[2]*e[26]+e[25]*e[0]*e[6]+e[25]*e[2]*e[8]+e[1]*e[18]*e[0]+e[1]*e[20]*e[2]-1.*e[1]*e[21]*e[3]-1.*e[1]*e[26]*e[8]-1.*e[1]*e[23]*e[5]-1.*e[1]*e[24]*e[6]+.5000000000*e[19]*ep2[4]+.5000000000*e[19]*ep2[0]-.5000000000*e[19]*ep2[6]-.5000000000*e[19]*ep2[5]+1.500000000*e[19]*ep2[1]+.5000000000*e[19]*ep2[7]-.5000000000*e[19]*ep2[3]+.5000000000*e[19]*ep2[2]-.5000000000*e[19]*ep2[8];
    A[166]=-.5000000000*e[7]*ep2[0]+e[4]*e[5]*e[8]+.5000000000*ep2[4]*e[7]-.5000000000*e[7]*ep2[2]+.5000000000*e[7]*ep2[8]-.5000000000*e[7]*ep2[5]+.5000000000*e[7]*ep2[6]+e[1]*e[0]*e[6]+.5000000000*ep3[7]+e[4]*e[3]*e[6]+e[1]*e[2]*e[8]-.5000000000*e[7]*ep2[3]+.5000000000*ep2[1]*e[7];
    A[53]=-1.*e[1]*e[32]*e[23]-1.*e[19]*e[32]*e[5]-1.*e[19]*e[33]*e[6]-1.*e[19]*e[30]*e[3]-1.*e[19]*e[35]*e[8]-1.*e[28]*e[21]*e[3]-1.*e[28]*e[26]*e[8]-1.*e[28]*e[23]*e[5]-1.*e[28]*e[24]*e[6]+e[7]*e[27]*e[24]+e[7]*e[18]*e[33]+e[7]*e[29]*e[26]+e[7]*e[20]*e[35]+e[22]*e[27]*e[3]+e[22]*e[0]*e[30]+e[22]*e[29]*e[5]+e[22]*e[2]*e[32]+e[31]*e[18]*e[3]+e[31]*e[0]*e[21]+e[31]*e[20]*e[5]+e[31]*e[2]*e[23]+e[25]*e[27]*e[6]+e[25]*e[0]*e[33]+e[25]*e[28]*e[7]+e[25]*e[1]*e[34]+e[25]*e[29]*e[8]+e[25]*e[2]*e[35]+e[34]*e[18]*e[6]+e[34]*e[0]*e[24]+e[34]*e[19]*e[7]+e[34]*e[20]*e[8]+e[34]*e[2]*e[26]+e[1]*e[27]*e[18]+3.*e[1]*e[28]*e[19]+e[1]*e[29]*e[20]+e[19]*e[27]*e[0]+e[19]*e[29]*e[2]+e[28]*e[18]*e[0]+e[28]*e[20]*e[2]+e[4]*e[27]*e[21]+e[4]*e[18]*e[30]+e[4]*e[28]*e[22]+e[4]*e[19]*e[31]+e[4]*e[29]*e[23]+e[4]*e[20]*e[32]-1.*e[1]*e[33]*e[24]-1.*e[1]*e[30]*e[21]-1.*e[1]*e[35]*e[26]+e[1]*e[31]*e[22];
    A[167]=e[10]*e[27]*e[15]+e[10]*e[9]*e[33]+e[10]*e[29]*e[17]+e[10]*e[11]*e[35]+e[10]*e[28]*e[16]+e[28]*e[11]*e[17]+e[28]*e[9]*e[15]+e[13]*e[30]*e[15]+e[13]*e[12]*e[33]+e[13]*e[32]*e[17]+e[13]*e[14]*e[35]+e[13]*e[31]*e[16]+e[31]*e[14]*e[17]+e[31]*e[12]*e[15]+e[16]*e[33]*e[15]+e[16]*e[35]*e[17]-1.*e[16]*e[27]*e[9]-1.*e[16]*e[30]*e[12]-1.*e[16]*e[32]*e[14]-1.*e[16]*e[29]*e[11]+.5000000000*ep2[10]*e[34]+1.500000000*e[34]*ep2[16]-.5000000000*e[34]*ep2[9]-.5000000000*e[34]*ep2[11]-.5000000000*e[34]*ep2[12]+.5000000000*e[34]*ep2[15]+.5000000000*e[34]*ep2[17]-.5000000000*e[34]*ep2[14]+.5000000000*e[34]*ep2[13];
    A[50]=.5000000000*e[19]*ep2[18]+.5000000000*e[19]*ep2[25]+.5000000000*e[19]*ep2[22]+e[25]*e[20]*e[26]-.5000000000*e[19]*ep2[21]+.5000000000*e[19]*ep2[20]-.5000000000*e[19]*ep2[26]-.5000000000*e[19]*ep2[23]-.5000000000*e[19]*ep2[24]+.5000000000*ep3[19]+e[22]*e[20]*e[23]+e[25]*e[18]*e[24]+e[22]*e[18]*e[21];
    A[160]=.5000000000*e[34]*ep2[33]+.5000000000*e[34]*ep2[35]-.5000000000*e[34]*ep2[27]-.5000000000*e[34]*ep2[32]-.5000000000*e[34]*ep2[29]-.5000000000*e[34]*ep2[30]+.5000000000*ep2[28]*e[34]+e[31]*e[30]*e[33]+e[31]*e[32]*e[35]+e[28]*e[27]*e[33]+.5000000000*ep3[34]+e[28]*e[29]*e[35]+.5000000000*ep2[31]*e[34];
    A[51]=e[4]*e[28]*e[13]+e[4]*e[10]*e[31]+e[7]*e[27]*e[15]+e[7]*e[9]*e[33]+e[7]*e[29]*e[17]+e[7]*e[11]*e[35]+e[7]*e[28]*e[16]+e[7]*e[10]*e[34]+e[1]*e[27]*e[9]+e[1]*e[29]*e[11]+3.*e[1]*e[28]*e[10]+e[10]*e[27]*e[0]-1.*e[10]*e[32]*e[5]-1.*e[10]*e[33]*e[6]-1.*e[10]*e[30]*e[3]-1.*e[10]*e[35]*e[8]+e[10]*e[29]*e[2]+e[13]*e[27]*e[3]+e[13]*e[0]*e[30]+e[13]*e[1]*e[31]+e[13]*e[29]*e[5]+e[13]*e[2]*e[32]+e[28]*e[11]*e[2]-1.*e[28]*e[15]*e[6]+e[28]*e[9]*e[0]-1.*e[28]*e[14]*e[5]-1.*e[28]*e[12]*e[3]-1.*e[28]*e[17]*e[8]+e[31]*e[9]*e[3]+e[31]*e[0]*e[12]+e[31]*e[11]*e[5]+e[31]*e[2]*e[14]+e[16]*e[27]*e[6]+e[16]*e[0]*e[33]+e[16]*e[1]*e[34]+e[16]*e[29]*e[8]+e[16]*e[2]*e[35]-1.*e[1]*e[30]*e[12]-1.*e[1]*e[32]*e[14]-1.*e[1]*e[33]*e[15]-1.*e[1]*e[35]*e[17]+e[34]*e[9]*e[6]+e[34]*e[0]*e[15]+e[34]*e[11]*e[8]+e[34]*e[2]*e[17]+e[4]*e[27]*e[12]+e[4]*e[9]*e[30]+e[4]*e[29]*e[14]+e[4]*e[11]*e[32];
    A[161]=e[4]*e[30]*e[33]+e[4]*e[32]*e[35]+e[4]*e[31]*e[34]+e[31]*e[30]*e[6]+e[31]*e[3]*e[33]+e[31]*e[32]*e[8]+e[31]*e[5]*e[35]+e[28]*e[27]*e[6]+e[28]*e[0]*e[33]+e[28]*e[29]*e[8]+e[28]*e[2]*e[35]+e[34]*e[33]*e[6]+e[34]*e[35]*e[8]-1.*e[34]*e[27]*e[0]-1.*e[34]*e[32]*e[5]-1.*e[34]*e[30]*e[3]-1.*e[34]*e[29]*e[2]+e[1]*e[27]*e[33]+e[1]*e[29]*e[35]+e[1]*e[28]*e[34]+.5000000000*ep2[31]*e[7]-.5000000000*e[7]*ep2[27]-.5000000000*e[7]*ep2[32]+.5000000000*e[7]*ep2[28]-.5000000000*e[7]*ep2[29]+.5000000000*e[7]*ep2[33]-.5000000000*e[7]*ep2[30]+1.500000000*e[7]*ep2[34]+.5000000000*e[7]*ep2[35];
    A[48]=-.5000000000*e[10]*ep2[14]-.5000000000*e[10]*ep2[17]-.5000000000*e[10]*ep2[15]+e[13]*e[11]*e[14]+e[16]*e[11]*e[17]+.5000000000*e[10]*ep2[13]+e[13]*e[9]*e[12]-.5000000000*e[10]*ep2[12]+.5000000000*ep3[10]+e[16]*e[9]*e[15]+.5000000000*e[10]*ep2[16]+.5000000000*e[10]*ep2[11]+.5000000000*e[10]*ep2[9];
    A[162]=e[22]*e[32]*e[35]+e[22]*e[31]*e[34]+e[31]*e[30]*e[24]+e[31]*e[21]*e[33]+e[31]*e[32]*e[26]+e[31]*e[23]*e[35]+e[34]*e[33]*e[24]+e[34]*e[35]*e[26]-1.*e[34]*e[27]*e[18]-1.*e[34]*e[30]*e[21]-1.*e[34]*e[29]*e[20]-1.*e[34]*e[32]*e[23]+e[19]*e[27]*e[33]+e[19]*e[29]*e[35]+e[19]*e[28]*e[34]+e[28]*e[27]*e[24]+e[28]*e[18]*e[33]+e[28]*e[29]*e[26]+e[28]*e[20]*e[35]+e[22]*e[30]*e[33]+.5000000000*ep2[28]*e[25]+.5000000000*ep2[31]*e[25]+.5000000000*e[25]*ep2[33]+.5000000000*e[25]*ep2[35]+1.500000000*e[25]*ep2[34]-.5000000000*e[25]*ep2[27]-.5000000000*e[25]*ep2[32]-.5000000000*e[25]*ep2[29]-.5000000000*e[25]*ep2[30];
    A[49]=-1.*e[19]*e[35]*e[26]-1.*e[19]*e[32]*e[23]+e[19]*e[27]*e[18]+e[19]*e[29]*e[20]+e[22]*e[27]*e[21]+e[22]*e[18]*e[30]+e[22]*e[19]*e[31]+e[22]*e[29]*e[23]+e[22]*e[20]*e[32]+e[31]*e[18]*e[21]+e[31]*e[20]*e[23]+e[25]*e[27]*e[24]+e[25]*e[18]*e[33]+e[25]*e[19]*e[34]+e[25]*e[29]*e[26]+e[25]*e[20]*e[35]+e[34]*e[18]*e[24]+e[34]*e[20]*e[26]-1.*e[19]*e[33]*e[24]-1.*e[19]*e[30]*e[21]+1.500000000*e[28]*ep2[19]+.5000000000*e[28]*ep2[18]+.5000000000*e[28]*ep2[20]+.5000000000*e[28]*ep2[22]+.5000000000*e[28]*ep2[25]-.5000000000*e[28]*ep2[26]-.5000000000*e[28]*ep2[23]-.5000000000*e[28]*ep2[24]-.5000000000*e[28]*ep2[21];
    A[163]=e[10]*e[27]*e[33]+e[10]*e[29]*e[35]+e[10]*e[28]*e[34]+e[34]*e[33]*e[15]+e[34]*e[35]*e[17]+e[28]*e[27]*e[15]+e[28]*e[9]*e[33]+e[28]*e[29]*e[17]+e[28]*e[11]*e[35]-1.*e[34]*e[27]*e[9]-1.*e[34]*e[30]*e[12]+e[34]*e[31]*e[13]-1.*e[34]*e[32]*e[14]-1.*e[34]*e[29]*e[11]+e[31]*e[30]*e[15]+e[31]*e[12]*e[33]+e[31]*e[32]*e[17]+e[31]*e[14]*e[35]+e[13]*e[30]*e[33]+e[13]*e[32]*e[35]-.5000000000*e[16]*ep2[27]-.5000000000*e[16]*ep2[32]+.5000000000*e[16]*ep2[28]-.5000000000*e[16]*ep2[29]+.5000000000*e[16]*ep2[31]+.5000000000*e[16]*ep2[33]-.5000000000*e[16]*ep2[30]+1.500000000*e[16]*ep2[34]+.5000000000*e[16]*ep2[35];
    A[63]=e[29]*e[32]*e[14]-1.*e[29]*e[33]*e[15]-1.*e[29]*e[34]*e[16]+e[32]*e[27]*e[12]+e[32]*e[9]*e[30]+e[32]*e[28]*e[13]+e[32]*e[10]*e[31]+e[14]*e[27]*e[30]+e[14]*e[28]*e[31]+e[17]*e[27]*e[33]+e[17]*e[28]*e[34]+e[35]*e[27]*e[15]+e[35]*e[9]*e[33]+e[35]*e[29]*e[17]+e[35]*e[28]*e[16]+e[35]*e[10]*e[34]+e[29]*e[27]*e[9]+e[29]*e[28]*e[10]-1.*e[29]*e[30]*e[12]-1.*e[29]*e[31]*e[13]+.5000000000*e[11]*ep2[27]+1.500000000*e[11]*ep2[29]+.5000000000*e[11]*ep2[28]+.5000000000*e[11]*ep2[32]-.5000000000*e[11]*ep2[31]-.5000000000*e[11]*ep2[33]-.5000000000*e[11]*ep2[30]-.5000000000*e[11]*ep2[34]+.5000000000*e[11]*ep2[35];
    A[173]=e[1]*e[20]*e[35]+e[19]*e[27]*e[6]+e[19]*e[0]*e[33]+e[19]*e[28]*e[7]+e[19]*e[29]*e[8]+e[19]*e[2]*e[35]+e[28]*e[18]*e[6]+e[28]*e[0]*e[24]+e[28]*e[20]*e[8]+e[28]*e[2]*e[26]+e[4]*e[30]*e[24]+e[4]*e[21]*e[33]+e[4]*e[31]*e[25]+e[4]*e[22]*e[34]+e[4]*e[32]*e[26]+e[4]*e[23]*e[35]-1.*e[7]*e[27]*e[18]+e[7]*e[33]*e[24]-1.*e[7]*e[30]*e[21]-1.*e[7]*e[29]*e[20]+e[7]*e[35]*e[26]+e[7]*e[31]*e[22]-1.*e[7]*e[32]*e[23]-1.*e[25]*e[27]*e[0]-1.*e[25]*e[32]*e[5]-1.*e[25]*e[30]*e[3]-1.*e[25]*e[29]*e[2]-1.*e[34]*e[21]*e[3]-1.*e[34]*e[20]*e[2]-1.*e[34]*e[18]*e[0]-1.*e[34]*e[23]*e[5]+e[22]*e[30]*e[6]+e[22]*e[3]*e[33]+e[22]*e[32]*e[8]+e[22]*e[5]*e[35]+e[31]*e[21]*e[6]+e[31]*e[3]*e[24]+e[31]*e[23]*e[8]+e[31]*e[5]*e[26]+e[34]*e[26]*e[8]+e[1]*e[27]*e[24]+e[1]*e[18]*e[33]+e[1]*e[28]*e[25]+e[1]*e[19]*e[34]+e[1]*e[29]*e[26]+e[34]*e[24]*e[6]+e[25]*e[33]*e[6]+3.*e[25]*e[34]*e[7]+e[25]*e[35]*e[8];
    A[62]=.5000000000*e[20]*ep2[27]+1.500000000*e[20]*ep2[29]+.5000000000*e[20]*ep2[28]+.5000000000*e[20]*ep2[32]+.5000000000*e[20]*ep2[35]-.5000000000*e[20]*ep2[31]-.5000000000*e[20]*ep2[33]-.5000000000*e[20]*ep2[30]-.5000000000*e[20]*ep2[34]+e[29]*e[27]*e[18]+e[29]*e[28]*e[19]+e[23]*e[27]*e[30]+e[23]*e[29]*e[32]+e[23]*e[28]*e[31]+e[32]*e[27]*e[21]+e[32]*e[18]*e[30]+e[32]*e[28]*e[22]+e[32]*e[19]*e[31]+e[26]*e[27]*e[33]+e[26]*e[29]*e[35]+e[26]*e[28]*e[34]+e[35]*e[27]*e[24]+e[35]*e[18]*e[33]+e[35]*e[28]*e[25]+e[35]*e[19]*e[34]-1.*e[29]*e[33]*e[24]-1.*e[29]*e[30]*e[21]-1.*e[29]*e[31]*e[22]-1.*e[29]*e[34]*e[25];
    A[172]=e[19]*e[1]*e[7]+e[19]*e[0]*e[6]+e[19]*e[2]*e[8]+e[4]*e[21]*e[6]+e[4]*e[3]*e[24]+e[4]*e[22]*e[7]+e[4]*e[23]*e[8]+e[4]*e[5]*e[26]+e[22]*e[3]*e[6]+e[22]*e[5]*e[8]+e[7]*e[24]*e[6]+e[7]*e[26]*e[8]+e[1]*e[18]*e[6]+e[1]*e[0]*e[24]+e[1]*e[20]*e[8]+e[1]*e[2]*e[26]-1.*e[7]*e[21]*e[3]-1.*e[7]*e[20]*e[2]-1.*e[7]*e[18]*e[0]-1.*e[7]*e[23]*e[5]+.5000000000*e[25]*ep2[4]-.5000000000*e[25]*ep2[0]+.5000000000*e[25]*ep2[6]-.5000000000*e[25]*ep2[5]+.5000000000*e[25]*ep2[1]+1.500000000*e[25]*ep2[7]-.5000000000*e[25]*ep2[3]-.5000000000*e[25]*ep2[2]+.5000000000*e[25]*ep2[8];
    A[61]=e[5]*e[27]*e[30]+e[5]*e[29]*e[32]+e[5]*e[28]*e[31]+e[32]*e[27]*e[3]+e[32]*e[0]*e[30]+e[32]*e[28]*e[4]+e[32]*e[1]*e[31]+e[8]*e[27]*e[33]+e[8]*e[29]*e[35]+e[8]*e[28]*e[34]+e[29]*e[27]*e[0]+e[29]*e[28]*e[1]+e[35]*e[27]*e[6]+e[35]*e[0]*e[33]+e[35]*e[28]*e[7]+e[35]*e[1]*e[34]-1.*e[29]*e[34]*e[7]-1.*e[29]*e[33]*e[6]-1.*e[29]*e[30]*e[3]-1.*e[29]*e[31]*e[4]+.5000000000*e[2]*ep2[27]+1.500000000*e[2]*ep2[29]+.5000000000*e[2]*ep2[28]+.5000000000*e[2]*ep2[32]-.5000000000*e[2]*ep2[31]-.5000000000*e[2]*ep2[33]-.5000000000*e[2]*ep2[30]-.5000000000*e[2]*ep2[34]+.5000000000*e[2]*ep2[35];
    A[175]=e[13]*e[12]*e[6]+e[13]*e[3]*e[15]+e[13]*e[4]*e[16]+e[13]*e[14]*e[8]+e[13]*e[5]*e[17]+e[16]*e[15]*e[6]+e[16]*e[17]*e[8]+e[1]*e[11]*e[17]+e[1]*e[9]*e[15]+e[1]*e[10]*e[16]+e[4]*e[14]*e[17]+e[4]*e[12]*e[15]+e[10]*e[9]*e[6]+e[10]*e[0]*e[15]+e[10]*e[11]*e[8]+e[10]*e[2]*e[17]-1.*e[16]*e[11]*e[2]-1.*e[16]*e[9]*e[0]-1.*e[16]*e[14]*e[5]-1.*e[16]*e[12]*e[3]+.5000000000*ep2[13]*e[7]+1.500000000*ep2[16]*e[7]+.5000000000*e[7]*ep2[17]+.5000000000*e[7]*ep2[15]-.5000000000*e[7]*ep2[9]-.5000000000*e[7]*ep2[11]-.5000000000*e[7]*ep2[12]+.5000000000*e[7]*ep2[10]-.5000000000*e[7]*ep2[14];
    A[60]=.5000000000*e[29]*ep2[32]+.5000000000*e[29]*ep2[35]-.5000000000*e[29]*ep2[31]-.5000000000*e[29]*ep2[33]-.5000000000*e[29]*ep2[30]-.5000000000*e[29]*ep2[34]+e[32]*e[27]*e[30]+.5000000000*ep3[29]+.5000000000*e[29]*ep2[28]+e[35]*e[28]*e[34]+.5000000000*e[29]*ep2[27]+e[35]*e[27]*e[33]+e[32]*e[28]*e[31];
    A[174]=-1.*e[16]*e[21]*e[12]+e[10]*e[18]*e[15]+e[10]*e[9]*e[24]+e[10]*e[20]*e[17]+e[10]*e[11]*e[26]+e[19]*e[11]*e[17]+e[19]*e[9]*e[15]+e[19]*e[10]*e[16]+e[13]*e[21]*e[15]+e[13]*e[12]*e[24]+e[13]*e[23]*e[17]+e[13]*e[14]*e[26]+e[13]*e[22]*e[16]+e[22]*e[14]*e[17]+e[22]*e[12]*e[15]+e[16]*e[24]*e[15]+e[16]*e[26]*e[17]-1.*e[16]*e[23]*e[14]-1.*e[16]*e[20]*e[11]-1.*e[16]*e[18]*e[9]+.5000000000*ep2[13]*e[25]+1.500000000*e[25]*ep2[16]+.5000000000*e[25]*ep2[17]+.5000000000*e[25]*ep2[15]+.5000000000*ep2[10]*e[25]-.5000000000*e[25]*ep2[9]-.5000000000*e[25]*ep2[11]-.5000000000*e[25]*ep2[12]-.5000000000*e[25]*ep2[14];
    A[59]=e[19]*e[20]*e[2]+e[22]*e[18]*e[3]+e[22]*e[0]*e[21]+e[22]*e[19]*e[4]+e[22]*e[20]*e[5]+e[22]*e[2]*e[23]-1.*e[19]*e[21]*e[3]-1.*e[19]*e[26]*e[8]+e[19]*e[25]*e[7]-1.*e[19]*e[23]*e[5]-1.*e[19]*e[24]*e[6]+e[4]*e[18]*e[21]+e[4]*e[20]*e[23]+e[25]*e[18]*e[6]+e[25]*e[0]*e[24]+e[25]*e[20]*e[8]+e[25]*e[2]*e[26]+e[7]*e[18]*e[24]+e[7]*e[20]*e[26]+e[19]*e[18]*e[0]+1.500000000*ep2[19]*e[1]+.5000000000*e[1]*ep2[22]+.5000000000*e[1]*ep2[18]+.5000000000*e[1]*ep2[20]+.5000000000*e[1]*ep2[25]-.5000000000*e[1]*ep2[26]-.5000000000*e[1]*ep2[23]-.5000000000*e[1]*ep2[24]-.5000000000*e[1]*ep2[21];
    A[169]=e[19]*e[27]*e[24]+e[19]*e[18]*e[33]+e[19]*e[28]*e[25]+e[19]*e[29]*e[26]+e[19]*e[20]*e[35]+e[28]*e[18]*e[24]+e[28]*e[20]*e[26]+e[22]*e[30]*e[24]+e[22]*e[21]*e[33]+e[22]*e[31]*e[25]+e[22]*e[32]*e[26]+e[22]*e[23]*e[35]+e[31]*e[21]*e[24]+e[31]*e[23]*e[26]+e[25]*e[33]*e[24]+e[25]*e[35]*e[26]-1.*e[25]*e[27]*e[18]-1.*e[25]*e[30]*e[21]-1.*e[25]*e[29]*e[20]-1.*e[25]*e[32]*e[23]-.5000000000*e[34]*ep2[18]-.5000000000*e[34]*ep2[23]-.5000000000*e[34]*ep2[20]-.5000000000*e[34]*ep2[21]+.5000000000*ep2[19]*e[34]+.5000000000*ep2[22]*e[34]+1.500000000*e[34]*ep2[25]+.5000000000*e[34]*ep2[24]+.5000000000*e[34]*ep2[26];
    A[58]=e[16]*e[0]*e[6]+e[16]*e[2]*e[8]+e[1]*e[11]*e[2]-1.*e[1]*e[15]*e[6]+e[1]*e[9]*e[0]-1.*e[1]*e[14]*e[5]-1.*e[1]*e[12]*e[3]-1.*e[1]*e[17]*e[8]+e[4]*e[9]*e[3]+e[4]*e[0]*e[12]+e[4]*e[1]*e[13]+e[4]*e[11]*e[5]+e[4]*e[2]*e[14]+e[13]*e[0]*e[3]+e[13]*e[2]*e[5]+e[7]*e[9]*e[6]+e[7]*e[0]*e[15]+e[7]*e[1]*e[16]+e[7]*e[11]*e[8]+e[7]*e[2]*e[17]-.5000000000*e[10]*ep2[6]-.5000000000*e[10]*ep2[5]-.5000000000*e[10]*ep2[3]-.5000000000*e[10]*ep2[8]+1.500000000*e[10]*ep2[1]+.5000000000*e[10]*ep2[0]+.5000000000*e[10]*ep2[2]+.5000000000*e[10]*ep2[4]+.5000000000*e[10]*ep2[7];
    A[168]=e[13]*e[14]*e[17]+e[13]*e[12]*e[15]+e[10]*e[9]*e[15]+.5000000000*e[16]*ep2[15]-.5000000000*e[16]*ep2[11]-.5000000000*e[16]*ep2[12]-.5000000000*e[16]*ep2[14]+e[10]*e[11]*e[17]+.5000000000*ep2[10]*e[16]+.5000000000*ep3[16]-.5000000000*e[16]*ep2[9]+.5000000000*e[16]*ep2[17]+.5000000000*ep2[13]*e[16];
    A[57]=e[10]*e[29]*e[20]+e[22]*e[27]*e[12]+e[22]*e[9]*e[30]+e[22]*e[29]*e[14]+e[22]*e[11]*e[32]+e[22]*e[10]*e[31]+e[31]*e[18]*e[12]+e[31]*e[9]*e[21]+e[31]*e[20]*e[14]+e[31]*e[11]*e[23]-1.*e[10]*e[33]*e[24]-1.*e[10]*e[30]*e[21]-1.*e[10]*e[35]*e[26]-1.*e[10]*e[32]*e[23]+e[10]*e[34]*e[25]+e[19]*e[27]*e[9]+e[19]*e[29]*e[11]+e[28]*e[18]*e[9]+e[28]*e[20]*e[11]+e[16]*e[27]*e[24]+e[16]*e[18]*e[33]+e[16]*e[28]*e[25]+e[16]*e[19]*e[34]+e[16]*e[29]*e[26]+e[16]*e[20]*e[35]-1.*e[19]*e[30]*e[12]-1.*e[19]*e[32]*e[14]-1.*e[19]*e[33]*e[15]-1.*e[19]*e[35]*e[17]-1.*e[28]*e[23]*e[14]-1.*e[28]*e[24]*e[15]-1.*e[28]*e[26]*e[17]-1.*e[28]*e[21]*e[12]+e[25]*e[27]*e[15]+e[25]*e[9]*e[33]+e[25]*e[29]*e[17]+e[25]*e[11]*e[35]+e[34]*e[18]*e[15]+e[34]*e[9]*e[24]+e[34]*e[20]*e[17]+e[34]*e[11]*e[26]+e[13]*e[27]*e[21]+e[13]*e[18]*e[30]+e[13]*e[28]*e[22]+e[13]*e[19]*e[31]+e[13]*e[29]*e[23]+e[13]*e[20]*e[32]+e[10]*e[27]*e[18]+3.*e[10]*e[28]*e[19];
    A[171]=e[4]*e[30]*e[15]+e[4]*e[12]*e[33]+e[4]*e[32]*e[17]+e[4]*e[14]*e[35]+e[4]*e[31]*e[16]+e[4]*e[13]*e[34]+e[7]*e[33]*e[15]+e[7]*e[35]*e[17]+3.*e[7]*e[34]*e[16]+e[1]*e[27]*e[15]+e[1]*e[9]*e[33]+e[1]*e[29]*e[17]+e[1]*e[11]*e[35]+e[1]*e[28]*e[16]+e[1]*e[10]*e[34]-1.*e[16]*e[27]*e[0]-1.*e[16]*e[32]*e[5]+e[16]*e[33]*e[6]-1.*e[16]*e[30]*e[3]+e[16]*e[35]*e[8]-1.*e[16]*e[29]*e[2]+e[13]*e[30]*e[6]+e[13]*e[3]*e[33]+e[13]*e[31]*e[7]+e[13]*e[32]*e[8]+e[13]*e[5]*e[35]-1.*e[34]*e[11]*e[2]+e[34]*e[15]*e[6]-1.*e[34]*e[9]*e[0]-1.*e[34]*e[14]*e[5]-1.*e[34]*e[12]*e[3]+e[34]*e[17]*e[8]+e[31]*e[12]*e[6]+e[31]*e[3]*e[15]+e[31]*e[14]*e[8]+e[31]*e[5]*e[17]-1.*e[7]*e[27]*e[9]-1.*e[7]*e[30]*e[12]+e[7]*e[28]*e[10]-1.*e[7]*e[32]*e[14]+e[10]*e[27]*e[6]+e[10]*e[0]*e[33]+e[10]*e[29]*e[8]+e[10]*e[2]*e[35]+e[28]*e[9]*e[6]+e[28]*e[0]*e[15]+e[28]*e[11]*e[8]+e[28]*e[2]*e[17]-1.*e[7]*e[29]*e[11];
    A[56]=e[22]*e[18]*e[12]+e[22]*e[9]*e[21]+e[22]*e[20]*e[14]+e[22]*e[11]*e[23]+e[22]*e[19]*e[13]+e[25]*e[18]*e[15]+e[25]*e[9]*e[24]+e[25]*e[20]*e[17]+e[25]*e[11]*e[26]+e[25]*e[19]*e[16]+e[16]*e[18]*e[24]+e[16]*e[20]*e[26]+e[13]*e[18]*e[21]+e[13]*e[20]*e[23]+e[19]*e[18]*e[9]+e[19]*e[20]*e[11]-1.*e[19]*e[23]*e[14]-1.*e[19]*e[24]*e[15]-1.*e[19]*e[26]*e[17]-1.*e[19]*e[21]*e[12]+.5000000000*e[10]*ep2[22]+.5000000000*e[10]*ep2[25]+1.500000000*e[10]*ep2[19]+.5000000000*e[10]*ep2[18]+.5000000000*e[10]*ep2[20]-.5000000000*e[10]*ep2[26]-.5000000000*e[10]*ep2[23]-.5000000000*e[10]*ep2[24]-.5000000000*e[10]*ep2[21];
    A[170]=e[19]*e[20]*e[26]-.5000000000*e[25]*ep2[20]+e[22]*e[21]*e[24]+e[19]*e[18]*e[24]+.5000000000*ep2[22]*e[25]-.5000000000*e[25]*ep2[21]-.5000000000*e[25]*ep2[23]+.5000000000*ep2[19]*e[25]-.5000000000*e[25]*ep2[18]+.5000000000*e[25]*ep2[24]+.5000000000*e[25]*ep2[26]+.5000000000*ep3[25]+e[22]*e[23]*e[26];
    A[73]=-1.*e[20]*e[33]*e[6]-1.*e[20]*e[30]*e[3]-1.*e[20]*e[31]*e[4]-1.*e[29]*e[21]*e[3]-1.*e[29]*e[22]*e[4]-1.*e[29]*e[25]*e[7]-1.*e[29]*e[24]*e[6]+e[8]*e[27]*e[24]+e[8]*e[18]*e[33]+e[8]*e[28]*e[25]+e[8]*e[19]*e[34]+e[23]*e[27]*e[3]+e[23]*e[0]*e[30]+e[23]*e[28]*e[4]+e[23]*e[1]*e[31]+e[32]*e[18]*e[3]+e[32]*e[0]*e[21]+e[32]*e[19]*e[4]+e[32]*e[1]*e[22]+e[26]*e[27]*e[6]+e[26]*e[0]*e[33]+e[26]*e[28]*e[7]+e[26]*e[1]*e[34]+e[26]*e[29]*e[8]+e[26]*e[2]*e[35]+e[35]*e[18]*e[6]+e[35]*e[0]*e[24]+e[35]*e[19]*e[7]+e[35]*e[1]*e[25]+e[35]*e[20]*e[8]+e[2]*e[27]*e[18]+e[2]*e[28]*e[19]+3.*e[2]*e[29]*e[20]+e[20]*e[27]*e[0]+e[20]*e[28]*e[1]+e[29]*e[18]*e[0]+e[29]*e[19]*e[1]+e[5]*e[27]*e[21]+e[5]*e[18]*e[30]+e[5]*e[28]*e[22]+e[5]*e[19]*e[31]+e[5]*e[29]*e[23]+e[5]*e[20]*e[32]-1.*e[2]*e[33]*e[24]-1.*e[2]*e[30]*e[21]-1.*e[2]*e[31]*e[22]+e[2]*e[32]*e[23]-1.*e[2]*e[34]*e[25]-1.*e[20]*e[34]*e[7];
    A[72]=e[5]*e[18]*e[3]+e[5]*e[0]*e[21]+e[5]*e[19]*e[4]+e[5]*e[1]*e[22]+e[5]*e[2]*e[23]+e[23]*e[1]*e[4]+e[23]*e[0]*e[3]+e[8]*e[18]*e[6]+e[8]*e[0]*e[24]+e[8]*e[19]*e[7]+e[8]*e[1]*e[25]+e[8]*e[2]*e[26]+e[26]*e[1]*e[7]+e[26]*e[0]*e[6]+e[2]*e[18]*e[0]+e[2]*e[19]*e[1]-1.*e[2]*e[21]*e[3]-1.*e[2]*e[22]*e[4]-1.*e[2]*e[25]*e[7]-1.*e[2]*e[24]*e[6]-.5000000000*e[20]*ep2[4]+.5000000000*e[20]*ep2[0]-.5000000000*e[20]*ep2[6]+.5000000000*e[20]*ep2[5]+.5000000000*e[20]*ep2[1]-.5000000000*e[20]*ep2[7]-.5000000000*e[20]*ep2[3]+1.500000000*e[20]*ep2[2]+.5000000000*e[20]*ep2[8];
    A[75]=e[14]*e[9]*e[3]+e[14]*e[0]*e[12]+e[14]*e[10]*e[4]+e[14]*e[1]*e[13]+e[14]*e[11]*e[5]+e[17]*e[9]*e[6]+e[17]*e[0]*e[15]+e[17]*e[10]*e[7]+e[17]*e[1]*e[16]+e[17]*e[11]*e[8]+e[8]*e[9]*e[15]+e[8]*e[10]*e[16]+e[5]*e[9]*e[12]+e[5]*e[10]*e[13]+e[11]*e[9]*e[0]+e[11]*e[10]*e[1]-1.*e[11]*e[13]*e[4]-1.*e[11]*e[16]*e[7]-1.*e[11]*e[15]*e[6]-1.*e[11]*e[12]*e[3]+.5000000000*e[2]*ep2[14]+.5000000000*e[2]*ep2[17]+1.500000000*e[2]*ep2[11]+.5000000000*e[2]*ep2[9]+.5000000000*e[2]*ep2[10]-.5000000000*e[2]*ep2[16]-.5000000000*e[2]*ep2[12]-.5000000000*e[2]*ep2[15]-.5000000000*e[2]*ep2[13];
    A[74]=e[14]*e[18]*e[12]+e[14]*e[9]*e[21]+e[14]*e[11]*e[23]+e[14]*e[19]*e[13]+e[14]*e[10]*e[22]+e[23]*e[9]*e[12]+e[23]*e[10]*e[13]+e[17]*e[18]*e[15]+e[17]*e[9]*e[24]+e[17]*e[11]*e[26]+e[17]*e[19]*e[16]+e[17]*e[10]*e[25]+e[26]*e[9]*e[15]+e[26]*e[10]*e[16]-1.*e[11]*e[24]*e[15]-1.*e[11]*e[25]*e[16]+e[11]*e[18]*e[9]-1.*e[11]*e[21]*e[12]+e[11]*e[19]*e[10]-1.*e[11]*e[22]*e[13]+1.500000000*e[20]*ep2[11]+.5000000000*e[20]*ep2[9]+.5000000000*e[20]*ep2[10]+.5000000000*e[20]*ep2[14]+.5000000000*e[20]*ep2[17]-.5000000000*e[20]*ep2[16]-.5000000000*e[20]*ep2[12]-.5000000000*e[20]*ep2[15]-.5000000000*e[20]*ep2[13];
    A[77]=e[23]*e[10]*e[31]+e[32]*e[18]*e[12]+e[32]*e[9]*e[21]+e[32]*e[19]*e[13]+e[32]*e[10]*e[22]-1.*e[11]*e[33]*e[24]-1.*e[11]*e[30]*e[21]+e[11]*e[35]*e[26]-1.*e[11]*e[31]*e[22]-1.*e[11]*e[34]*e[25]+e[20]*e[27]*e[9]+e[20]*e[28]*e[10]+e[29]*e[18]*e[9]+e[29]*e[19]*e[10]+e[17]*e[27]*e[24]+e[17]*e[18]*e[33]+e[17]*e[28]*e[25]+e[17]*e[19]*e[34]+e[17]*e[29]*e[26]+e[17]*e[20]*e[35]-1.*e[20]*e[30]*e[12]-1.*e[20]*e[31]*e[13]-1.*e[20]*e[33]*e[15]-1.*e[20]*e[34]*e[16]-1.*e[29]*e[24]*e[15]-1.*e[29]*e[25]*e[16]-1.*e[29]*e[21]*e[12]-1.*e[29]*e[22]*e[13]+e[26]*e[27]*e[15]+e[26]*e[9]*e[33]+e[26]*e[28]*e[16]+e[26]*e[10]*e[34]+e[35]*e[18]*e[15]+e[35]*e[9]*e[24]+e[35]*e[19]*e[16]+e[35]*e[10]*e[25]+e[14]*e[27]*e[21]+e[14]*e[18]*e[30]+e[14]*e[28]*e[22]+e[14]*e[19]*e[31]+e[14]*e[29]*e[23]+e[14]*e[20]*e[32]+e[11]*e[27]*e[18]+e[11]*e[28]*e[19]+3.*e[11]*e[29]*e[20]+e[23]*e[27]*e[12]+e[23]*e[9]*e[30]+e[23]*e[11]*e[32]+e[23]*e[28]*e[13];
    A[76]=e[23]*e[18]*e[12]+e[23]*e[9]*e[21]+e[23]*e[20]*e[14]+e[23]*e[19]*e[13]+e[23]*e[10]*e[22]+e[26]*e[18]*e[15]+e[26]*e[9]*e[24]+e[26]*e[20]*e[17]+e[26]*e[19]*e[16]+e[26]*e[10]*e[25]+e[17]*e[19]*e[25]+e[17]*e[18]*e[24]+e[14]*e[19]*e[22]+e[14]*e[18]*e[21]+e[20]*e[18]*e[9]+e[20]*e[19]*e[10]-1.*e[20]*e[24]*e[15]-1.*e[20]*e[25]*e[16]-1.*e[20]*e[21]*e[12]-1.*e[20]*e[22]*e[13]+.5000000000*e[11]*ep2[23]+.5000000000*e[11]*ep2[26]+.5000000000*e[11]*ep2[19]+.5000000000*e[11]*ep2[18]+1.500000000*e[11]*ep2[20]-.5000000000*e[11]*ep2[22]-.5000000000*e[11]*ep2[24]-.5000000000*e[11]*ep2[21]-.5000000000*e[11]*ep2[25];
    A[79]=-1.*e[20]*e[21]*e[3]+e[20]*e[26]*e[8]-1.*e[20]*e[22]*e[4]-1.*e[20]*e[25]*e[7]-1.*e[20]*e[24]*e[6]+e[5]*e[19]*e[22]+e[5]*e[18]*e[21]+e[26]*e[18]*e[6]+e[26]*e[0]*e[24]+e[26]*e[19]*e[7]+e[26]*e[1]*e[25]+e[8]*e[19]*e[25]+e[8]*e[18]*e[24]+e[20]*e[18]*e[0]+e[20]*e[19]*e[1]+e[23]*e[18]*e[3]+e[23]*e[0]*e[21]+e[23]*e[19]*e[4]+e[23]*e[1]*e[22]+e[23]*e[20]*e[5]+1.500000000*ep2[20]*e[2]+.5000000000*e[2]*ep2[23]+.5000000000*e[2]*ep2[19]+.5000000000*e[2]*ep2[18]+.5000000000*e[2]*ep2[26]-.5000000000*e[2]*ep2[22]-.5000000000*e[2]*ep2[24]-.5000000000*e[2]*ep2[21]-.5000000000*e[2]*ep2[25];
    A[78]=-1.*e[2]*e[15]*e[6]+e[2]*e[9]*e[0]-1.*e[2]*e[12]*e[3]+e[5]*e[9]*e[3]+e[5]*e[0]*e[12]+e[5]*e[10]*e[4]+e[5]*e[1]*e[13]+e[5]*e[2]*e[14]+e[14]*e[1]*e[4]+e[14]*e[0]*e[3]+e[8]*e[9]*e[6]+e[8]*e[0]*e[15]+e[8]*e[10]*e[7]+e[8]*e[1]*e[16]+e[8]*e[2]*e[17]+e[17]*e[1]*e[7]+e[17]*e[0]*e[6]+e[2]*e[10]*e[1]-1.*e[2]*e[13]*e[4]-1.*e[2]*e[16]*e[7]+.5000000000*e[11]*ep2[1]+.5000000000*e[11]*ep2[0]+1.500000000*e[11]*ep2[2]+.5000000000*e[11]*ep2[5]+.5000000000*e[11]*ep2[8]-.5000000000*e[11]*ep2[4]-.5000000000*e[11]*ep2[6]-.5000000000*e[11]*ep2[7]-.5000000000*e[11]*ep2[3];
    A[64]=e[5]*e[19]*e[13]+e[5]*e[10]*e[22]+e[8]*e[18]*e[15]+e[8]*e[9]*e[24]+e[8]*e[20]*e[17]+e[8]*e[11]*e[26]+e[8]*e[19]*e[16]+e[8]*e[10]*e[25]+e[2]*e[18]*e[9]+e[2]*e[19]*e[10]-1.*e[11]*e[21]*e[3]-1.*e[11]*e[22]*e[4]-1.*e[11]*e[25]*e[7]-1.*e[11]*e[24]*e[6]+e[14]*e[18]*e[3]+e[14]*e[0]*e[21]+e[14]*e[19]*e[4]+e[14]*e[1]*e[22]+e[14]*e[2]*e[23]-1.*e[20]*e[13]*e[4]-1.*e[20]*e[16]*e[7]-1.*e[20]*e[15]*e[6]-1.*e[20]*e[12]*e[3]+e[23]*e[9]*e[3]+e[23]*e[0]*e[12]+e[23]*e[10]*e[4]+e[23]*e[1]*e[13]+e[17]*e[18]*e[6]+e[17]*e[0]*e[24]+e[17]*e[19]*e[7]+e[17]*e[1]*e[25]+e[17]*e[2]*e[26]-1.*e[2]*e[24]*e[15]-1.*e[2]*e[25]*e[16]-1.*e[2]*e[21]*e[12]-1.*e[2]*e[22]*e[13]+e[26]*e[9]*e[6]+e[26]*e[0]*e[15]+e[26]*e[10]*e[7]+e[26]*e[1]*e[16]+e[11]*e[18]*e[0]+e[11]*e[19]*e[1]+3.*e[11]*e[20]*e[2]+e[20]*e[9]*e[0]+e[20]*e[10]*e[1]+e[5]*e[18]*e[12]+e[5]*e[9]*e[21]+e[5]*e[20]*e[14]+e[5]*e[11]*e[23];
    A[65]=e[32]*e[1]*e[4]+e[32]*e[0]*e[3]+e[8]*e[27]*e[6]+e[8]*e[0]*e[33]+e[8]*e[28]*e[7]+e[8]*e[1]*e[34]+e[35]*e[1]*e[7]+e[35]*e[0]*e[6]+e[2]*e[27]*e[0]+e[2]*e[28]*e[1]-1.*e[2]*e[34]*e[7]+e[2]*e[32]*e[5]-1.*e[2]*e[33]*e[6]-1.*e[2]*e[30]*e[3]+e[2]*e[35]*e[8]-1.*e[2]*e[31]*e[4]+e[5]*e[27]*e[3]+e[5]*e[0]*e[30]+e[5]*e[28]*e[4]+e[5]*e[1]*e[31]+1.500000000*e[29]*ep2[2]-.5000000000*e[29]*ep2[4]+.5000000000*e[29]*ep2[0]-.5000000000*e[29]*ep2[6]+.5000000000*e[29]*ep2[5]+.5000000000*e[29]*ep2[1]-.5000000000*e[29]*ep2[7]-.5000000000*e[29]*ep2[3]+.5000000000*e[29]*ep2[8];
    A[66]=e[5]*e[0]*e[3]+e[8]*e[1]*e[7]+e[8]*e[0]*e[6]+e[5]*e[1]*e[4]-.5000000000*e[2]*ep2[4]+.5000000000*ep3[2]+.5000000000*e[2]*ep2[1]-.5000000000*e[2]*ep2[3]+.5000000000*e[2]*ep2[0]+.5000000000*e[2]*ep2[8]+.5000000000*e[2]*ep2[5]-.5000000000*e[2]*ep2[6]-.5000000000*e[2]*ep2[7];
    A[67]=e[35]*e[9]*e[15]+e[35]*e[10]*e[16]-1.*e[11]*e[30]*e[12]-1.*e[11]*e[31]*e[13]-1.*e[11]*e[33]*e[15]-1.*e[11]*e[34]*e[16]+e[11]*e[27]*e[9]+e[11]*e[28]*e[10]+e[14]*e[27]*e[12]+e[14]*e[9]*e[30]+e[14]*e[11]*e[32]+e[14]*e[28]*e[13]+e[14]*e[10]*e[31]+e[32]*e[9]*e[12]+e[32]*e[10]*e[13]+e[17]*e[27]*e[15]+e[17]*e[9]*e[33]+e[17]*e[11]*e[35]+e[17]*e[28]*e[16]+e[17]*e[10]*e[34]+1.500000000*e[29]*ep2[11]-.5000000000*e[29]*ep2[16]+.5000000000*e[29]*ep2[9]-.5000000000*e[29]*ep2[12]-.5000000000*e[29]*ep2[15]+.5000000000*e[29]*ep2[17]+.5000000000*e[29]*ep2[10]+.5000000000*e[29]*ep2[14]-.5000000000*e[29]*ep2[13];
    A[68]=e[14]*e[9]*e[12]+e[17]*e[10]*e[16]+e[17]*e[9]*e[15]+.5000000000*ep3[11]+e[14]*e[10]*e[13]+.5000000000*e[11]*ep2[10]-.5000000000*e[11]*ep2[15]+.5000000000*e[11]*ep2[14]-.5000000000*e[11]*ep2[13]-.5000000000*e[11]*ep2[12]+.5000000000*e[11]*ep2[9]-.5000000000*e[11]*ep2[16]+.5000000000*e[11]*ep2[17];
    A[69]=e[20]*e[27]*e[18]+e[20]*e[28]*e[19]+e[23]*e[27]*e[21]+e[23]*e[18]*e[30]+e[23]*e[28]*e[22]+e[23]*e[19]*e[31]+e[23]*e[20]*e[32]+e[32]*e[19]*e[22]+e[32]*e[18]*e[21]+e[26]*e[27]*e[24]+e[26]*e[18]*e[33]+e[26]*e[28]*e[25]+e[26]*e[19]*e[34]+e[26]*e[20]*e[35]+e[35]*e[19]*e[25]+e[35]*e[18]*e[24]-1.*e[20]*e[33]*e[24]-1.*e[20]*e[30]*e[21]-1.*e[20]*e[31]*e[22]-1.*e[20]*e[34]*e[25]+.5000000000*e[29]*ep2[23]+.5000000000*e[29]*ep2[26]-.5000000000*e[29]*ep2[22]-.5000000000*e[29]*ep2[24]-.5000000000*e[29]*ep2[21]-.5000000000*e[29]*ep2[25]+1.500000000*e[29]*ep2[20]+.5000000000*e[29]*ep2[19]+.5000000000*e[29]*ep2[18];
    A[70]=.5000000000*e[20]*ep2[26]+.5000000000*e[20]*ep2[18]+.5000000000*ep3[20]+.5000000000*e[20]*ep2[19]+e[26]*e[18]*e[24]+.5000000000*e[20]*ep2[23]-.5000000000*e[20]*ep2[25]+e[23]*e[19]*e[22]-.5000000000*e[20]*ep2[24]-.5000000000*e[20]*ep2[21]-.5000000000*e[20]*ep2[22]+e[23]*e[18]*e[21]+e[26]*e[19]*e[25];
    A[71]=e[8]*e[28]*e[16]+e[8]*e[10]*e[34]+e[2]*e[27]*e[9]+3.*e[2]*e[29]*e[11]+e[2]*e[28]*e[10]+e[11]*e[27]*e[0]-1.*e[11]*e[34]*e[7]-1.*e[11]*e[33]*e[6]-1.*e[11]*e[30]*e[3]+e[11]*e[28]*e[1]-1.*e[11]*e[31]*e[4]+e[14]*e[27]*e[3]+e[14]*e[0]*e[30]+e[14]*e[28]*e[4]+e[14]*e[1]*e[31]+e[14]*e[2]*e[32]+e[29]*e[10]*e[1]-1.*e[29]*e[13]*e[4]-1.*e[29]*e[16]*e[7]-1.*e[29]*e[15]*e[6]+e[29]*e[9]*e[0]-1.*e[29]*e[12]*e[3]+e[32]*e[9]*e[3]+e[32]*e[0]*e[12]+e[32]*e[10]*e[4]+e[32]*e[1]*e[13]+e[17]*e[27]*e[6]+e[17]*e[0]*e[33]+e[17]*e[28]*e[7]+e[17]*e[1]*e[34]+e[17]*e[2]*e[35]-1.*e[2]*e[30]*e[12]-1.*e[2]*e[31]*e[13]-1.*e[2]*e[33]*e[15]-1.*e[2]*e[34]*e[16]+e[35]*e[9]*e[6]+e[35]*e[0]*e[15]+e[35]*e[10]*e[7]+e[35]*e[1]*e[16]+e[5]*e[27]*e[12]+e[5]*e[9]*e[30]+e[5]*e[29]*e[14]+e[5]*e[11]*e[32]+e[5]*e[28]*e[13]+e[5]*e[10]*e[31]+e[8]*e[27]*e[15]+e[8]*e[9]*e[33]+e[8]*e[29]*e[17]+e[8]*e[11]*e[35];
    A[91]=-1.*e[12]*e[34]*e[7]+e[12]*e[32]*e[5]-1.*e[12]*e[35]*e[8]-1.*e[12]*e[29]*e[2]-1.*e[12]*e[28]*e[1]+e[12]*e[31]*e[4]-1.*e[30]*e[11]*e[2]-1.*e[30]*e[10]*e[1]+e[30]*e[13]*e[4]-1.*e[30]*e[16]*e[7]+e[30]*e[14]*e[5]-1.*e[30]*e[17]*e[8]+e[15]*e[3]*e[33]+e[15]*e[31]*e[7]+e[15]*e[4]*e[34]+e[15]*e[32]*e[8]+e[15]*e[5]*e[35]+e[3]*e[27]*e[9]-1.*e[3]*e[28]*e[10]-1.*e[3]*e[34]*e[16]-1.*e[3]*e[35]*e[17]-1.*e[3]*e[29]*e[11]+e[33]*e[13]*e[7]+e[33]*e[4]*e[16]+e[33]*e[14]*e[8]+e[33]*e[5]*e[17]+e[9]*e[28]*e[4]+e[9]*e[1]*e[31]+e[9]*e[29]*e[5]+e[9]*e[2]*e[32]+e[27]*e[10]*e[4]+e[27]*e[1]*e[13]+e[27]*e[11]*e[5]+e[27]*e[2]*e[14]+3.*e[3]*e[30]*e[12]+e[3]*e[32]*e[14]+e[3]*e[31]*e[13]+e[6]*e[30]*e[15]+e[6]*e[12]*e[33]+e[6]*e[32]*e[17]+e[6]*e[14]*e[35]+e[6]*e[31]*e[16]+e[6]*e[13]*e[34]+e[0]*e[27]*e[12]+e[0]*e[9]*e[30]+e[0]*e[29]*e[14]+e[0]*e[11]*e[32]+e[0]*e[28]*e[13]+e[0]*e[10]*e[31];
    A[90]=.5000000000*e[21]*ep2[24]-.5000000000*e[21]*ep2[25]+.5000000000*e[21]*ep2[23]-.5000000000*e[21]*ep2[26]+.5000000000*ep2[18]*e[21]+.5000000000*e[21]*ep2[22]-.5000000000*e[21]*ep2[20]+e[24]*e[22]*e[25]+e[24]*e[23]*e[26]-.5000000000*e[21]*ep2[19]+e[18]*e[19]*e[22]+e[18]*e[20]*e[23]+.5000000000*ep3[21];
    A[89]=-.5000000000*e[30]*ep2[26]-.5000000000*e[30]*ep2[19]-.5000000000*e[30]*ep2[20]-.5000000000*e[30]*ep2[25]+.5000000000*ep2[18]*e[30]+1.500000000*e[30]*ep2[21]+.5000000000*e[30]*ep2[22]+.5000000000*e[30]*ep2[23]+.5000000000*e[30]*ep2[24]+e[18]*e[27]*e[21]+e[18]*e[28]*e[22]+e[18]*e[19]*e[31]+e[18]*e[29]*e[23]+e[18]*e[20]*e[32]+e[27]*e[19]*e[22]+e[27]*e[20]*e[23]+e[21]*e[31]*e[22]+e[21]*e[32]*e[23]+e[24]*e[21]*e[33]+e[24]*e[31]*e[25]+e[24]*e[22]*e[34]+e[24]*e[32]*e[26]+e[24]*e[23]*e[35]+e[33]*e[22]*e[25]+e[33]*e[23]*e[26]-1.*e[21]*e[29]*e[20]-1.*e[21]*e[35]*e[26]-1.*e[21]*e[28]*e[19]-1.*e[21]*e[34]*e[25];
    A[88]=.5000000000*e[12]*ep2[15]-.5000000000*e[12]*ep2[17]+e[15]*e[13]*e[16]-.5000000000*e[12]*ep2[10]+e[15]*e[14]*e[17]-.5000000000*e[12]*ep2[16]-.5000000000*e[12]*ep2[11]+e[9]*e[10]*e[13]+.5000000000*e[12]*ep2[13]+.5000000000*ep2[9]*e[12]+.5000000000*ep3[12]+e[9]*e[11]*e[14]+.5000000000*e[12]*ep2[14];
    A[95]=e[12]*e[13]*e[4]+e[12]*e[14]*e[5]+e[15]*e[12]*e[6]+e[15]*e[13]*e[7]+e[15]*e[4]*e[16]+e[15]*e[14]*e[8]+e[15]*e[5]*e[17]+e[6]*e[14]*e[17]+e[6]*e[13]*e[16]+e[0]*e[11]*e[14]+e[0]*e[9]*e[12]+e[0]*e[10]*e[13]+e[9]*e[10]*e[4]+e[9]*e[1]*e[13]+e[9]*e[11]*e[5]+e[9]*e[2]*e[14]-1.*e[12]*e[11]*e[2]-1.*e[12]*e[10]*e[1]-1.*e[12]*e[16]*e[7]-1.*e[12]*e[17]*e[8]+1.500000000*ep2[12]*e[3]+.5000000000*e[3]*ep2[15]-.5000000000*e[3]*ep2[16]+.5000000000*e[3]*ep2[9]-.5000000000*e[3]*ep2[11]-.5000000000*e[3]*ep2[17]-.5000000000*e[3]*ep2[10]+.5000000000*e[3]*ep2[14]+.5000000000*e[3]*ep2[13];
    A[94]=e[18]*e[11]*e[14]+e[18]*e[9]*e[12]+e[18]*e[10]*e[13]+e[12]*e[23]*e[14]+e[12]*e[22]*e[13]+e[15]*e[12]*e[24]+e[15]*e[23]*e[17]+e[15]*e[14]*e[26]+e[15]*e[22]*e[16]+e[15]*e[13]*e[25]+e[24]*e[14]*e[17]+e[24]*e[13]*e[16]-1.*e[12]*e[25]*e[16]-1.*e[12]*e[26]*e[17]-1.*e[12]*e[20]*e[11]-1.*e[12]*e[19]*e[10]+e[9]*e[20]*e[14]+e[9]*e[11]*e[23]+e[9]*e[19]*e[13]+e[9]*e[10]*e[22]+.5000000000*ep2[9]*e[21]-.5000000000*e[21]*ep2[16]-.5000000000*e[21]*ep2[11]-.5000000000*e[21]*ep2[17]-.5000000000*e[21]*ep2[10]+1.500000000*e[21]*ep2[12]+.5000000000*e[21]*ep2[14]+.5000000000*e[21]*ep2[13]+.5000000000*e[21]*ep2[15];
    A[93]=-1.*e[21]*e[35]*e[8]-1.*e[21]*e[29]*e[2]-1.*e[21]*e[28]*e[1]+e[21]*e[31]*e[4]-1.*e[30]*e[26]*e[8]-1.*e[30]*e[20]*e[2]-1.*e[30]*e[19]*e[1]+e[30]*e[22]*e[4]-1.*e[30]*e[25]*e[7]+e[30]*e[23]*e[5]+e[6]*e[31]*e[25]+e[6]*e[22]*e[34]+e[6]*e[32]*e[26]+e[6]*e[23]*e[35]+e[24]*e[30]*e[6]+e[24]*e[3]*e[33]+e[24]*e[31]*e[7]+e[24]*e[4]*e[34]+e[24]*e[32]*e[8]+e[24]*e[5]*e[35]+e[33]*e[21]*e[6]+e[33]*e[22]*e[7]+e[33]*e[4]*e[25]+e[33]*e[23]*e[8]+e[33]*e[5]*e[26]+e[0]*e[27]*e[21]+e[0]*e[18]*e[30]+e[0]*e[28]*e[22]+e[0]*e[19]*e[31]+e[0]*e[29]*e[23]+e[0]*e[20]*e[32]+e[18]*e[27]*e[3]+e[18]*e[28]*e[4]+e[18]*e[1]*e[31]+e[18]*e[29]*e[5]+e[18]*e[2]*e[32]+e[27]*e[19]*e[4]+e[27]*e[1]*e[22]+e[27]*e[20]*e[5]+e[27]*e[2]*e[23]+3.*e[3]*e[30]*e[21]+e[3]*e[31]*e[22]+e[3]*e[32]*e[23]-1.*e[3]*e[29]*e[20]-1.*e[3]*e[35]*e[26]-1.*e[3]*e[28]*e[19]-1.*e[3]*e[34]*e[25]-1.*e[21]*e[34]*e[7]+e[21]*e[32]*e[5];
    A[92]=e[18]*e[1]*e[4]+e[18]*e[0]*e[3]+e[18]*e[2]*e[5]+e[3]*e[22]*e[4]+e[3]*e[23]*e[5]+e[6]*e[3]*e[24]+e[6]*e[22]*e[7]+e[6]*e[4]*e[25]+e[6]*e[23]*e[8]+e[6]*e[5]*e[26]+e[24]*e[4]*e[7]+e[24]*e[5]*e[8]+e[0]*e[19]*e[4]+e[0]*e[1]*e[22]+e[0]*e[20]*e[5]+e[0]*e[2]*e[23]-1.*e[3]*e[26]*e[8]-1.*e[3]*e[20]*e[2]-1.*e[3]*e[19]*e[1]-1.*e[3]*e[25]*e[7]+.5000000000*e[21]*ep2[4]+.5000000000*e[21]*ep2[0]+.5000000000*e[21]*ep2[6]+.5000000000*e[21]*ep2[5]-.5000000000*e[21]*ep2[1]-.5000000000*e[21]*ep2[7]+1.500000000*e[21]*ep2[3]-.5000000000*e[21]*ep2[2]-.5000000000*e[21]*ep2[8];
    A[82]=.5000000000*ep2[27]*e[21]+1.500000000*e[21]*ep2[30]+.5000000000*e[21]*ep2[32]+.5000000000*e[21]*ep2[31]+.5000000000*e[21]*ep2[33]-.5000000000*e[21]*ep2[28]-.5000000000*e[21]*ep2[29]-.5000000000*e[21]*ep2[34]-.5000000000*e[21]*ep2[35]+e[18]*e[27]*e[30]+e[18]*e[29]*e[32]+e[18]*e[28]*e[31]+e[27]*e[28]*e[22]+e[27]*e[19]*e[31]+e[27]*e[29]*e[23]+e[27]*e[20]*e[32]+e[30]*e[31]*e[22]+e[30]*e[32]*e[23]+e[24]*e[30]*e[33]+e[24]*e[32]*e[35]+e[24]*e[31]*e[34]+e[33]*e[31]*e[25]+e[33]*e[22]*e[34]+e[33]*e[32]*e[26]+e[33]*e[23]*e[35]-1.*e[30]*e[29]*e[20]-1.*e[30]*e[35]*e[26]-1.*e[30]*e[28]*e[19]-1.*e[30]*e[34]*e[25];
    A[192]=-.5000000000*e[26]*ep2[4]-.5000000000*e[26]*ep2[0]+.5000000000*e[26]*ep2[6]+.5000000000*e[26]*ep2[5]-.5000000000*e[26]*ep2[1]+.5000000000*e[26]*ep2[7]-.5000000000*e[26]*ep2[3]+.5000000000*e[26]*ep2[2]+1.500000000*e[26]*ep2[8]+e[20]*e[0]*e[6]+e[20]*e[2]*e[8]+e[5]*e[21]*e[6]+e[5]*e[3]*e[24]+e[5]*e[22]*e[7]+e[5]*e[4]*e[25]+e[5]*e[23]*e[8]+e[23]*e[4]*e[7]+e[23]*e[3]*e[6]+e[8]*e[24]*e[6]+e[8]*e[25]*e[7]+e[2]*e[18]*e[6]+e[2]*e[0]*e[24]+e[2]*e[19]*e[7]+e[2]*e[1]*e[25]-1.*e[8]*e[21]*e[3]-1.*e[8]*e[19]*e[1]-1.*e[8]*e[22]*e[4]-1.*e[8]*e[18]*e[0]+e[20]*e[1]*e[7];
    A[83]=e[9]*e[27]*e[30]+e[9]*e[29]*e[32]+e[9]*e[28]*e[31]+e[33]*e[30]*e[15]+e[33]*e[32]*e[17]+e[33]*e[14]*e[35]+e[33]*e[31]*e[16]+e[33]*e[13]*e[34]+e[27]*e[29]*e[14]+e[27]*e[11]*e[32]+e[27]*e[28]*e[13]+e[27]*e[10]*e[31]-1.*e[30]*e[28]*e[10]+e[30]*e[31]*e[13]+e[30]*e[32]*e[14]-1.*e[30]*e[34]*e[16]-1.*e[30]*e[35]*e[17]-1.*e[30]*e[29]*e[11]+e[15]*e[32]*e[35]+e[15]*e[31]*e[34]-.5000000000*e[12]*ep2[34]-.5000000000*e[12]*ep2[35]+.5000000000*e[12]*ep2[27]+.5000000000*e[12]*ep2[32]-.5000000000*e[12]*ep2[28]-.5000000000*e[12]*ep2[29]+.5000000000*e[12]*ep2[31]+.5000000000*e[12]*ep2[33]+1.500000000*e[12]*ep2[30];
    A[193]=e[23]*e[30]*e[6]+e[23]*e[3]*e[33]+e[23]*e[31]*e[7]+e[23]*e[4]*e[34]+e[32]*e[21]*e[6]+e[32]*e[3]*e[24]+e[32]*e[22]*e[7]+e[32]*e[4]*e[25]+e[26]*e[33]*e[6]+e[26]*e[34]*e[7]+3.*e[26]*e[35]*e[8]+e[35]*e[24]*e[6]+e[35]*e[25]*e[7]+e[2]*e[27]*e[24]+e[2]*e[18]*e[33]+e[2]*e[28]*e[25]+e[2]*e[19]*e[34]+e[2]*e[29]*e[26]+e[2]*e[20]*e[35]+e[20]*e[27]*e[6]+e[20]*e[0]*e[33]+e[20]*e[28]*e[7]+e[20]*e[1]*e[34]+e[20]*e[29]*e[8]+e[29]*e[18]*e[6]+e[29]*e[0]*e[24]+e[29]*e[19]*e[7]+e[29]*e[1]*e[25]+e[5]*e[30]*e[24]+e[5]*e[21]*e[33]+e[5]*e[31]*e[25]+e[5]*e[22]*e[34]+e[5]*e[32]*e[26]+e[5]*e[23]*e[35]-1.*e[8]*e[27]*e[18]+e[8]*e[33]*e[24]-1.*e[8]*e[30]*e[21]-1.*e[8]*e[31]*e[22]+e[8]*e[32]*e[23]-1.*e[8]*e[28]*e[19]+e[8]*e[34]*e[25]-1.*e[26]*e[27]*e[0]-1.*e[26]*e[30]*e[3]-1.*e[26]*e[28]*e[1]-1.*e[26]*e[31]*e[4]-1.*e[35]*e[21]*e[3]-1.*e[35]*e[19]*e[1]-1.*e[35]*e[22]*e[4]-1.*e[35]*e[18]*e[0];
    A[80]=e[27]*e[29]*e[32]+e[27]*e[28]*e[31]+e[33]*e[32]*e[35]+e[33]*e[31]*e[34]+.5000000000*ep3[30]-.5000000000*e[30]*ep2[28]-.5000000000*e[30]*ep2[29]-.5000000000*e[30]*ep2[34]+.5000000000*e[30]*ep2[33]+.5000000000*ep2[27]*e[30]+.5000000000*e[30]*ep2[32]+.5000000000*e[30]*ep2[31]-.5000000000*e[30]*ep2[35];
    A[194]=.5000000000*ep2[14]*e[26]+1.500000000*e[26]*ep2[17]+.5000000000*e[26]*ep2[15]+.5000000000*e[26]*ep2[16]+.5000000000*ep2[11]*e[26]-.5000000000*e[26]*ep2[9]-.5000000000*e[26]*ep2[12]-.5000000000*e[26]*ep2[10]-.5000000000*e[26]*ep2[13]+e[20]*e[11]*e[17]+e[20]*e[9]*e[15]+e[20]*e[10]*e[16]+e[14]*e[21]*e[15]+e[14]*e[12]*e[24]+e[14]*e[23]*e[17]+e[14]*e[22]*e[16]+e[14]*e[13]*e[25]+e[23]*e[12]*e[15]+e[23]*e[13]*e[16]+e[17]*e[24]*e[15]+e[17]*e[25]*e[16]-1.*e[17]*e[18]*e[9]-1.*e[17]*e[21]*e[12]-1.*e[17]*e[19]*e[10]-1.*e[17]*e[22]*e[13]+e[11]*e[18]*e[15]+e[11]*e[9]*e[24]+e[11]*e[19]*e[16]+e[11]*e[10]*e[25];
    A[81]=e[0]*e[27]*e[30]+e[0]*e[29]*e[32]+e[0]*e[28]*e[31]+e[30]*e[31]*e[4]+e[30]*e[32]*e[5]+e[6]*e[30]*e[33]+e[6]*e[32]*e[35]+e[6]*e[31]*e[34]+e[27]*e[28]*e[4]+e[27]*e[1]*e[31]+e[27]*e[29]*e[5]+e[27]*e[2]*e[32]+e[33]*e[31]*e[7]+e[33]*e[4]*e[34]+e[33]*e[32]*e[8]+e[33]*e[5]*e[35]-1.*e[30]*e[34]*e[7]-1.*e[30]*e[35]*e[8]-1.*e[30]*e[29]*e[2]-1.*e[30]*e[28]*e[1]+1.500000000*e[3]*ep2[30]+.5000000000*e[3]*ep2[32]+.5000000000*e[3]*ep2[31]+.5000000000*e[3]*ep2[27]-.5000000000*e[3]*ep2[28]-.5000000000*e[3]*ep2[29]+.5000000000*e[3]*ep2[33]-.5000000000*e[3]*ep2[34]-.5000000000*e[3]*ep2[35];
    A[195]=.5000000000*ep2[14]*e[8]+1.500000000*ep2[17]*e[8]+.5000000000*e[8]*ep2[15]+.5000000000*e[8]*ep2[16]-.5000000000*e[8]*ep2[9]+.5000000000*e[8]*ep2[11]-.5000000000*e[8]*ep2[12]-.5000000000*e[8]*ep2[10]-.5000000000*e[8]*ep2[13]+e[14]*e[12]*e[6]+e[14]*e[3]*e[15]+e[14]*e[13]*e[7]+e[14]*e[4]*e[16]+e[14]*e[5]*e[17]+e[17]*e[15]*e[6]+e[17]*e[16]*e[7]+e[2]*e[11]*e[17]+e[2]*e[9]*e[15]+e[2]*e[10]*e[16]+e[5]*e[12]*e[15]+e[5]*e[13]*e[16]+e[11]*e[9]*e[6]+e[11]*e[0]*e[15]+e[11]*e[10]*e[7]+e[11]*e[1]*e[16]-1.*e[17]*e[10]*e[1]-1.*e[17]*e[13]*e[4]-1.*e[17]*e[9]*e[0]-1.*e[17]*e[12]*e[3];
    A[86]=-.5000000000*e[3]*ep2[1]-.5000000000*e[3]*ep2[7]+.5000000000*ep3[3]-.5000000000*e[3]*ep2[8]+e[0]*e[2]*e[5]+.5000000000*e[3]*ep2[6]+.5000000000*e[3]*ep2[4]-.5000000000*e[3]*ep2[2]+e[0]*e[1]*e[4]+e[6]*e[4]*e[7]+.5000000000*ep2[0]*e[3]+.5000000000*e[3]*ep2[5]+e[6]*e[5]*e[8];
    A[196]=.5000000000*ep2[23]*e[17]+1.500000000*ep2[26]*e[17]+.5000000000*e[17]*ep2[25]+.5000000000*e[17]*ep2[24]-.5000000000*e[17]*ep2[18]-.5000000000*e[17]*ep2[19]+.5000000000*e[17]*ep2[20]-.5000000000*e[17]*ep2[22]-.5000000000*e[17]*ep2[21]+e[23]*e[21]*e[15]+e[23]*e[12]*e[24]+e[23]*e[14]*e[26]+e[23]*e[22]*e[16]+e[23]*e[13]*e[25]+e[26]*e[24]*e[15]+e[26]*e[25]*e[16]+e[11]*e[19]*e[25]+e[11]*e[18]*e[24]+e[11]*e[20]*e[26]+e[14]*e[22]*e[25]+e[14]*e[21]*e[24]+e[20]*e[18]*e[15]+e[20]*e[9]*e[24]+e[20]*e[19]*e[16]+e[20]*e[10]*e[25]-1.*e[26]*e[18]*e[9]-1.*e[26]*e[21]*e[12]-1.*e[26]*e[19]*e[10]-1.*e[26]*e[22]*e[13];
    A[87]=-1.*e[12]*e[34]*e[16]-1.*e[12]*e[35]*e[17]-1.*e[12]*e[29]*e[11]+e[9]*e[27]*e[12]+e[9]*e[29]*e[14]+e[9]*e[11]*e[32]+e[9]*e[28]*e[13]+e[9]*e[10]*e[31]+e[27]*e[11]*e[14]+e[27]*e[10]*e[13]+e[12]*e[32]*e[14]+e[12]*e[31]*e[13]+e[15]*e[12]*e[33]+e[15]*e[32]*e[17]+e[15]*e[14]*e[35]+e[15]*e[31]*e[16]+e[15]*e[13]*e[34]+e[33]*e[14]*e[17]+e[33]*e[13]*e[16]-1.*e[12]*e[28]*e[10]+.5000000000*ep2[9]*e[30]-.5000000000*e[30]*ep2[16]-.5000000000*e[30]*ep2[11]+1.500000000*e[30]*ep2[12]+.5000000000*e[30]*ep2[15]-.5000000000*e[30]*ep2[17]-.5000000000*e[30]*ep2[10]+.5000000000*e[30]*ep2[14]+.5000000000*e[30]*ep2[13];
    A[197]=e[32]*e[22]*e[16]+e[32]*e[13]*e[25]-1.*e[17]*e[27]*e[18]+e[17]*e[33]*e[24]-1.*e[17]*e[30]*e[21]+e[17]*e[29]*e[20]+3.*e[17]*e[35]*e[26]-1.*e[17]*e[31]*e[22]-1.*e[17]*e[28]*e[19]+e[17]*e[34]*e[25]+e[20]*e[27]*e[15]+e[20]*e[9]*e[33]+e[20]*e[28]*e[16]+e[20]*e[10]*e[34]+e[29]*e[18]*e[15]+e[29]*e[9]*e[24]+e[29]*e[19]*e[16]+e[29]*e[10]*e[25]-1.*e[26]*e[27]*e[9]-1.*e[26]*e[30]*e[12]-1.*e[26]*e[28]*e[10]-1.*e[26]*e[31]*e[13]+e[26]*e[33]*e[15]+e[26]*e[34]*e[16]+e[35]*e[24]*e[15]+e[35]*e[25]*e[16]-1.*e[35]*e[18]*e[9]-1.*e[35]*e[21]*e[12]-1.*e[35]*e[19]*e[10]-1.*e[35]*e[22]*e[13]+e[14]*e[30]*e[24]+e[14]*e[21]*e[33]+e[14]*e[31]*e[25]+e[14]*e[22]*e[34]+e[14]*e[32]*e[26]+e[14]*e[23]*e[35]+e[11]*e[27]*e[24]+e[11]*e[18]*e[33]+e[11]*e[28]*e[25]+e[11]*e[19]*e[34]+e[11]*e[29]*e[26]+e[11]*e[20]*e[35]+e[23]*e[30]*e[15]+e[23]*e[12]*e[33]+e[23]*e[32]*e[17]+e[23]*e[31]*e[16]+e[23]*e[13]*e[34]+e[32]*e[21]*e[15]+e[32]*e[12]*e[24];
    A[84]=e[6]*e[23]*e[17]+e[6]*e[14]*e[26]+e[6]*e[22]*e[16]+e[6]*e[13]*e[25]+e[0]*e[20]*e[14]+e[0]*e[11]*e[23]+e[0]*e[19]*e[13]+e[0]*e[10]*e[22]-1.*e[12]*e[26]*e[8]-1.*e[12]*e[20]*e[2]-1.*e[12]*e[19]*e[1]+e[12]*e[22]*e[4]-1.*e[12]*e[25]*e[7]+e[12]*e[23]*e[5]-1.*e[21]*e[11]*e[2]-1.*e[21]*e[10]*e[1]+e[21]*e[13]*e[4]-1.*e[21]*e[16]*e[7]+e[21]*e[14]*e[5]-1.*e[21]*e[17]*e[8]+e[15]*e[3]*e[24]+e[15]*e[22]*e[7]+e[15]*e[4]*e[25]+e[15]*e[23]*e[8]+e[15]*e[5]*e[26]-1.*e[3]*e[25]*e[16]-1.*e[3]*e[26]*e[17]-1.*e[3]*e[20]*e[11]-1.*e[3]*e[19]*e[10]+e[24]*e[13]*e[7]+e[24]*e[4]*e[16]+e[24]*e[14]*e[8]+e[24]*e[5]*e[17]+e[9]*e[18]*e[3]+e[9]*e[0]*e[21]+e[9]*e[19]*e[4]+e[9]*e[1]*e[22]+e[9]*e[20]*e[5]+e[9]*e[2]*e[23]+e[18]*e[0]*e[12]+e[18]*e[10]*e[4]+e[18]*e[1]*e[13]+e[18]*e[11]*e[5]+e[18]*e[2]*e[14]+3.*e[3]*e[21]*e[12]+e[3]*e[23]*e[14]+e[3]*e[22]*e[13]+e[6]*e[21]*e[15]+e[6]*e[12]*e[24];
    A[198]=.5000000000*ep2[5]*e[17]+1.500000000*e[17]*ep2[8]+.5000000000*e[17]*ep2[7]+.5000000000*e[17]*ep2[6]+.5000000000*ep2[2]*e[17]-.5000000000*e[17]*ep2[4]-.5000000000*e[17]*ep2[0]-.5000000000*e[17]*ep2[1]-.5000000000*e[17]*ep2[3]+e[11]*e[1]*e[7]+e[11]*e[0]*e[6]+e[11]*e[2]*e[8]+e[5]*e[12]*e[6]+e[5]*e[3]*e[15]+e[5]*e[13]*e[7]+e[5]*e[4]*e[16]+e[5]*e[14]*e[8]+e[14]*e[4]*e[7]+e[14]*e[3]*e[6]+e[8]*e[15]*e[6]+e[8]*e[16]*e[7]-1.*e[8]*e[10]*e[1]-1.*e[8]*e[13]*e[4]-1.*e[8]*e[9]*e[0]-1.*e[8]*e[12]*e[3]+e[2]*e[9]*e[6]+e[2]*e[0]*e[15]+e[2]*e[10]*e[7]+e[2]*e[1]*e[16];
    A[85]=e[6]*e[4]*e[34]+e[6]*e[32]*e[8]+e[6]*e[5]*e[35]+e[33]*e[4]*e[7]+e[33]*e[5]*e[8]+e[0]*e[27]*e[3]+e[0]*e[28]*e[4]+e[0]*e[1]*e[31]+e[0]*e[29]*e[5]+e[0]*e[2]*e[32]-1.*e[3]*e[34]*e[7]+e[3]*e[32]*e[5]+e[3]*e[33]*e[6]-1.*e[3]*e[35]*e[8]-1.*e[3]*e[29]*e[2]-1.*e[3]*e[28]*e[1]+e[3]*e[31]*e[4]+e[27]*e[1]*e[4]+e[27]*e[2]*e[5]+e[6]*e[31]*e[7]+.5000000000*e[30]*ep2[4]+.5000000000*e[30]*ep2[6]+.5000000000*e[30]*ep2[5]-.5000000000*e[30]*ep2[1]-.5000000000*e[30]*ep2[7]-.5000000000*e[30]*ep2[2]-.5000000000*e[30]*ep2[8]+.5000000000*ep2[0]*e[30]+1.500000000*e[30]*ep2[3];
    A[199]=.5000000000*ep2[23]*e[8]+1.500000000*ep2[26]*e[8]-.5000000000*e[8]*ep2[18]-.5000000000*e[8]*ep2[19]-.5000000000*e[8]*ep2[22]+.5000000000*e[8]*ep2[24]-.5000000000*e[8]*ep2[21]+.5000000000*e[8]*ep2[25]+.5000000000*ep2[20]*e[8]+e[20]*e[18]*e[6]+e[20]*e[0]*e[24]+e[20]*e[19]*e[7]+e[20]*e[1]*e[25]+e[20]*e[2]*e[26]+e[23]*e[21]*e[6]+e[23]*e[3]*e[24]+e[23]*e[22]*e[7]+e[23]*e[4]*e[25]+e[23]*e[5]*e[26]-1.*e[26]*e[21]*e[3]-1.*e[26]*e[19]*e[1]-1.*e[26]*e[22]*e[4]-1.*e[26]*e[18]*e[0]+e[26]*e[25]*e[7]+e[26]*e[24]*e[6]+e[2]*e[19]*e[25]+e[2]*e[18]*e[24]+e[5]*e[22]*e[25]+e[5]*e[21]*e[24];
    A[109]=e[19]*e[27]*e[21]+e[19]*e[18]*e[30]+e[19]*e[28]*e[22]+e[19]*e[29]*e[23]+e[19]*e[20]*e[32]+e[28]*e[18]*e[21]+e[28]*e[20]*e[23]+e[22]*e[30]*e[21]+e[22]*e[32]*e[23]+e[25]*e[30]*e[24]+e[25]*e[21]*e[33]+e[25]*e[22]*e[34]+e[25]*e[32]*e[26]+e[25]*e[23]*e[35]+e[34]*e[21]*e[24]+e[34]*e[23]*e[26]-1.*e[22]*e[27]*e[18]-1.*e[22]*e[33]*e[24]-1.*e[22]*e[29]*e[20]-1.*e[22]*e[35]*e[26]+.5000000000*ep2[19]*e[31]+1.500000000*e[31]*ep2[22]+.5000000000*e[31]*ep2[21]+.5000000000*e[31]*ep2[23]+.5000000000*e[31]*ep2[25]-.5000000000*e[31]*ep2[26]-.5000000000*e[31]*ep2[18]-.5000000000*e[31]*ep2[20]-.5000000000*e[31]*ep2[24];
    A[108]=-.5000000000*e[13]*ep2[15]+.5000000000*e[13]*ep2[16]+.5000000000*e[13]*ep2[12]+e[16]*e[12]*e[15]+.5000000000*ep3[13]+e[10]*e[11]*e[14]+.5000000000*e[13]*ep2[14]-.5000000000*e[13]*ep2[17]-.5000000000*e[13]*ep2[11]-.5000000000*e[13]*ep2[9]+.5000000000*ep2[10]*e[13]+e[10]*e[9]*e[12]+e[16]*e[14]*e[17];
    A[111]=-1.*e[13]*e[29]*e[2]-1.*e[31]*e[11]*e[2]-1.*e[31]*e[15]*e[6]-1.*e[31]*e[9]*e[0]+e[31]*e[14]*e[5]+e[31]*e[12]*e[3]-1.*e[31]*e[17]*e[8]+e[16]*e[30]*e[6]+e[16]*e[3]*e[33]+e[16]*e[4]*e[34]+e[16]*e[32]*e[8]+e[16]*e[5]*e[35]-1.*e[4]*e[27]*e[9]+e[4]*e[28]*e[10]-1.*e[4]*e[33]*e[15]-1.*e[4]*e[35]*e[17]-1.*e[4]*e[29]*e[11]+e[34]*e[12]*e[6]+e[34]*e[3]*e[15]+e[34]*e[14]*e[8]+e[34]*e[5]*e[17]+e[10]*e[27]*e[3]+e[10]*e[0]*e[30]+e[10]*e[29]*e[5]+e[10]*e[2]*e[32]+e[28]*e[9]*e[3]+e[28]*e[0]*e[12]+e[28]*e[11]*e[5]+e[28]*e[2]*e[14]+e[4]*e[30]*e[12]+e[4]*e[32]*e[14]+3.*e[4]*e[31]*e[13]+e[7]*e[30]*e[15]+e[7]*e[12]*e[33]+e[7]*e[32]*e[17]+e[7]*e[14]*e[35]+e[7]*e[31]*e[16]+e[7]*e[13]*e[34]+e[1]*e[27]*e[12]+e[1]*e[9]*e[30]+e[1]*e[29]*e[14]+e[1]*e[11]*e[32]+e[1]*e[28]*e[13]+e[1]*e[10]*e[31]-1.*e[13]*e[27]*e[0]+e[13]*e[32]*e[5]-1.*e[13]*e[33]*e[6]+e[13]*e[30]*e[3]-1.*e[13]*e[35]*e[8];
    A[110]=e[25]*e[23]*e[26]+e[19]*e[20]*e[23]+e[19]*e[18]*e[21]+e[25]*e[21]*e[24]+.5000000000*ep3[22]+.5000000000*e[22]*ep2[23]+.5000000000*ep2[19]*e[22]-.5000000000*e[22]*ep2[18]-.5000000000*e[22]*ep2[24]+.5000000000*e[22]*ep2[21]+.5000000000*e[22]*ep2[25]-.5000000000*e[22]*ep2[20]-.5000000000*e[22]*ep2[26];
    A[105]=e[34]*e[5]*e[8]+e[1]*e[27]*e[3]+e[1]*e[0]*e[30]+e[1]*e[28]*e[4]+e[1]*e[29]*e[5]+e[1]*e[2]*e[32]-1.*e[4]*e[27]*e[0]+e[4]*e[34]*e[7]+e[4]*e[32]*e[5]-1.*e[4]*e[33]*e[6]+e[4]*e[30]*e[3]-1.*e[4]*e[35]*e[8]-1.*e[4]*e[29]*e[2]+e[28]*e[0]*e[3]+e[28]*e[2]*e[5]+e[7]*e[30]*e[6]+e[7]*e[3]*e[33]+e[7]*e[32]*e[8]+e[7]*e[5]*e[35]+e[34]*e[3]*e[6]+.5000000000*ep2[1]*e[31]+1.500000000*e[31]*ep2[4]-.5000000000*e[31]*ep2[0]-.5000000000*e[31]*ep2[6]+.5000000000*e[31]*ep2[5]+.5000000000*e[31]*ep2[7]+.5000000000*e[31]*ep2[3]-.5000000000*e[31]*ep2[2]-.5000000000*e[31]*ep2[8];
    A[104]=e[1]*e[20]*e[14]+e[1]*e[11]*e[23]+e[13]*e[21]*e[3]-1.*e[13]*e[26]*e[8]-1.*e[13]*e[20]*e[2]-1.*e[13]*e[18]*e[0]+e[13]*e[23]*e[5]-1.*e[13]*e[24]*e[6]-1.*e[22]*e[11]*e[2]-1.*e[22]*e[15]*e[6]-1.*e[22]*e[9]*e[0]+e[22]*e[14]*e[5]+e[22]*e[12]*e[3]-1.*e[22]*e[17]*e[8]+e[16]*e[21]*e[6]+e[16]*e[3]*e[24]+e[16]*e[4]*e[25]+e[16]*e[23]*e[8]+e[16]*e[5]*e[26]-1.*e[4]*e[24]*e[15]-1.*e[4]*e[26]*e[17]-1.*e[4]*e[20]*e[11]-1.*e[4]*e[18]*e[9]+e[25]*e[12]*e[6]+e[25]*e[3]*e[15]+e[25]*e[14]*e[8]+e[25]*e[5]*e[17]+e[10]*e[18]*e[3]+e[10]*e[0]*e[21]+e[10]*e[19]*e[4]+e[10]*e[1]*e[22]+e[10]*e[20]*e[5]+e[10]*e[2]*e[23]+e[19]*e[9]*e[3]+e[19]*e[0]*e[12]+e[19]*e[1]*e[13]+e[19]*e[11]*e[5]+e[19]*e[2]*e[14]+e[4]*e[21]*e[12]+e[4]*e[23]*e[14]+3.*e[4]*e[22]*e[13]+e[7]*e[21]*e[15]+e[7]*e[12]*e[24]+e[7]*e[23]*e[17]+e[7]*e[14]*e[26]+e[7]*e[22]*e[16]+e[7]*e[13]*e[25]+e[1]*e[18]*e[12]+e[1]*e[9]*e[21];
    A[107]=e[10]*e[27]*e[12]+e[10]*e[9]*e[30]+e[10]*e[29]*e[14]+e[10]*e[11]*e[32]+e[10]*e[28]*e[13]+e[28]*e[11]*e[14]+e[28]*e[9]*e[12]+e[13]*e[30]*e[12]+e[13]*e[32]*e[14]+e[16]*e[30]*e[15]+e[16]*e[12]*e[33]+e[16]*e[32]*e[17]+e[16]*e[14]*e[35]+e[16]*e[13]*e[34]+e[34]*e[14]*e[17]+e[34]*e[12]*e[15]-1.*e[13]*e[27]*e[9]-1.*e[13]*e[33]*e[15]-1.*e[13]*e[35]*e[17]-1.*e[13]*e[29]*e[11]+.5000000000*ep2[10]*e[31]+.5000000000*e[31]*ep2[16]-.5000000000*e[31]*ep2[9]-.5000000000*e[31]*ep2[11]+.5000000000*e[31]*ep2[12]-.5000000000*e[31]*ep2[15]-.5000000000*e[31]*ep2[17]+.5000000000*e[31]*ep2[14]+1.500000000*e[31]*ep2[13];
    A[106]=-.5000000000*e[4]*ep2[6]-.5000000000*e[4]*ep2[0]+e[1]*e[2]*e[5]+.5000000000*e[4]*ep2[7]+e[1]*e[0]*e[3]+e[7]*e[5]*e[8]-.5000000000*e[4]*ep2[8]+.5000000000*e[4]*ep2[3]+.5000000000*e[4]*ep2[5]+e[7]*e[3]*e[6]-.5000000000*e[4]*ep2[2]+.5000000000*ep3[4]+.5000000000*ep2[1]*e[4];
    A[100]=e[34]*e[32]*e[35]-.5000000000*e[31]*ep2[35]+.5000000000*e[31]*ep2[34]+.5000000000*ep2[28]*e[31]+.5000000000*ep3[31]+.5000000000*e[31]*ep2[32]+e[34]*e[30]*e[33]-.5000000000*e[31]*ep2[27]+.5000000000*e[31]*ep2[30]-.5000000000*e[31]*ep2[33]-.5000000000*e[31]*ep2[29]+e[28]*e[29]*e[32]+e[28]*e[27]*e[30];
    A[101]=e[1]*e[27]*e[30]+e[1]*e[29]*e[32]+e[1]*e[28]*e[31]+e[31]*e[30]*e[3]+e[31]*e[32]*e[5]+e[7]*e[30]*e[33]+e[7]*e[32]*e[35]+e[7]*e[31]*e[34]+e[28]*e[27]*e[3]+e[28]*e[0]*e[30]+e[28]*e[29]*e[5]+e[28]*e[2]*e[32]+e[34]*e[30]*e[6]+e[34]*e[3]*e[33]+e[34]*e[32]*e[8]+e[34]*e[5]*e[35]-1.*e[31]*e[27]*e[0]-1.*e[31]*e[33]*e[6]-1.*e[31]*e[35]*e[8]-1.*e[31]*e[29]*e[2]+.5000000000*e[4]*ep2[30]+.5000000000*e[4]*ep2[32]+1.500000000*e[4]*ep2[31]-.5000000000*e[4]*ep2[27]+.5000000000*e[4]*ep2[28]-.5000000000*e[4]*ep2[29]-.5000000000*e[4]*ep2[33]+.5000000000*e[4]*ep2[34]-.5000000000*e[4]*ep2[35];
    A[102]=.5000000000*e[22]*ep2[30]+.5000000000*e[22]*ep2[32]+1.500000000*e[22]*ep2[31]+.5000000000*e[22]*ep2[34]-.5000000000*e[22]*ep2[27]-.5000000000*e[22]*ep2[29]-.5000000000*e[22]*ep2[33]-.5000000000*e[22]*ep2[35]+e[28]*e[18]*e[30]+e[28]*e[29]*e[23]+e[28]*e[20]*e[32]+e[31]*e[30]*e[21]+e[31]*e[32]*e[23]+e[25]*e[30]*e[33]+e[25]*e[32]*e[35]+e[25]*e[31]*e[34]+e[34]*e[30]*e[24]+e[34]*e[21]*e[33]+e[34]*e[32]*e[26]+e[34]*e[23]*e[35]-1.*e[31]*e[27]*e[18]-1.*e[31]*e[33]*e[24]-1.*e[31]*e[29]*e[20]-1.*e[31]*e[35]*e[26]+e[19]*e[27]*e[30]+e[19]*e[29]*e[32]+e[19]*e[28]*e[31]+e[28]*e[27]*e[21]+.5000000000*ep2[28]*e[22];
    A[103]=e[16]*e[30]*e[33]+e[16]*e[32]*e[35]+e[10]*e[27]*e[30]+e[10]*e[29]*e[32]+e[10]*e[28]*e[31]+e[34]*e[30]*e[15]+e[34]*e[12]*e[33]+e[34]*e[32]*e[17]+e[34]*e[14]*e[35]+e[34]*e[31]*e[16]+e[28]*e[27]*e[12]+e[28]*e[9]*e[30]+e[28]*e[29]*e[14]+e[28]*e[11]*e[32]-1.*e[31]*e[27]*e[9]+e[31]*e[30]*e[12]+e[31]*e[32]*e[14]-1.*e[31]*e[33]*e[15]-1.*e[31]*e[35]*e[17]-1.*e[31]*e[29]*e[11]-.5000000000*e[13]*ep2[27]+.5000000000*e[13]*ep2[32]+.5000000000*e[13]*ep2[28]-.5000000000*e[13]*ep2[29]+1.500000000*e[13]*ep2[31]-.5000000000*e[13]*ep2[33]+.5000000000*e[13]*ep2[30]+.5000000000*e[13]*ep2[34]-.5000000000*e[13]*ep2[35];
    A[96]=e[21]*e[23]*e[14]+e[21]*e[22]*e[13]+e[24]*e[21]*e[15]+e[24]*e[23]*e[17]+e[24]*e[14]*e[26]+e[24]*e[22]*e[16]+e[24]*e[13]*e[25]+e[15]*e[22]*e[25]+e[15]*e[23]*e[26]+e[9]*e[19]*e[22]+e[9]*e[18]*e[21]+e[9]*e[20]*e[23]+e[18]*e[20]*e[14]+e[18]*e[11]*e[23]+e[18]*e[19]*e[13]+e[18]*e[10]*e[22]-1.*e[21]*e[25]*e[16]-1.*e[21]*e[26]*e[17]-1.*e[21]*e[20]*e[11]-1.*e[21]*e[19]*e[10]+1.500000000*ep2[21]*e[12]+.5000000000*e[12]*ep2[24]-.5000000000*e[12]*ep2[26]+.5000000000*e[12]*ep2[18]+.5000000000*e[12]*ep2[23]-.5000000000*e[12]*ep2[19]-.5000000000*e[12]*ep2[20]+.5000000000*e[12]*ep2[22]-.5000000000*e[12]*ep2[25];
    A[97]=-1.*e[12]*e[29]*e[20]-1.*e[12]*e[35]*e[26]-1.*e[12]*e[28]*e[19]-1.*e[12]*e[34]*e[25]+e[18]*e[29]*e[14]+e[18]*e[11]*e[32]+e[18]*e[28]*e[13]+e[18]*e[10]*e[31]+e[27]*e[20]*e[14]+e[27]*e[11]*e[23]+e[27]*e[19]*e[13]+e[27]*e[10]*e[22]+e[15]*e[30]*e[24]+e[15]*e[21]*e[33]+e[15]*e[31]*e[25]+e[15]*e[22]*e[34]+e[15]*e[32]*e[26]+e[15]*e[23]*e[35]-1.*e[21]*e[28]*e[10]-1.*e[21]*e[34]*e[16]-1.*e[21]*e[35]*e[17]-1.*e[21]*e[29]*e[11]-1.*e[30]*e[25]*e[16]-1.*e[30]*e[26]*e[17]-1.*e[30]*e[20]*e[11]-1.*e[30]*e[19]*e[10]+e[24]*e[32]*e[17]+e[24]*e[14]*e[35]+e[24]*e[31]*e[16]+e[24]*e[13]*e[34]+e[33]*e[23]*e[17]+e[33]*e[14]*e[26]+e[33]*e[22]*e[16]+e[33]*e[13]*e[25]+3.*e[12]*e[30]*e[21]+e[12]*e[31]*e[22]+e[12]*e[32]*e[23]+e[9]*e[27]*e[21]+e[9]*e[18]*e[30]+e[9]*e[28]*e[22]+e[9]*e[19]*e[31]+e[9]*e[29]*e[23]+e[9]*e[20]*e[32]+e[21]*e[32]*e[14]+e[21]*e[31]*e[13]+e[30]*e[23]*e[14]+e[30]*e[22]*e[13]+e[12]*e[27]*e[18]+e[12]*e[33]*e[24];
    A[98]=e[0]*e[11]*e[5]+e[0]*e[2]*e[14]+e[9]*e[1]*e[4]+e[9]*e[0]*e[3]+e[9]*e[2]*e[5]+e[3]*e[13]*e[4]+e[3]*e[14]*e[5]+e[6]*e[3]*e[15]+e[6]*e[13]*e[7]+e[6]*e[4]*e[16]+e[6]*e[14]*e[8]+e[6]*e[5]*e[17]+e[15]*e[4]*e[7]+e[15]*e[5]*e[8]-1.*e[3]*e[11]*e[2]-1.*e[3]*e[10]*e[1]-1.*e[3]*e[16]*e[7]-1.*e[3]*e[17]*e[8]+e[0]*e[10]*e[4]+e[0]*e[1]*e[13]+1.500000000*e[12]*ep2[3]+.5000000000*e[12]*ep2[4]+.5000000000*e[12]*ep2[5]+.5000000000*e[12]*ep2[6]+.5000000000*ep2[0]*e[12]-.5000000000*e[12]*ep2[1]-.5000000000*e[12]*ep2[7]-.5000000000*e[12]*ep2[2]-.5000000000*e[12]*ep2[8];
    A[99]=e[21]*e[24]*e[6]+e[0]*e[19]*e[22]+e[0]*e[20]*e[23]+e[24]*e[22]*e[7]+e[24]*e[4]*e[25]+e[24]*e[23]*e[8]+e[24]*e[5]*e[26]+e[6]*e[22]*e[25]+e[6]*e[23]*e[26]+e[18]*e[0]*e[21]+e[18]*e[19]*e[4]+e[18]*e[1]*e[22]+e[18]*e[20]*e[5]+e[18]*e[2]*e[23]+e[21]*e[22]*e[4]+e[21]*e[23]*e[5]-1.*e[21]*e[26]*e[8]-1.*e[21]*e[20]*e[2]-1.*e[21]*e[19]*e[1]-1.*e[21]*e[25]*e[7]+1.500000000*ep2[21]*e[3]+.5000000000*e[3]*ep2[22]+.5000000000*e[3]*ep2[23]+.5000000000*e[3]*ep2[24]-.5000000000*e[3]*ep2[26]-.5000000000*e[3]*ep2[19]-.5000000000*e[3]*ep2[20]-.5000000000*e[3]*ep2[25]+.5000000000*ep2[18]*e[3];
    A[127]=e[11]*e[27]*e[12]+e[11]*e[9]*e[30]+e[11]*e[29]*e[14]+e[11]*e[28]*e[13]+e[11]*e[10]*e[31]+e[29]*e[9]*e[12]+e[29]*e[10]*e[13]+e[14]*e[30]*e[12]+e[14]*e[31]*e[13]+e[17]*e[30]*e[15]+e[17]*e[12]*e[33]+e[17]*e[14]*e[35]+e[17]*e[31]*e[16]+e[17]*e[13]*e[34]+e[35]*e[12]*e[15]+e[35]*e[13]*e[16]-1.*e[14]*e[27]*e[9]-1.*e[14]*e[28]*e[10]-1.*e[14]*e[33]*e[15]-1.*e[14]*e[34]*e[16]+.5000000000*ep2[11]*e[32]-.5000000000*e[32]*ep2[16]-.5000000000*e[32]*ep2[9]+.5000000000*e[32]*ep2[12]-.5000000000*e[32]*ep2[15]+.5000000000*e[32]*ep2[17]-.5000000000*e[32]*ep2[10]+1.500000000*e[32]*ep2[14]+.5000000000*e[32]*ep2[13];
    A[126]=e[8]*e[3]*e[6]+.5000000000*ep2[2]*e[5]-.5000000000*e[5]*ep2[0]+.5000000000*e[5]*ep2[4]-.5000000000*e[5]*ep2[6]+.5000000000*e[5]*ep2[8]+e[8]*e[4]*e[7]+.5000000000*ep3[5]+e[2]*e[0]*e[3]+.5000000000*e[5]*ep2[3]-.5000000000*e[5]*ep2[7]+e[2]*e[1]*e[4]-.5000000000*e[5]*ep2[1];
    A[125]=e[2]*e[27]*e[3]+e[2]*e[0]*e[30]+e[2]*e[28]*e[4]+e[2]*e[1]*e[31]+e[2]*e[29]*e[5]-1.*e[5]*e[27]*e[0]-1.*e[5]*e[34]*e[7]-1.*e[5]*e[33]*e[6]+e[5]*e[30]*e[3]+e[5]*e[35]*e[8]-1.*e[5]*e[28]*e[1]+e[5]*e[31]*e[4]+e[29]*e[1]*e[4]+e[29]*e[0]*e[3]+e[8]*e[30]*e[6]+e[8]*e[3]*e[33]+e[8]*e[31]*e[7]+e[8]*e[4]*e[34]+e[35]*e[4]*e[7]+e[35]*e[3]*e[6]+.5000000000*ep2[2]*e[32]+1.500000000*e[32]*ep2[5]+.5000000000*e[32]*ep2[4]-.5000000000*e[32]*ep2[0]-.5000000000*e[32]*ep2[6]-.5000000000*e[32]*ep2[1]-.5000000000*e[32]*ep2[7]+.5000000000*e[32]*ep2[3]+.5000000000*e[32]*ep2[8];
    A[124]=-1.*e[14]*e[19]*e[1]+e[14]*e[22]*e[4]-1.*e[14]*e[18]*e[0]-1.*e[14]*e[25]*e[7]-1.*e[14]*e[24]*e[6]-1.*e[23]*e[10]*e[1]+e[23]*e[13]*e[4]-1.*e[23]*e[16]*e[7]-1.*e[23]*e[15]*e[6]-1.*e[23]*e[9]*e[0]+e[23]*e[12]*e[3]+e[17]*e[21]*e[6]+e[17]*e[3]*e[24]+e[17]*e[22]*e[7]+e[17]*e[4]*e[25]+e[17]*e[5]*e[26]-1.*e[5]*e[24]*e[15]-1.*e[5]*e[25]*e[16]-1.*e[5]*e[18]*e[9]-1.*e[5]*e[19]*e[10]+e[26]*e[12]*e[6]+e[26]*e[3]*e[15]+e[26]*e[13]*e[7]+e[26]*e[4]*e[16]+e[11]*e[18]*e[3]+e[11]*e[0]*e[21]+e[11]*e[19]*e[4]+e[11]*e[1]*e[22]+e[11]*e[20]*e[5]+e[11]*e[2]*e[23]+e[20]*e[9]*e[3]+e[20]*e[0]*e[12]+e[20]*e[10]*e[4]+e[20]*e[1]*e[13]+e[20]*e[2]*e[14]+e[5]*e[21]*e[12]+3.*e[5]*e[23]*e[14]+e[5]*e[22]*e[13]+e[8]*e[21]*e[15]+e[8]*e[12]*e[24]+e[8]*e[23]*e[17]+e[8]*e[14]*e[26]+e[8]*e[22]*e[16]+e[8]*e[13]*e[25]+e[2]*e[18]*e[12]+e[2]*e[9]*e[21]+e[2]*e[19]*e[13]+e[2]*e[10]*e[22]+e[14]*e[21]*e[3];
    A[123]=-.5000000000*e[14]*ep2[27]+1.500000000*e[14]*ep2[32]-.5000000000*e[14]*ep2[28]+.5000000000*e[14]*ep2[29]+.5000000000*e[14]*ep2[31]-.5000000000*e[14]*ep2[33]+.5000000000*e[14]*ep2[30]-.5000000000*e[14]*ep2[34]+.5000000000*e[14]*ep2[35]+e[11]*e[27]*e[30]+e[11]*e[29]*e[32]+e[11]*e[28]*e[31]+e[35]*e[30]*e[15]+e[35]*e[12]*e[33]+e[35]*e[32]*e[17]+e[35]*e[31]*e[16]+e[35]*e[13]*e[34]+e[29]*e[27]*e[12]+e[29]*e[9]*e[30]+e[29]*e[28]*e[13]+e[29]*e[10]*e[31]-1.*e[32]*e[27]*e[9]+e[32]*e[30]*e[12]-1.*e[32]*e[28]*e[10]+e[32]*e[31]*e[13]-1.*e[32]*e[33]*e[15]-1.*e[32]*e[34]*e[16]+e[17]*e[30]*e[33]+e[17]*e[31]*e[34];
    A[122]=-.5000000000*e[23]*ep2[33]-.5000000000*e[23]*ep2[34]+.5000000000*ep2[29]*e[23]+.5000000000*e[23]*ep2[30]+1.500000000*e[23]*ep2[32]+.5000000000*e[23]*ep2[31]+.5000000000*e[23]*ep2[35]-.5000000000*e[23]*ep2[27]-.5000000000*e[23]*ep2[28]+e[32]*e[30]*e[21]+e[32]*e[31]*e[22]+e[26]*e[30]*e[33]+e[26]*e[32]*e[35]+e[26]*e[31]*e[34]+e[35]*e[30]*e[24]+e[35]*e[21]*e[33]+e[35]*e[31]*e[25]+e[35]*e[22]*e[34]-1.*e[32]*e[27]*e[18]-1.*e[32]*e[33]*e[24]-1.*e[32]*e[28]*e[19]-1.*e[32]*e[34]*e[25]+e[20]*e[27]*e[30]+e[20]*e[29]*e[32]+e[20]*e[28]*e[31]+e[29]*e[27]*e[21]+e[29]*e[18]*e[30]+e[29]*e[28]*e[22]+e[29]*e[19]*e[31];
    A[121]=e[2]*e[27]*e[30]+e[2]*e[29]*e[32]+e[2]*e[28]*e[31]+e[32]*e[30]*e[3]+e[32]*e[31]*e[4]+e[8]*e[30]*e[33]+e[8]*e[32]*e[35]+e[8]*e[31]*e[34]+e[29]*e[27]*e[3]+e[29]*e[0]*e[30]+e[29]*e[28]*e[4]+e[29]*e[1]*e[31]+e[35]*e[30]*e[6]+e[35]*e[3]*e[33]+e[35]*e[31]*e[7]+e[35]*e[4]*e[34]-1.*e[32]*e[27]*e[0]-1.*e[32]*e[34]*e[7]-1.*e[32]*e[33]*e[6]-1.*e[32]*e[28]*e[1]+.5000000000*e[5]*ep2[30]+1.500000000*e[5]*ep2[32]+.5000000000*e[5]*ep2[31]-.5000000000*e[5]*ep2[27]-.5000000000*e[5]*ep2[28]+.5000000000*e[5]*ep2[29]-.5000000000*e[5]*ep2[33]-.5000000000*e[5]*ep2[34]+.5000000000*e[5]*ep2[35];
    A[120]=.5000000000*e[32]*ep2[31]+.5000000000*e[32]*ep2[35]-.5000000000*e[32]*ep2[27]+e[29]*e[27]*e[30]+e[29]*e[28]*e[31]+e[35]*e[30]*e[33]+e[35]*e[31]*e[34]+.5000000000*ep2[29]*e[32]+.5000000000*ep3[32]-.5000000000*e[32]*ep2[33]-.5000000000*e[32]*ep2[34]+.5000000000*e[32]*ep2[30]-.5000000000*e[32]*ep2[28];
    A[118]=e[10]*e[1]*e[4]+e[10]*e[0]*e[3]+e[10]*e[2]*e[5]+e[4]*e[12]*e[3]+e[4]*e[14]*e[5]+e[7]*e[12]*e[6]+e[7]*e[3]*e[15]+e[7]*e[4]*e[16]+e[7]*e[14]*e[8]+e[7]*e[5]*e[17]+e[16]*e[3]*e[6]+e[16]*e[5]*e[8]-1.*e[4]*e[11]*e[2]-1.*e[4]*e[15]*e[6]-1.*e[4]*e[9]*e[0]-1.*e[4]*e[17]*e[8]+e[1]*e[9]*e[3]+e[1]*e[0]*e[12]+e[1]*e[11]*e[5]+e[1]*e[2]*e[14]+1.500000000*e[13]*ep2[4]+.5000000000*e[13]*ep2[3]+.5000000000*e[13]*ep2[5]+.5000000000*e[13]*ep2[7]+.5000000000*ep2[1]*e[13]-.5000000000*e[13]*ep2[0]-.5000000000*e[13]*ep2[6]-.5000000000*e[13]*ep2[2]-.5000000000*e[13]*ep2[8];
    A[119]=e[25]*e[21]*e[6]+e[25]*e[3]*e[24]+e[25]*e[23]*e[8]+e[25]*e[5]*e[26]+e[7]*e[21]*e[24]+e[7]*e[23]*e[26]+e[19]*e[18]*e[3]+e[19]*e[0]*e[21]+e[19]*e[1]*e[22]+e[19]*e[20]*e[5]+e[19]*e[2]*e[23]+e[22]*e[21]*e[3]+e[22]*e[23]*e[5]-1.*e[22]*e[26]*e[8]-1.*e[22]*e[20]*e[2]-1.*e[22]*e[18]*e[0]+e[22]*e[25]*e[7]-1.*e[22]*e[24]*e[6]+e[1]*e[18]*e[21]+e[1]*e[20]*e[23]+.5000000000*e[4]*ep2[25]-.5000000000*e[4]*ep2[26]-.5000000000*e[4]*ep2[18]-.5000000000*e[4]*ep2[20]-.5000000000*e[4]*ep2[24]+.5000000000*ep2[19]*e[4]+1.500000000*ep2[22]*e[4]+.5000000000*e[4]*ep2[21]+.5000000000*e[4]*ep2[23];
    A[116]=e[22]*e[21]*e[12]+e[22]*e[23]*e[14]+e[25]*e[21]*e[15]+e[25]*e[12]*e[24]+e[25]*e[23]*e[17]+e[25]*e[14]*e[26]+e[25]*e[22]*e[16]+e[16]*e[21]*e[24]+e[16]*e[23]*e[26]+e[10]*e[19]*e[22]+e[10]*e[18]*e[21]+e[10]*e[20]*e[23]+e[19]*e[18]*e[12]+e[19]*e[9]*e[21]+e[19]*e[20]*e[14]+e[19]*e[11]*e[23]-1.*e[22]*e[24]*e[15]-1.*e[22]*e[26]*e[17]-1.*e[22]*e[20]*e[11]-1.*e[22]*e[18]*e[9]-.5000000000*e[13]*ep2[26]-.5000000000*e[13]*ep2[18]+.5000000000*e[13]*ep2[23]+.5000000000*e[13]*ep2[19]-.5000000000*e[13]*ep2[20]-.5000000000*e[13]*ep2[24]+.5000000000*e[13]*ep2[21]+1.500000000*ep2[22]*e[13]+.5000000000*e[13]*ep2[25];
    A[117]=e[13]*e[30]*e[21]+3.*e[13]*e[31]*e[22]+e[13]*e[32]*e[23]+e[10]*e[27]*e[21]+e[10]*e[18]*e[30]+e[10]*e[28]*e[22]+e[10]*e[19]*e[31]+e[10]*e[29]*e[23]+e[10]*e[20]*e[32]+e[22]*e[30]*e[12]+e[22]*e[32]*e[14]+e[31]*e[21]*e[12]+e[31]*e[23]*e[14]-1.*e[13]*e[27]*e[18]-1.*e[13]*e[33]*e[24]-1.*e[13]*e[29]*e[20]-1.*e[13]*e[35]*e[26]+e[13]*e[28]*e[19]+e[13]*e[34]*e[25]+e[19]*e[27]*e[12]+e[19]*e[9]*e[30]+e[19]*e[29]*e[14]+e[19]*e[11]*e[32]+e[28]*e[18]*e[12]+e[28]*e[9]*e[21]+e[28]*e[20]*e[14]+e[28]*e[11]*e[23]+e[16]*e[30]*e[24]+e[16]*e[21]*e[33]+e[16]*e[31]*e[25]+e[16]*e[22]*e[34]+e[16]*e[32]*e[26]+e[16]*e[23]*e[35]-1.*e[22]*e[27]*e[9]-1.*e[22]*e[33]*e[15]-1.*e[22]*e[35]*e[17]-1.*e[22]*e[29]*e[11]-1.*e[31]*e[24]*e[15]-1.*e[31]*e[26]*e[17]-1.*e[31]*e[20]*e[11]-1.*e[31]*e[18]*e[9]+e[25]*e[30]*e[15]+e[25]*e[12]*e[33]+e[25]*e[32]*e[17]+e[25]*e[14]*e[35]+e[34]*e[21]*e[15]+e[34]*e[12]*e[24]+e[34]*e[23]*e[17]+e[34]*e[14]*e[26];
    A[114]=e[19]*e[11]*e[14]+e[19]*e[9]*e[12]+e[19]*e[10]*e[13]+e[13]*e[21]*e[12]+e[13]*e[23]*e[14]+e[16]*e[21]*e[15]+e[16]*e[12]*e[24]+e[16]*e[23]*e[17]+e[16]*e[14]*e[26]+e[16]*e[13]*e[25]+e[25]*e[14]*e[17]+e[25]*e[12]*e[15]-1.*e[13]*e[24]*e[15]-1.*e[13]*e[26]*e[17]-1.*e[13]*e[20]*e[11]-1.*e[13]*e[18]*e[9]+e[10]*e[18]*e[12]+e[10]*e[9]*e[21]+e[10]*e[20]*e[14]+e[10]*e[11]*e[23]+1.500000000*e[22]*ep2[13]+.5000000000*e[22]*ep2[14]+.5000000000*e[22]*ep2[12]+.5000000000*e[22]*ep2[16]+.5000000000*ep2[10]*e[22]-.5000000000*e[22]*ep2[9]-.5000000000*e[22]*ep2[11]-.5000000000*e[22]*ep2[15]-.5000000000*e[22]*ep2[17];
    A[115]=e[13]*e[12]*e[3]+e[13]*e[14]*e[5]+e[16]*e[12]*e[6]+e[16]*e[3]*e[15]+e[16]*e[13]*e[7]+e[16]*e[14]*e[8]+e[16]*e[5]*e[17]+e[7]*e[14]*e[17]+e[7]*e[12]*e[15]+e[1]*e[11]*e[14]+e[1]*e[9]*e[12]+e[1]*e[10]*e[13]+e[10]*e[9]*e[3]+e[10]*e[0]*e[12]+e[10]*e[11]*e[5]+e[10]*e[2]*e[14]-1.*e[13]*e[11]*e[2]-1.*e[13]*e[15]*e[6]-1.*e[13]*e[9]*e[0]-1.*e[13]*e[17]*e[8]+1.500000000*ep2[13]*e[4]+.5000000000*e[4]*ep2[16]-.5000000000*e[4]*ep2[9]-.5000000000*e[4]*ep2[11]+.5000000000*e[4]*ep2[12]-.5000000000*e[4]*ep2[15]-.5000000000*e[4]*ep2[17]+.5000000000*e[4]*ep2[10]+.5000000000*e[4]*ep2[14];
    A[112]=e[19]*e[1]*e[4]+e[19]*e[0]*e[3]+e[19]*e[2]*e[5]+e[4]*e[21]*e[3]+e[4]*e[23]*e[5]+e[7]*e[21]*e[6]+e[7]*e[3]*e[24]+e[7]*e[4]*e[25]+e[7]*e[23]*e[8]+e[7]*e[5]*e[26]+e[25]*e[3]*e[6]+e[25]*e[5]*e[8]+e[1]*e[18]*e[3]+e[1]*e[0]*e[21]+e[1]*e[20]*e[5]+e[1]*e[2]*e[23]-1.*e[4]*e[26]*e[8]-1.*e[4]*e[20]*e[2]-1.*e[4]*e[18]*e[0]-1.*e[4]*e[24]*e[6]+1.500000000*e[22]*ep2[4]-.5000000000*e[22]*ep2[0]-.5000000000*e[22]*ep2[6]+.5000000000*e[22]*ep2[5]+.5000000000*e[22]*ep2[1]+.5000000000*e[22]*ep2[7]+.5000000000*e[22]*ep2[3]-.5000000000*e[22]*ep2[2]-.5000000000*e[22]*ep2[8];
    A[113]=-1.*e[31]*e[20]*e[2]-1.*e[31]*e[18]*e[0]+e[31]*e[23]*e[5]-1.*e[31]*e[24]*e[6]+e[7]*e[30]*e[24]+e[7]*e[21]*e[33]+e[7]*e[32]*e[26]+e[7]*e[23]*e[35]+e[25]*e[30]*e[6]+e[25]*e[3]*e[33]+e[25]*e[31]*e[7]+e[25]*e[4]*e[34]+e[25]*e[32]*e[8]+e[25]*e[5]*e[35]+e[34]*e[21]*e[6]+e[34]*e[3]*e[24]+e[34]*e[22]*e[7]+e[34]*e[23]*e[8]+e[34]*e[5]*e[26]+e[1]*e[27]*e[21]+e[1]*e[18]*e[30]+e[1]*e[28]*e[22]+e[1]*e[19]*e[31]+e[1]*e[29]*e[23]+e[1]*e[20]*e[32]+e[19]*e[27]*e[3]+e[19]*e[0]*e[30]+e[19]*e[28]*e[4]+e[19]*e[29]*e[5]+e[19]*e[2]*e[32]+e[28]*e[18]*e[3]+e[28]*e[0]*e[21]+e[28]*e[20]*e[5]+e[28]*e[2]*e[23]+e[4]*e[30]*e[21]+3.*e[4]*e[31]*e[22]+e[4]*e[32]*e[23]-1.*e[4]*e[27]*e[18]-1.*e[4]*e[33]*e[24]-1.*e[4]*e[29]*e[20]-1.*e[4]*e[35]*e[26]-1.*e[22]*e[27]*e[0]+e[22]*e[32]*e[5]-1.*e[22]*e[33]*e[6]+e[22]*e[30]*e[3]-1.*e[22]*e[35]*e[8]-1.*e[22]*e[29]*e[2]+e[31]*e[21]*e[3]-1.*e[31]*e[26]*e[8];

    int perm[20] = {6, 8, 18, 15, 12, 5, 14, 7, 4, 11, 19, 13, 1, 16, 17, 3, 10, 9, 2, 0};
    double AA[200];
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) AA[i + j * 20] = A[perm[i] + j * 20];
    }

    for (int i = 0; i < 200; i++) {
        A[i] = AA[i];
    }
}
