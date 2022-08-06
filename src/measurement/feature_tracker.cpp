#include <include/measurement/feature_tracker.hpp>

// 声明静态变量
size_t FeatureTracker::maxFeatureID = 1;
size_t FeatureTracker::maxImageID = 1;

/* 设置相机模型 */
bool FeatureTracker::SetCameraModel(std::shared_ptr<PinholeCamera> &camera) {
    if (camera == nullptr) {
        return false;
    } else {
        this->camera = camera;
        return true;
    }
}


/* 设置可配置参数 */
bool FeatureTracker::SetParams(int minFeatureNum, float minDistance) {
    this->minFeatureNum = minFeatureNum;
    this->minDistance = minDistance;
    this->Reserve(this->minFeatureNum);
    return true;
}


/* 输入一帧新的图像，追踪特征点 */
bool FeatureTracker::Track(const cv::Mat &image, double timeStamp) {
    // 检查输入图像的尺寸
    if (image.rows != this->camera->rows || image.cols != this->camera->cols) {
        return false;
    }

    // 为图像进行均值归一化
    cv::Mat newImage;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(image, newImage);

    // 更新时间戳
    this->next.timeStamp = timeStamp;

    // 首次输入时，需要做特殊处理，否则图像递进
    if (this->next.image.empty()) {
        this->pre.image = this->next.image = newImage;
        this->pre.rawPoints.clear();
        this->next.rawPoints.clear();
        this->pre.undistortedPoints.clear();
        this->next.undistortedPoints.clear();
    } else {
        this->next.image = newImage;
    }

    // 当且仅当上一次也成功追踪到特征点时，才继续追踪
    if (this->pre.rawPoints.size() > 0) {
        // 采用 LK 光流，以上一次的特征点位置为起点，在下一帧图像中追踪特征点
        this->next.rawPoints.clear();
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(this->pre.image, this->next.image,
                                 this->pre.rawPoints, this->next.rawPoints,
                                 status, err, cv::Size(21, 21), 3);
        // 基于光流追踪的结果，继续去除掉扎堆的多余特征点
        this->DetectDenseFeatures(status, this->minDistance);
        this->AdjustVectors(status);
        // 计算去畸变结果
        this->next.undistortedPoints.resize(this->next.rawPoints.size());
        for (size_t i = 0; i < this->next.rawPoints.size(); ++i) {
            this->next.undistortedPoints[i] = this->camera->Distortion_Undistortion(this->next.rawPoints[i]);
        }
        // 采用 RANSAC 方法，基于对极约束模型剔除 outliers
        cv::findEssentialMat(this->pre.undistortedPoints,
                             this->next.undistortedPoints,
                             this->camera->fx,
                             cv::Point2f(this->camera->cx, this->camera->cy),
                             cv::RANSAC,
                             0.99, 1.0, status);
        this->AdjustVectors(status);
        // 更新所有特征点的被追踪次数
        for (auto &cnt : this->trackedTimes) {
            ++cnt;
        }
    }

    // 检查特征点是否足够，不足时则补充检测新特征点
    if (this->next.rawPoints.size() < this->minFeatureNum) {
        cv::Mat mask = this->CreateMask(this->minDistance);
        std::vector<cv::Point2f> newPoints;
        size_t needNum = this->minFeatureNum - this->next.rawPoints.size();
        newPoints.reserve(needNum);
        cv::goodFeaturesToTrack(this->next.image,
                                newPoints,
                                needNum,
                                0.01,
                                this->minDistance,
                                mask);
        this->AddNewFeatures(newPoints);
    }

    // 基于去畸变像素坐标，计算光流速度
    this->ComputePixelVelocity();
    // 更新所有 ID 为 0 的特征点的 ID
    this->UpdateFeaturesID();
    // 更替存储，为下一次 track 作准备
    this->pre = this->next;
    // 更新输出结果
    this->UpdateTrackingResult();
    return true;
}


/* 回退图像 ID 的编号 */
void FeatureTracker::AdjustImageIDCounter(size_t offset) {
    maxImageID -= offset;
}


/* 重置图像 ID 的编号 */
void FeatureTracker::ResetImageIDCounter(size_t id) {
    maxImageID = id;
}


/* 绘制出原始图像的特征点追踪结果 */
void FeatureTracker::DrawTrackingResult(int delay) {
    cv::Mat showImage(this->next.image.rows, this->next.image.cols, CV_8UC3);
    cv::cvtColor(this->next.image, showImage, CV_GRAY2BGR);
    for (unsigned long i = 0; i < this->next.rawPoints.size(); i++) {
        cv::line(showImage, this->next.rawPoints[i], this->next.rawPoints[i] + this->pixelVelocity[i], cv::Scalar(0, 255, 0), 2);
        int color = this->trackedTimes[i] * 25;
        if (color > 255) {
            color = 255;
        }
        cv::circle(showImage, this->next.rawPoints[i], 2, cv::Scalar(255 - color, 0, color), 5);
    }
    cv::imshow("Feature tracking image", showImage);
    cv::waitKey(delay);
}


/* 绘制出畸变校正后图像的特征点追踪结果 */
void FeatureTracker::DrawUndistortedTrackingResult(void) {
    // 去除图像的畸变
    cv::Mat undistortedImage(this->next.image.rows, this->next.image.cols, CV_8UC1);
    for (unsigned int v = 0; v < undistortedImage.rows; v++) {
        for (unsigned int u = 0; u < undistortedImage.cols; u++) {
            cv::Point2f uv = this->camera->Undistortion_Distortion(cv::Point2f(u, v));
            undistortedImage.at<uchar>(v, u) = this->next.image.at<uchar>(uv.y, uv.x);
        }
    }

    cv::Mat showImage(this->next.image.rows, this->next.image.cols, CV_8UC3);
    cv::cvtColor(undistortedImage, showImage, CV_GRAY2BGR);
    for (unsigned long i = 0; i < this->next.undistortedPoints.size(); i++) {
        cv::line(showImage, this->next.undistortedPoints[i], this->next.undistortedPoints[i] + this->pixelVelocity[i], cv::Scalar(0, 255, 0), 2);
        int color = this->trackedTimes[i] * 25;
        if (color > 255) {
            color = 255;
        }
        cv::circle(showImage, this->next.undistortedPoints[i], 2, cv::Scalar(255 - color, 0, color), 5);
    }
    cv::imshow("Feature tracking undistorted image", showImage);
    cv::waitKey(1);
}


/* 特征点追踪器预分配空间 */
void FeatureTracker::Reserve(size_t size) {
    this->pre.rawPoints.reserve(size);
    this->pre.undistortedPoints.reserve(size);
    this->next.rawPoints.reserve(size);
    this->next.undistortedPoints.reserve(size);
    this->trackedTimes.reserve(size);
    this->pixelVelocity.reserve(size);
    this->tracked_IDs.reserve(size);
    this->tracked_pixel.reserve(size);
    this->tracked_norm.reserve(size);
}


/* 捆绑追踪次数、像素坐标和特征点 ID，基于追踪次数进行排序 */
std::vector<std::pair<size_t, std::pair<cv::Point2f, size_t>>> FeatureTracker::SortFeatures(void) {
    // 捆绑追踪次数、像素坐标和特征点 ID
    std::vector<std::pair<size_t, std::pair<cv::Point2f, size_t>>> cnt_pts_id;
    cnt_pts_id.reserve(this->next.rawPoints.size());
    for (size_t i = 0; i < this->next.rawPoints.size(); ++i) {
        cnt_pts_id.emplace_back(std::make_pair(this->trackedTimes[i], std::make_pair(
                                               this->next.rawPoints[i],
                                               this->tracked_IDs[i])));
    }

    // 按照特征点被追踪到的次数，进行重新排序
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(),
        [] (const std::pair<size_t, std::pair<cv::Point2f, size_t>> &a,
            const std::pair<size_t, std::pair<cv::Point2f, size_t>> &b) {
                return a.first > b.first;
            }
    );
    return cnt_pts_id;
}


/* 基于特征点被追踪到的次数，检测扎堆的特征点 */
void FeatureTracker::DetectDenseFeatures(std::vector<uchar> &status, int radius) {
    // 初始化屏蔽域和需要剔除的点的标志
    cv::Mat mask = cv::Mat(this->next.image.rows, this->next.image.cols,
                           CV_8UC1, cv::Scalar(255));
    // 确保 status 和所有特征点能够一一对应
    if (status.size() != this->next.rawPoints.size()) {
        status.resize(this->next.rawPoints.size(), 1);
    }

    // 按照被追踪次数，对特征点排序
    auto cnt_pts_id = this->SortFeatures();

    // 构造屏蔽域，优先保留追踪次数多的特征点，以此调整 status 作为输出
    for (size_t i = 0; i < cnt_pts_id.size(); ++i) {
        if (mask.at<uchar>(cnt_pts_id[i].second.first) == 255 && status[i] == 1) {
            cv::circle(mask, cnt_pts_id[i].second.first, radius, 0, -1);
        } else if (mask.at<uchar>(cnt_pts_id[i].second.first) == 0) {
            status[i] = 0;
        }
    }
}


/* 基于特征点被追踪到的次数，构造屏蔽域 */
cv::Mat FeatureTracker::CreateMask(int radius) {
    cv::Mat mask = cv::Mat(this->next.image.rows, this->next.image.cols, CV_8UC1, cv::Scalar(255));
    if (this->next.rawPoints.empty()) {
        return mask;
    }
    auto cnt_pts_id = this->SortFeatures();
    // 构造屏蔽域，优先保留追踪次数多的特征点
    for (size_t i = 0; i < cnt_pts_id.size(); ++i) {
        if (mask.at<uchar>(cnt_pts_id[i].second.first) == 255) {
            cv::circle(mask, cnt_pts_id[i].second.first, radius, 0, -1);
        }
    }
    return mask;
}


/* 基于待删除索引 status，调整 vector 容器 */
template<typename T>
void FeatureTracker::AdjustVector(std::vector<T> &v, const std::vector<uchar> &status) {
    size_t j = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1) {
            v[j] = v[i];
            ++j;
        }
    }
    v.resize(j);
}


/* 基于待删除索引 status，一次性调整所有需要调整的 vector 容器 */
void FeatureTracker::AdjustVectors(const std::vector<uchar> &status) {
    this->AdjustVector(this->pre.rawPoints, status);
    this->AdjustVector(this->pre.undistortedPoints, status);
    this->AdjustVector(this->next.rawPoints, status);
    this->AdjustVector(this->next.undistortedPoints, status);
    this->AdjustVector(this->tracked_IDs, status);
    this->AdjustVector(this->trackedTimes, status);
}


/* 添加新检测到的特征点 */
void FeatureTracker::AddNewFeatures(const std::vector<cv::Point2f> &newFeatures) {
    // 所有新的特征点的 ID 号先设置为 0，当且近当被多次追踪之后才会设置为可用 ID，可用 ID 至少为 1
    for (auto &point : newFeatures) {
        this->next.rawPoints.emplace_back(point);
        this->next.undistortedPoints.emplace_back(this->camera->Distortion_Undistortion(point));
        this->tracked_IDs.emplace_back(0);
        this->trackedTimes.emplace_back(1);
    }
}


/* 更新特征点的 ID 号 */
void FeatureTracker::UpdateFeaturesID(void) {
    for (unsigned long i = 0; i < this->next.rawPoints.size(); i++) {
        if (this->tracked_IDs[i] == 0) {
            this->tracked_IDs[i] = this->maxFeatureID;
            this->maxFeatureID++;
        }
    }
}


/* 基于去畸变像素坐标，计算光流速度 */
void FeatureTracker::ComputePixelVelocity(void) {
    // 因为 Track 的最后检测了新的特征点，所以 this->pre.rawPoints 的大小一定小于或者等于 this->next.undistortedPoints
    this->pixelVelocity.resize(this->next.undistortedPoints.size());
    size_t idx = 0;
    for ( ; idx < this->pre.rawPoints.size(); idx++) {
        this->pixelVelocity[idx].x = (this->pre.rawPoints[idx].x - this->next.rawPoints[idx].x);
        this->pixelVelocity[idx].y = (this->pre.rawPoints[idx].y - this->next.rawPoints[idx].y);
    }
    for ( ; idx < this->next.undistortedPoints.size(); idx++) {
        this->pixelVelocity[idx] = cv::Point2f(0.0, 0.0);
    }
}


/* 更新追踪结果 */
void FeatureTracker::UpdateTrackingResult(void) {
    this->currentImageID = this->maxImageID;
    ++this->maxImageID;

    this->tracked_pixel.resize(this->next.rawPoints.size());
    this->tracked_norm.resize(this->next.rawPoints.size());
    for (size_t i = 0; i < this->tracked_IDs.size(); ++i) {
        this->tracked_pixel[i] = Eigen::Vector2f(this->next.rawPoints[i].x,
                                                 this->next.rawPoints[i].y);
        this->tracked_norm[i] = this->camera->PixelPlane_NormlizedPlane(this->next.undistortedPoints[i]);
    }
}

