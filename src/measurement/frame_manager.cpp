#include <include/measurement/frame_manager.hpp>

/* 构造函数 */
Frame::Frame(const size_t &setID) {
    this->landmarks.clear();
    this->q_wc.setIdentity();
    this->t_wc.setZero();
    this->v_wc.setZero();
    this->ID = setID;
}


// 添加一个特征点
bool Frame::AddLandmark(std::shared_ptr<Landmark> &newLandmark) {
    if (this->landmarks.find(newLandmark->ID) == this->landmarks.end()) {
        this->landmarks.insert(std::make_pair(newLandmark->ID, newLandmark));
        return true;
    } else {
        return false;
    }
}


// 找到指定 ID 的一帧与此帧的共视特征点
std::vector<std::shared_ptr<Landmark>> Frame::GetCovisibleLandmarks(const std::shared_ptr<Frame> &target) {
    std::vector<std::shared_ptr<Landmark>> ret;
    ret.reserve(std::min(this->landmarks.size(), target->landmarks.size()));
    for (auto it = this->landmarks.begin(); it != this->landmarks.end(); ++it) {
        if (target->landmarks.find(it->first) != target->landmarks.end()) {
            ret.emplace_back(it->second);
        }
    }
    return ret;
}


// 计算指定 ID 的一帧与此帧的平均视差
float Frame::GetAverageParallax(const std::shared_ptr<Frame> &target) {
    std::vector<std::shared_ptr<Landmark>> landmarks = this->GetCovisibleLandmarks(target);
    if (landmarks.empty()) {
        return -1.0f;
    }
    Eigen::Vector2f temp;
    float averageParallax = 0.0f;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        temp = landmarks[i]->GetNorm(this->ID) - landmarks[i]->GetNorm(target->ID);
        averageParallax += temp.norm();
    }
    averageParallax /= static_cast<float>(landmarks.size());
    return averageParallax;
}


// 确定给定 ID 的目标帧与此帧的关联性，返回平均视差和共视特征点数量
std::pair<float, size_t> Frame::GetCorrespondence(const std::shared_ptr<Frame> &target) {
    std::pair<float, size_t> ret = std::make_pair(-1.0f, 0);
    std::vector<std::shared_ptr<Landmark>> landmarks = this->GetCovisibleLandmarks(target);
    if (landmarks.empty()) {
        return ret;
    }
    Eigen::Vector2f temp;
    ret.first = 0.0f;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        temp = landmarks[i]->GetNorm(this->ID) - landmarks[i]->GetNorm(target->ID);
        ret.first += temp.norm();
        ++ret.second;
    }
    ret.first /= static_cast<float>(ret.second);
    return ret;
}


/* 构造函数 */
CombinedFrame::CombinedFrame(const size_t &setID, const std::shared_ptr<IMUPreintegration> &setIMU) :
    Frame(setID), imu(setIMU) {
    this->q_wb.setIdentity();
    this->t_wb.setZero();
    this->v_wb.setZero();
}


/* 构造函数 */
FrameManager::FrameManager(const size_t &setWindowSize) : windowSize(setWindowSize) {
    this->frames.clear();
}


/* 增加新一帧图像帧，添加前检查 ID 是否合法 */
bool FrameManager::AddNewFrame(const std::shared_ptr<CombinedFrame> &newFrame) {
    if (this->frames.empty()) {
        this->frames.emplace_back(newFrame);
        return true;
    }
    if (newFrame->ID != this->frames.back()->ID + 1) {
        return false;
    } else {
        this->frames.emplace_back(newFrame);
        return true;
    }
}


/* 移除一帧图像帧，给定图像帧在滑动窗口内的索引（ 从 0 到 windowSize-1 ） */
bool FrameManager::RemoveFrame(const size_t offset) {
    if (offset >= this->frames.size()) {
        return false;
    }
    auto needRemove = this->frames.begin();
    // 如果被移除的不是第一帧，则需要将被移除一帧的 IMU 数据累加到后一帧上
    if (offset > 0) {
        size_t cnt = offset;
        while (cnt != 0) {
            ++needRemove;
            --cnt;
        }
        auto next = std::next(needRemove);
        // 将被移除帧的 IMU 量测累加到后一帧上（先把最后一帧的 IMU 按顺序累加到前一帧上，然后将最后一帧的 IMU 指向前一帧的）
        for (auto &item : (*next)->imu->GetBuff()) {
            (*needRemove)->imu->Propagate(item.timeStamp, item.accel, item.gyro);
        }
        (*next)->imu = (*needRemove)->imu;
        // 调整后续图像帧的 ID 号
        for (auto it = needRemove; it != this->frames.end(); ++it) {
            --(*it)->ID;
        }
    }
    // 删除此帧
    this->frames.erase(needRemove);
    return true;
}


/* 提取某一帧图像帧，提取前检查 ID 是否合法 */
std::shared_ptr<CombinedFrame> FrameManager::GetFrame(const size_t id) {
    if (this->frames.empty()) {
        return nullptr;
    }
    if (id < this->frames.front()->ID || id > this->frames.back()->ID) {
        return nullptr;
    }
    size_t offset = id - this->frames.front()->ID;
    auto it = this->frames.begin();
    while (offset) {
        --offset;
        ++it;
    }
    return (*it);
}


/* 打印出所有图像帧的信息 */
void FrameManager::PrintAllFrames(void) {
    std::cout << "<Frame Manager> Print all frames' information:" << std::endl;
    for (auto &item : this->frames) {
        std::cout << "  frame " << item->ID << " has " << item->landmarks.size() << " features." << std::endl;
        std::cout << "    imu buff size is " << item->imu->GetBuff().size() << std::endl;
        std::cout << "    q_wc is [" << item->q_wc.w() << ", " << item->q_wc.x() << ", " << item->q_wc.y() <<
            ", " << item->q_wc.z() << "], \tt_wc is [" << item->t_wc.transpose() << "]" << std::endl;
        std::cout << "    q_wb is [" << item->q_wb.w() << ", " << item->q_wb.x() << ", " << item->q_wb.y() <<
            ", " << item->q_wb.z() << "], \tt_wb is [" << item->t_wb.transpose() << "]" << std::endl;
    }
}


/* 绘制出指定两帧的特征点共视关系 */
void FrameManager::DrawCovisibleLandmarks(const size_t offset1, const size_t offset2) {
    // 检查索引是否合法
    if (offset1 >= this->frames.size() || offset2 >= this->frames.size() || offset1 == offset2) {
        return;
    }
    // 提取出两帧
    std::shared_ptr<CombinedFrame> frame1, frame2;
    size_t cnt = 0;
    for (auto it = this->frames.begin(); it != this->frames.end(); ++it) {
        if (cnt == offset1) {
            frame1 = *it;
        }
        if (cnt == offset2) {
            frame2 = *it;
        }
        ++cnt;
    }
    this->DrawCovisibleLandmarks(frame1, frame2);
}


/* 绘制出指定两帧的特征点共视关系 */
void FrameManager::DrawCovisibleLandmarks(const std::shared_ptr<CombinedFrame> &frame1, const std::shared_ptr<CombinedFrame> &frame2) {
    // 提取出两帧图像匹配的特征点，构造像素点对
    std::vector<std::shared_ptr<Landmark>> landmarks = frame1->GetCovisibleLandmarks(frame2);
    std::vector<cv::Point2f> points1(landmarks.size());
    std::vector<cv::Point2f> points2(landmarks.size());
    Eigen::Vector2f temp;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        temp = landmarks[i]->GetPixel(frame1->ID);
        points1[i] = cv::Point2f(temp.x(), temp.y());
        temp = landmarks[i]->GetPixel(frame2->ID);
        points2[i] = cv::Point2f(temp.x(), temp.y());
    }
    // 合并两帧图像
    cv::Mat mergedImage(frame1->image.rows, frame1->image.cols * 2, CV_8UC1);
    for (int v = 0; v < mergedImage.rows; ++v) {
        for (int u = 0; u < mergedImage.cols; ++u) {
            if (u < frame1->image.cols) {
                mergedImage.at<uchar>(v, u) = frame1->image.at<uchar>(v, u);
            } else {
                mergedImage.at<uchar>(v, u) = frame2->image.at<uchar>(v, u - frame1->image.cols);
            }
        }
    }
    // 构造用于显示的图像
    cv::Mat showImage(mergedImage.rows, mergedImage.cols, CV_8UC3);
    cv::cvtColor(mergedImage, showImage, CV_GRAY2BGR);
    cv::Point2f offset(frame1->image.cols, 0);
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::circle(showImage, points1[i], 2, cv::Scalar(0, 0, 255), 5);
        cv::circle(showImage, points2[i] + offset, 2, cv::Scalar(0, 255, 0), 5);
        cv::line(showImage, points1[i], points2[i] + offset, cv::Scalar(255, 0, 0), 1);
    }

    // 显示图像并等待
    cv::imshow("Covisible features in two frames", showImage);
    cv::waitKey(0);
}