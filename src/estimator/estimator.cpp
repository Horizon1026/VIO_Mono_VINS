#include <include/estimator/estimator.hpp>


/* 生效参数配置器 config 中的参数 */
bool VIOMono::UpdateConfigParams(void) {
    // TODO: 与 config 中参数有关的模块初始化在此添加
    bool res = this->UpdateConfigParams_FeatureTracker();
    if (res == false) {
        return false;
    }
    res = this->UpdateConfigParams_FrameManager();
    if (res == false) {
        return false;
    }
    res = this->UpdateConfigParams_IMU();
    if (res == false) {
        return false;
    }
    res = this->UpdateConfigParams_EstimatorInit();
    if (res == false) {
        return false;
    }
    res = this->UpdateConfigParams_Estimator();
    return res;
}


/* 输出一帧图像和 IMU 捆绑的数据。如果当前缓存区中并不存在有效量测，则返回 false */
bool VIOMono::GetCombinedMessage(CombinedMessage &message) {
    // 如果队列为空，则返回 false
    if (this->imuMessages.empty() || this->imageMessages.empty()) {
        return false;
    }

    // 如果当前 IMU 的最新量测的时间戳还没有超过队列中的首帧图像，则需要继续等待 IMU
    if (this->imuMessages.back()->timeStamp <= this->imageMessages.front()->timeStamp + this->current_imageTimeOffset) {
        return false;
    }

    // 如果首帧图像来了之后才有有效的 IMU 消息，则丢弃初始图像，因为他无法捆绑有效的 IMU 量测
    if (this->imuMessages.front()->timeStamp >= this->imageMessages.front()->timeStamp + this->current_imageTimeOffset) {
        this->imageMessages.pop_front();
        return false;
    }

    // 通过上述条件，则提取出队列中的首帧图像
    std::shared_ptr<ImageMessage> imageMessage = this->imageMessages.front();
    this->imageMessages.pop_front();

    // 提取出与此图像对应时间段的 IMU 量测序列
    std::vector<std::shared_ptr<IMUMessage>> subIMUMessages;
    subIMUMessages.reserve(20);
    while (this->imuMessages.front()->timeStamp <= imageMessage->timeStamp + this->current_imageTimeOffset) {
        subIMUMessages.emplace_back(this->imuMessages.front());
        this->imuMessages.pop_front();
    }

    // 如果最新一帧 IMU 量测的时间戳和图像时间戳不一致，则线性插值一个 IMU 量测
    if (std::fabs(subIMUMessages.back()->timeStamp - imageMessage->timeStamp - this->current_imageTimeOffset) > 1e-6) {
        // 采用线性插值的方法，计算图像时间戳对应位置的 IMU 量测
        std::shared_ptr<IMUMessage> mid(new IMUMessage());
        mid->timeStamp = imageMessage->timeStamp + this->current_imageTimeOffset;
        std::shared_ptr<IMUMessage> left = subIMUMessages.back();
        std::shared_ptr<IMUMessage> right = this->imuMessages.front();
        double rate = (mid->timeStamp - left->timeStamp) / (right->timeStamp - left->timeStamp);
        mid->gyro = left->gyro * (1 - rate) + right->gyro * rate;
        mid->accel = left->accel * (1 - rate) + right->accel * rate;

        // 添加时间戳与图像时间戳同步的 IMU 量测
        subIMUMessages.emplace_back(mid);
        // 给下一帧相机量测设置起点
        this->imuMessages.push_front(mid);
    } else {
        // 给下一帧相机量测设置起点
        this->imuMessages.push_front(subIMUMessages.back());
    }

    // 返回结果
    message = CombinedMessage(subIMUMessages, imageMessage);
    return true;
}


/* 打印出一帧图像和 IMU 捆绑的数据 */
void VIOMono::PrintCombinedMesage(CombinedMessage &message) {
    std::cout << "<Print> A combined message:" << std::endl;
    std::cout << "Image time is " << message.second->timeStamp << std::endl;
    for (unsigned long i = 0; i < message.first.size(); i++) {
        std::cout << "  Imu " << i << " time is " << message.first[i]->timeStamp <<
            ", gyro " << message.first[i]->gyro.transpose() <<
            ", accel " << message.first[i]->accel.transpose() << std::endl;
    }
}


/* 为特征点追踪器配置参数 */
bool VIOMono::UpdateConfigParams_FeatureTracker(void) {
    if (this->config == nullptr) {
        return false;
    }
    // 构造相机模型实例，并设置参数
    std::shared_ptr<PinholeCamera> newCamera(new PinholeCamera());
    newCamera->InitBaseParams(this->config->cameraIntrinsic.rows,
                              this->config->cameraIntrinsic.cols,
                              this->config->cameraIntrinsic.k1,
                              this->config->cameraIntrinsic.k2,
                              this->config->cameraIntrinsic.p1,
                              this->config->cameraIntrinsic.p2);
    newCamera->InitParams(this->config->cameraIntrinsic.fx,
                          this->config->cameraIntrinsic.fy,
                          this->config->cameraIntrinsic.cx,
                          this->config->cameraIntrinsic.cy,
                          this->config->cameraIntrinsic.mapScale);

    // 构造特征点追踪器实例，并设置参数
    std::shared_ptr<FeatureTracker> newFeatureTracker(new FeatureTracker());
    this->featureTracker = newFeatureTracker;
    this->featureTracker->SetCameraModel(newCamera);
    this->featureTracker->SetParams(this->config->trackerParams.featureNum,
                                    this->config->trackerParams.minDistance);
    std::cout << "<System> Feature tracker and camera model params configed." << std::endl;
    return true;
}


/* 初始化特征点管理器 */
bool VIOMono::InitializeLandmarkManager(void) {
    if (this->landmarkManager == nullptr) {
        std::shared_ptr<LandmarkManager> newLandmarkManager(new LandmarkManager());
        this->landmarkManager = newLandmarkManager;
    }
    this->landmarkManager->landmarks.clear();
    return true;
}


/* 初始化帧管理器，为其配置参数 */
bool VIOMono::UpdateConfigParams_FrameManager(void) {
    if (this->config == nullptr) {
        return false;
    }
    if (this->frameManager == nullptr) {
        std::shared_ptr<FrameManager> newFrameManager(new FrameManager(this->config->windowSize));
        this->frameManager = newFrameManager;
    }
    this->frameManager->frames.clear();
    return true;
}


/* 为 IMU 噪声模型配置参数 */
bool VIOMono::UpdateConfigParams_IMU(void) {
    if (this->config == nullptr) {
        return false;
    }
    IMUPreintegration::accel_noise = this->config->imuNoise.accel_n;
    IMUPreintegration::accel_random_walk = this->config->imuNoise.accel_w;
    IMUPreintegration::gyro_noise = this->config->imuNoise.gyro_n;
    IMUPreintegration::gyro_random_walk = this->config->imuNoise.gyro_w;
    
    // 初始化量测噪声协方差矩阵
    Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();
    float accel_n2 = IMUPreintegration::accel_noise * IMUPreintegration::accel_noise;
    float gyro_n2 = IMUPreintegration::gyro_noise * IMUPreintegration::gyro_noise;
    float accel_w2 = IMUPreintegration::accel_random_walk * IMUPreintegration::accel_random_walk;
    float gyro_w2 = IMUPreintegration::gyro_random_walk * IMUPreintegration::gyro_random_walk;
    IMUPreintegration::Q.block<3, 3>(0, 0) = accel_n2 * I3;
    IMUPreintegration::Q.block<3, 3>(3, 3) = gyro_n2 * I3;
    IMUPreintegration::Q.block<3, 3>(6, 6) = accel_n2 * I3;
    IMUPreintegration::Q.block<3, 3>(9, 9) = gyro_n2 * I3;
    IMUPreintegration::Q.block<3, 3>(12, 12) = accel_w2 * I3;
    IMUPreintegration::Q.block<3, 3>(15, 15) = gyro_w2 * I3;
    
    return true;
}


/* 处理一组捆绑数据 */
bool VIOMono::ProcessCombinedMessage(const CombinedMessage &message) {
    // 在图像中追踪特征点
    this->featureTracker->Track(message.second->image, message.second->timeStamp);
    // TODO: 画出特征点追踪效果
    this->featureTracker->DrawTrackingResult(1);
    // 将特征点追踪结果更新到特征点管理器
    this->landmarkManager->AddNewLandmarks(this->featureTracker->tracked_IDs,
                                           this->featureTracker->tracked_norm,
                                           this->featureTracker->tracked_pixel,
                                           this->featureTracker->currentImageID,
                                           this->current_imageTimeOffset);
    // TODO: 打印出管理的特征点的信息
    // this->landmarkManager->PrintAllLandmarks();
    // 构建新的 IMU 预积分块
    std::shared_ptr<IMUPreintegration> imuBlock(new IMUPreintegration(this->current_bias_a, this->current_bias_g));
    for (auto it = message.first.begin(); it != message.first.end(); ++it) {
        imuBlock->Propagate((*it)->timeStamp, (*it)->accel, (*it)->gyro);
    }
    // TODO: 打印出 IMU 预积分块的信息
    // imuBlock->PrintContent();
    // 构建新的扩展图像帧，为其添加图像和特征点信息
    std::shared_ptr<CombinedFrame> frame(new CombinedFrame(this->featureTracker->currentImageID, imuBlock));
    frame->image = message.second->image;
    frame->timeStamp = message.second->timeStamp;
    for (size_t &id : this->featureTracker->tracked_IDs) {
        auto it = this->landmarkManager->landmarks.find(id);
        if (it != this->landmarkManager->landmarks.end()) {
            frame->AddLandmark(it->second);
        }
    }
    // 将新的扩展图像帧添加到帧管理器中
    this->frameManager->AddNewFrame(frame);
    // TODO: 打印出关键帧管理器中的存储信息
    // this->frameManager->PrintAllFrames();
    // TODO: 画出第一帧和最后一帧的共视情况
    // this->frameManager->DrawCovisibleLandmarks(0, this->frameManager->frames.size() - 1);

    return true;
}


/* 移除最旧观测以及相关信息 */
bool VIOMono::RemoveOldestFrame(void) {
    this->landmarkManager->RemoveByFrameID(this->frameManager->frames.front()->ID, false);
    this->frameManager->RemoveFrame(0);
    // 基于特征点的 p_w 来更新他们的逆深度
    for (auto &item : this->landmarkManager->landmarks) {
        this->UsePosToUpdateInvdep(item.second);
    }
    return true;
}


/* 移除次新观测以及相关信息 */
bool VIOMono::RemoveSubnewFrame(void) {
    this->landmarkManager->RemoveByFrameID(this->frameManager->frames.back()->ID - 1, true);
    this->frameManager->RemoveFrame(this->frameManager->frames.size() - 2);
    this->featureTracker->AdjustImageIDCounter(1);
    // 基于特征点的 p_w 来更新他们的逆深度
    for (auto &item : this->landmarkManager->landmarks) {
        this->UsePosToUpdateInvdep(item.second);
    }
    return true;
}


/* 移除被标记为 Error 的特征点，同步移除在 frame 中的索引 */
bool VIOMono::RemoveErrorLandmarks(void) {
    std::vector<std::shared_ptr<Landmark>> needRemove;
    needRemove.reserve(30);
    for (auto &item : this->landmarkManager->landmarks) {
        auto &lm_ptr = item.second;
        if (lm_ptr->isSolved == Landmark::SolveStatus::ERROR) {
            // 遍历观测到这个特征点的所有观测帧，从中删除其索引
            for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
                auto frame = this->frameManager->GetFrame(lm_ptr->firstFrameID + i);
                frame->landmarks.erase(lm_ptr->ID);
            }
            needRemove.emplace_back(lm_ptr);
        }
    }
    // 从特征点管理器中移除特征点
    for (size_t i = 0; i < needRemove.size(); ++i) {
        this->landmarkManager->RemoveByID(needRemove[i]->ID);
    }
    // 从前端特征追踪器中移除特征点（可能没必要）
    // TODO
    return true;
}


/* 检查滑动窗口内的信息是否合法 */
bool VIOMono::CheckSlidingWindowInfo(void) {
    // 检查关键帧信息是否合法
    auto frame_0 = this->frameManager->frames.front();
    auto frame_n = this->frameManager->frames.back();
    if (frame_n->ID - frame_0->ID != this->frameManager->frames.size() - 1) {
        std::cout << "<SlidingWindow> Frame info is invalid." << std::endl;
        return false;
    }
    for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
        if ((*it)->ID + 1 != (*(std::next(it)))->ID) {
            return false;
        }
    }
    // 检查特征点信息是否合法，即特征点保存的观测是否能够全部找到对应
    for (auto &item : this->landmarkManager->landmarks) {
        auto &lm_ptr = item.second;
        for (size_t i = 0; i < lm_ptr->observes.size(); ++i) {
            auto frame = this->frameManager->GetFrame(lm_ptr->firstFrameID + i);
            if (frame == nullptr) {
                std::cout << "<SlidingWindow> Landmark " << lm_ptr->ID << " info is invalid. Cannot find frame " <<
                lm_ptr->firstFrameID + i << " in frames : ";
                for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
                    std::cout << (*it)->ID << " ";
                }
                std::cout << std::endl;
                return false;
            }
        }
    }
    return true;
}


/* 打印出最新状态数据 */
void VIOMono::PrintCurrentParams(void) {
    std::cout << "<CurrentParams> q_wc = [" << this->current_q_wc.w() << ", " << this->current_q_wc.x() << ", " <<
        this->current_q_wc.y() << ", " << this->current_q_wc.z() << "], \tt_wc = [" << this->current_t_wc.transpose() <<
        "], \tv_wc = [" << this->current_v_wc.transpose() << "]\n";
    std::cout << "<CurrentParams> q_wb = [" << this->current_q_wb.w() << ", " << this->current_q_wb.x() << ", " <<
        this->current_q_wb.y() << ", " << this->current_q_wb.z() << "], \tt_wb = [" << this->current_t_wb.transpose() <<
        "], \tv_wb = [" << this->current_v_wb.transpose() << "]\n";
    std::cout << "<CurrentParams> bias a = [" << this->current_bias_a.transpose() << "], bias g = [" <<
        this->current_bias_g.transpose() << "]\n";
}


/* 更新最新时刻状态值 */
void VIOMono::UpdateCurrentStatus(void) {
    // 更新全局最新参数
    this->current_bias_g = this->frameManager->frames.back()->imu->GetBiasG();
    this->current_bias_a = this->frameManager->frames.back()->imu->GetBiasA();
    this->current_q_wc = this->frameManager->frames.back()->q_wc;
    this->current_t_wc = this->frameManager->frames.back()->t_wc;
    this->current_v_wc = this->frameManager->frames.back()->v_wc;
    this->current_q_wb = this->frameManager->frames.back()->q_wb;
    this->current_t_wb = this->frameManager->frames.back()->t_wb;
    this->current_v_wb = this->frameManager->frames.back()->v_wb;
    this->current_timeStamp = this->frameManager->frames.back()->timeStamp;
}


/* 将最旧帧添加到历史轨迹中 */
void VIOMono::SaveOldestFrameToPoses(void) {
    auto oldest = this->frameManager->frames.front();
    std::shared_ptr<Pose> newPose(new Pose(oldest->timeStamp, oldest->q_wb, oldest->t_wb, oldest->v_wb));
    this->poses.emplace_back(newPose);
}