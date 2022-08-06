#include <unistd.h>

#include <include/estimator/estimator.hpp>

/* 输入一帧图像数据 */
bool VIOMono::PushImageMessage(std::shared_ptr<ImageMessage> newImageMessage) {
    if (newImageMessage == nullptr) {
        return false;
    } else {
        if (this->originTimeStamp < 0) {
            this->originTimeStamp = newImageMessage->timeStamp;
        }
        newImageMessage->timeStamp -= this->originTimeStamp;
        this->imageMessages.emplace_back(newImageMessage);
        return true;
    }
}


/* 输入一帧 IMU 数据 */
bool VIOMono::PushIMUMessage(std::shared_ptr<IMUMessage> newIMUMessage) {
    if (newIMUMessage == nullptr) {
        return false;
    } else {
        if (this->originTimeStamp < 0) {
            this->originTimeStamp = newIMUMessage->timeStamp;
        }
        newIMUMessage->timeStamp -= this->originTimeStamp;
        this->imuMessages.emplace_back(newIMUMessage);
        return true;
    }
}


/* VIO 初始化 */
bool VIOMono::Initialize(std::string &configFilePath) {
    this->imageMessages.clear();
    this->imuMessages.clear();
    this->current_imageTimeOffset = 0.0;
    this->originTimeStamp = -1;
    this->current_bias_a.setZero();
    this->current_bias_g.setZero();
    this->status = Status::NeedInit;
    this->poses.clear();
    this->costTimes.clear();

    // 读取配置参数，并生效其配置
    if (this->config == nullptr) {
        std::shared_ptr<Config> newConfig(new Config());
        this->config = newConfig;
    }
    bool res = this->config->LoadConfigFile(configFilePath);
    if (res == true) {
        // 生效配置器，本质上是对部分模块的初始化
        res = this->UpdateConfigParams();
        if (res == false) {
            return false;
        }
    }

    // 初始化其他管理器
    res = this->InitializeLandmarkManager();
    return res;
}


/* 单步运行 */
bool VIOMono::RunOnce(void) {
    CombinedMessage message;
    // 尝试获取有效捆绑数据
    bool res = this->GetCombinedMessage(message);
    if (res == true) {
        // 初步处理捆绑数据
        res = this->ProcessCombinedMessage(message);
        // 检查滑动窗口内的信息是否合法
        res = this->CheckSlidingWindowInfo();
        if (res == false) {
            return false;
        }
        // 当图像帧的数量达到需求
        if (this->frameManager->frames.size() >= this->frameManager->windowSize + 1) {
            // 如果系统未初始化，则需要先完成初始化
            if (this->status == Status::NeedInit) {
                bool res = this->EstimatorInit();
                if (res == false) {
                    // 初始化失败时，需要重新初始化，因此将特征点的三角化状态复原，并移除掉最旧一帧
                    for (auto &item : this->landmarkManager->landmarks) {
                        item.second->isSolved = Landmark::SolveStatus::NO;
                    }
                } else {
                    // 初始化成功时，调整系统状态
                    this->status = Status::Running;
                    // 更新最新状态
                    this->UpdateCurrentStatus();
                    // TODO: 画出特征点追踪效果(单步测试)
                    // this->featureTracker->DrawTrackingResult(0);
                    // this->Visualize();
                }
            }

            // 如果系统完成了初始化，则进行后端优化，并更新最新状态
            if (this->status != Status::NeedInit) {
                bool res = this->EstimatorOptimize();
                this->UpdateCurrentStatus();
                // TODO: 打印出特征点估计结果
                // this->landmarkManager->PrintAllLandmarks();
                // TODO: 打印出滑动窗口内关键帧状态
                // this->frameManager->PrintAllFrames();
                // TODO: 打印出最新估计结果
                this->PrintCurrentParams();
                // TODO: 画出特征点追踪效果(单步测试)
                // this->featureTracker->DrawTrackingResult(0);
                // this->Visualize();
                // this->frameManager->DrawCovisibleLandmarks(this->frameManager->frames.size() - 2, this->frameManager->frames.size() - 1);

                if (res == false) {
                    this->status = Status::NeedInit;
                    // 失败时，需要重新初始化，因此将特征点的三角化状态复原，并移除掉最旧一帧
                    for (auto &item : this->landmarkManager->landmarks) {
                        item.second->isSolved = Landmark::SolveStatus::NO;
                    }
                } else {
                    // 优化成功后，删除标记为 Error 的特征点
                    this->RemoveErrorLandmarks();
                }
            }

            // 根据求解器状态，判断滑动窗口内的数据管理应当做什么工作
            if (this->status == Status::MargOldest || this->status == Status::NeedInit) {
                // 将最旧的一帧保存到历史轨迹中
                this->SaveOldestFrameToPoses();
                // 移除最旧的一帧以及相关特征点观测
                this->RemoveOldestFrame();
            } else if (this->status == Status::MargSubnew) {
                // 移除次新的一帧以及相关特征点观测
                this->RemoveSubnewFrame();
            } else {
                std::cout << "<Estimator> Status error!" << std::endl;
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}


/* 开始运行，并设定超时时间 */
bool VIOMono::Run(size_t timeOut) {
    for (size_t timeCnt = 0; timeCnt < timeOut; ) {
        // 尝试单步运行
        bool res = this->RunOnce();

        // 超过一定时间没有接收到有效数据，则终止运行
        if (res == false) {
            ++timeCnt;
        }
        usleep(1000);
    }
    return true;
}


/* 将历史位姿写入到 this->config->pathes.savePoses 指定文件中 */
bool VIOMono::SavePosesAsFile() {
    std::ofstream file;
    file.open(this->config->pathes.savePoses.c_str());
    if (file.is_open()) {
        for (auto it = this->poses.begin(); it != this->poses.end(); ++it) {
            file << (*it)->timeStamp << " "
                 << (*it)->t_wb.x() << " "
                 << (*it)->t_wb.y() << " "
                 << (*it)->t_wb.z() << " "
                 << (*it)->q_wb.w() << " "
                 << (*it)->q_wb.x() << " "
                 << (*it)->q_wb.y() << " "
                 << (*it)->q_wb.z() << " "
                 << (*it)->v_wb.x() << " "
                 << (*it)->v_wb.y() << " "
                 << (*it)->v_wb.z() << " "
                 << std::endl;
        }
        for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
            file << (*it)->timeStamp << " "
                 << (*it)->t_wb.x() << " "
                 << (*it)->t_wb.y() << " "
                 << (*it)->t_wb.z() << " "
                 << (*it)->q_wb.w() << " "
                 << (*it)->q_wb.x() << " "
                 << (*it)->q_wb.y() << " "
                 << (*it)->q_wb.z() << " "
                 << (*it)->v_wb.x() << " "
                 << (*it)->v_wb.y() << " "
                 << (*it)->v_wb.z() << " "
                 << std::endl;
        }
        return true;
    } else {
        return false;
    }
}


/* 将历史优化耗时写入到 this->config->pathes.saveCostTimes 指定文件中 */
bool VIOMono::SaveCostTimesAsFile() {
    std::ofstream file;
    file.open(this->config->pathes.saveCostTimes.c_str());
    if (file.is_open()) {
        for (auto it = this->costTimes.begin(); it != this->costTimes.end(); ++it) {
            file << *it << std::endl;
        }
        return true;
    } else {
        return false;
    }
}