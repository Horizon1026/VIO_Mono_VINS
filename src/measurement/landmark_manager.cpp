#include <include/measurement/landmark_manager.hpp>
#include <iostream>

/* Landmark 实例构造函数 */
Landmark::Landmark(const size_t ID_, const size_t firstFrameID_, const std::vector<std::shared_ptr<Observe>> &observes_) :
    ID(ID_), firstFrameID(firstFrameID_), observes(observes_) {
    this->invDep = 1.0f;
    this->p_w.setZero();
    this->isSolved = SolveStatus::NO;
    this->observes.reserve(15 > this->observes.size() ? 15 : this->observes.size() * 2);
}


/* 为此特征点添加一个观测 */
void Landmark::AddNewObserve(const std::shared_ptr<Observe> &newObserve) {
    this->observes.emplace_back(newObserve);
}


/* 获取指定 FrameID 的归一化平面坐标观测 */
Eigen::Vector2f Landmark::GetNorm(const size_t frameID) {
    size_t idx = frameID - this->firstFrameID;
    if (idx < this->observes.size()) {
        return this->observes[idx]->norm;
    } else {
        return Eigen::Vector2f::Zero();
    }
}


/* 获取指定 FrameID 的畸变像素坐标 */
Eigen::Vector2f Landmark::GetPixel(const size_t frameID) {
    size_t idx = frameID - this->firstFrameID;
    if (idx < this->observes.size()) {
        return this->observes[idx]->pixel;
    } else {
        return Eigen::Vector2f::Zero();
    }
}


/* 获取最后一个观测到此特征点的图像帧的 ID */
size_t Landmark::FinalFrameID(void) {
    return this->firstFrameID + this->observes.size() - 1;
}


/* 添加特征点信息 */
bool LandmarkManager::AddNewLandmarks(const std::vector<size_t> &IDs,
                                      const std::vector<Eigen::Vector2f> &norms,
                                      const std::vector<Eigen::Vector2f> &pixels,
                                      const size_t frameID,
                                      const float dt) {
    // 检查输入数据是否合法
    if (IDs.size() != norms.size() || IDs.size() != pixels.size()) {
        return false;
    }
    
    // 遍历输入的每一个特征点
    for (size_t i = 0; i < IDs.size(); ++i) {
        // 构造新的观测
        std::shared_ptr<Observe> newObserve(new Observe(norms[i], pixels[i], dt));
        // 查找当前特征点已经存在
        auto it = this->landmarks.find(IDs[i]);
        if (it != this->landmarks.end()) {
            // 如果当前特征点已存在，则在直接添加观测
            it->second->AddNewObserve(newObserve);
        } else {
            // 否则构造一个新的特征点
            std::vector<std::shared_ptr<Observe>> newObserves;
            newObserves.emplace_back(newObserve);
            std::shared_ptr<Landmark> newLandmark(new Landmark(IDs[i], frameID, newObserves));
            this->landmarks.insert(std::make_pair(newLandmark->ID, newLandmark));
        }
    }
    return true;
}


/* 移除指定关键帧所观测到的特征点。但此关键帧之后的观测帧 ID 会依次向前偏移 */
void LandmarkManager::RemoveByFrameID(const size_t frameID, bool offset) {
    // 记录失去所有观测的特征点的 ID，预先分配内存
    std::vector<size_t> needDelete;
    needDelete.reserve(100);

    // 根据 offset 决定关键帧的 ID 是否发生了偏移（是否进行了调整）
    if (offset) {
        // 滑动窗口内的帧 ID 发生了调整，对应着 marg subnew 的情况
        // 需要删除的观测一般位于中间，直接删除此观测即可
        // 如果需要删除的观测是第一个观测，则删除之后首次观测不变
        // 如果需要删除的观测是最后一个观测，直接删除即可
        for (auto &item : this->landmarks) {
            auto &landmark = item.second;
            if (landmark->firstFrameID <= frameID && frameID <= landmark->FinalFrameID()) {
                // 如果此特征点只有这一个观测，则准备删除
                if (landmark->observes.size()  == 1) {
                    landmark->observes.clear();
                    needDelete.emplace_back(landmark->ID);
                    continue;
                }
                // 如果此特征点有多个观测，则删除对应观测
                for (size_t i = frameID - landmark->firstFrameID; i < landmark->observes.size() - 1; ++i) {
                    landmark->observes[i] = landmark->observes[i + 1];
                }
                landmark->observes.resize(landmark->observes.size() - 1);
            } else if (frameID < landmark->firstFrameID) {
                // 如果需要删除的观测，在这个特征点的所有观测之前，需要调整其首次观测
                --landmark->firstFrameID;
            }
        }
    } else {
        // 滑动窗口内的帧 ID 没有发生调整，对应着 marg oldest 的情况
        // 只需要关注 landmark 的首次观测帧。去掉首次观测帧的观测之后，首次观测帧的 ID 需要加一
        for (auto &item : this->landmarks) {
            auto &landmark = item.second;
            if (landmark->firstFrameID == frameID) {
                // 如果此特征点只有这一个观测，则准备删除
                if (landmark->observes.size()  == 1) {
                    landmark->observes.clear();
                    needDelete.emplace_back(landmark->ID);
                    continue;
                }
                // 如果此特征点有多个观测，则删除对应观测
                for (size_t i = 0; i < landmark->observes.size() - 1; ++i) {
                    landmark->observes[i] = landmark->observes[i + 1];
                }
                landmark->observes.resize(landmark->observes.size() - 1);
                // 调整首帧观测的 ID
                ++landmark->firstFrameID;
            }
        }
    }

    // 剔除掉失去所有观测的特征点
    for (size_t i = 0; i < needDelete.size(); ++i) {
        this->landmarks.erase(needDelete[i]);
    }
}


/* 移除指定 ID 的特征点 */
void LandmarkManager::RemoveByID(const size_t landmarkID) {
    this->landmarks.erase(landmarkID);
}


/* 将所有特征点的指针都存放到一个 vector 容器中 */
std::vector<std::shared_ptr<Landmark>> LandmarkManager::GetAllLandmarks(void) {
    std::vector<std::shared_ptr<Landmark>> docker;
    docker.reserve(this->landmarks.size());
    size_t idx = 0;
    for (auto &landmark : this->landmarks) {
        docker[idx] = landmark.second;
        ++idx;
    }
    return docker;
}


/* 打印出保存的所有特征点的所有信息 */
void LandmarkManager::PrintAllLandmarks(void) {
    std::cout << "<Landmark Manager> Print all landmarks' information:" << std::endl;
    for (auto &item : this->landmarks) {
        std::cout << "  landmark " << item.second->ID << ", status ";
        if (item.second->isSolved == Landmark::SolveStatus::YES) {
            std::cout << "YES";
        } else if (item.second->isSolved == Landmark::SolveStatus::NO) {
            std::cout << "NO";
        } else {
            std::cout << "Error";
        }
        std::cout << ", invdep " << item.second->invDep << ", p_w " << item.second->p_w.transpose() << std::endl;
        for (size_t i = 0; i < item.second->observes.size(); ++i) {
            std::cout << "    observed in frame " << i + item.second->firstFrameID << ", norm " <<
                item.second->observes[i]->norm.transpose() << ", pixel " << item.second->observes[i]->pixel.transpose() <<
                ", delta t " << item.second->observes[i]->dt << std::endl;
        }
    }
}

