#include <include/camera_model/camera_base.hpp>

/* 相机模型初始化 */
void CameraBase::InitBaseParams(int rows, int cols, float k1, float k2, float p1, float p2) {
    this->rows = rows;
    this->cols = cols;
    this->k1 = k1;
    this->k2 = k2;
    this->p1 = p1;
    this->p2 = p2;
    this->mask = cv::Mat(this->rows, this->cols, CV_8UC1, cv::Scalar(255));
}


/* 设置屏蔽域 */
bool CameraBase::SetMask(const cv::Mat &mask) {
    if (mask.rows != this->rows || mask.cols != this->cols || mask.type() != CV_8UC1) {
        return false;
    } else {
        this->mask = mask;
        return true;
    }
}


/* 根据畸变参数和图像尺寸，构造反畸变映射表。粒度 scale 越大，反畸变映射越精确 */
bool CameraBase::CreateUndistortedMap(int scale) {
    // 初始化内存空间
    int mapCol = this->cols * scale;
    int mapRow = this->rows * scale;
    this->distortedToCorrectMap.reserve(mapRow);
    std::vector<std::pair<float, float>> line(mapCol, std::pair<float, float>(INFINITY, INFINITY));
    for (int i = 0; i < mapRow; i++) {
        this->distortedToCorrectMap.emplace_back(line);
    }

    //由正确的点坐标，计算畸变的点坐标
    float step = 1.0 / float(scale);
    for (float vc = - rows / 2; vc < rows + rows / 2; vc+=step) {
        for (float uc = - cols / 2; uc < cols + cols / 2; uc+=step) {
            cv::Point2f pc = cv::Point2f(uc, vc);
            cv::Point2f pd = this->Undistortion_Distortion(pc);
            int ud = int(pd.x * scale);
            int vd = int(pd.y * scale);
            if (pd.x - ud > 0.5) {
                ud++;
            }
            if (pd.y - vd > 0.5) {
                vd++;
            }
            if (vd > -1 && vd < mapRow && ud > -1 && ud < mapCol) {
                if (this->distortedToCorrectMap[vd][ud].first == INFINITY) {
                    this->distortedToCorrectMap[vd][ud].first = pc.x;
                    this->distortedToCorrectMap[vd][ud].second = pc.y;
                } else {
                    this->distortedToCorrectMap[vd][ud].first = (this->distortedToCorrectMap[vd][ud].first + pc.x) / 2;
                    this->distortedToCorrectMap[vd][ud].second = (this->distortedToCorrectMap[vd][ud].second + pc.y) / 2;
                }
            }
        }
    }
    this->mapScale = scale;
    return true;
}


/* 由畸变像素平面坐标，变化为非畸变像素平面坐标 */
cv::Point2f CameraBase::Distortion_Undistortion(cv::Point2f p_distort) {
    int ud = int(p_distort.x * this->mapScale);
    int vd = int(p_distort.y * this->mapScale);
    if (p_distort.x - ud > 0.5) {
        ++ud;
    }
    if (p_distort.y - vd > 0.5) {
        ++vd;
    }
    cv::Point2f p_correct = cv::Point2f(0, 0);
    if (vd > -1 && vd < (int)distortedToCorrectMap.size() && ud > -1 && ud < (int)distortedToCorrectMap[0].size()) {
        p_correct.x = distortedToCorrectMap[vd][ud].first;
        p_correct.y = distortedToCorrectMap[vd][ud].second;
    }

    return p_correct;
}