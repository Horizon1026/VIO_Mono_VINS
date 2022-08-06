#include <include/config_parameters/config.hpp>

/* 读取 config 配置文件 */
bool Config::LoadConfigFile(std::string &filePath) {
    // 尝试打开文件，如果打开失败则返回 false
    cv::FileStorage file;
    file.open(filePath, cv::FileStorage::READ);
    if (!file.isOpened()) {
        std::cout << "<Config> Cannot open config file: " << filePath << std::endl;
        return false;
    } else {
        std::cout << "<Config> Load parameters from " << filePath << std::endl;
    }

    // 读取路径参数
    this->pathes.savePoses = static_cast<std::string>(file["save_poses_file"]);
    this->pathes.globalMask = static_cast<std::string>(file["global_mask"]);
    this->pathes.saveCostTimes = static_cast<std::string>(file["save_cost_times_file"]);

    // 读取后端优化求解器类型
    std::string solverName = static_cast<std::string>(file["solver"]);
    if (solverName == "graph_optimize") {
        solverType = 0;
        std::cout << "<Config> Back end solver type is " << solverName << std::endl;
    } else if (solverName == "root_vio") {
        solverType = 1;
        std::cout << "<Config> Back end solver type is " << solverName << std::endl;
    } else {
        solverType = 0;
        std::cout << "<Config> Back end solver type is graph_optimize" << std::endl;
    }

    // 读取相机参数
    this->cameraIntrinsic.rows = file["camera_rows"];
    this->cameraIntrinsic.cols = file["camera_cols"];
    std::cout << "<Config> Image size: rows(" << this->cameraIntrinsic.rows << ") * cols(" <<
        this->cameraIntrinsic.cols << ")" << std::endl;
    cv::FileNode node = file["camera_distortion"];
    this->cameraIntrinsic.k1 = static_cast<float>(node["k1"]);
    this->cameraIntrinsic.k2 = static_cast<float>(node["k2"]);
    this->cameraIntrinsic.p1 = static_cast<float>(node["p1"]);
    this->cameraIntrinsic.p2 = static_cast<float>(node["p2"]);
    std::cout << "<Config> Camera distortion parameters: k1 = " << this->cameraIntrinsic.k1 << ", k2 = " <<
        this->cameraIntrinsic.k2 << ", p1 = " << this->cameraIntrinsic.p1 << ", p2 = " << this->cameraIntrinsic.p2 << std::endl;
    node = file["camera_intrinsic"];
    this->cameraIntrinsic.fx = static_cast<float>(node["fx"]);
    this->cameraIntrinsic.fy = static_cast<float>(node["fy"]);
    this->cameraIntrinsic.cx = static_cast<float>(node["cx"]);
    this->cameraIntrinsic.cy = static_cast<float>(node["cy"]);
    std::cout << "<Config> Camera intrinsic parameters: fx = " << this->cameraIntrinsic.fx << ", fy = " << this->cameraIntrinsic.fy <<
        ", cx = " << this->cameraIntrinsic.cx << ", cy = " << this->cameraIntrinsic.cy << std::endl;
    this->cameraIntrinsic.mapScale = static_cast<int>(file["camera_map_scale"]);
    std::cout << "<Config> Camera undistortion map scale is " << this->cameraIntrinsic.mapScale << std::endl;

    // 读取 IMU 噪声参数
    this->imuNoise.accel_n = file["acc_n"];
    this->imuNoise.accel_w = file["acc_w"];
    this->imuNoise.gyro_n = file["gyr_n"];
    this->imuNoise.gyro_w = file["gyr_w"];
    std::cout << "<Config> IMU noise: accel_n = " << this->imuNoise.accel_n << ", gyro_n = " << this->imuNoise.gyro_n <<
        ", accel_w = " << this->imuNoise.accel_w << ", gyro_w = " << this->imuNoise.gyro_w << std::endl;

    // 读取 IMU 和相机之间的相对位姿
    file["extrinsicRotation"] >> this->cameraExtrinsic.R_bc;
    file["extrinsicTranslation"] >> this->cameraExtrinsic.t_bc;

    // 读取系统管理参数
    this->windowSize = file["window_size"];
    std::cout << "<Config> Window size : " << this->windowSize << std::endl;

    // 读取前端追踪器参数
    this->trackerParams.featureNum = file["tracker_feature_num"];
    this->trackerParams.minDistance = file["tracker_min_distance"];

    // 读取初始化参数
    this->initParams.minIMUMotion = file["initial_min_imu_motion"];
    this->initParams.minMeanParallax = file["initial_min_mean_parallax"];
    this->initParams.minCovisibleNum = file["initial_min_covisible_num"];
    std::cout << "<Config> Initializer params : " << this->initParams.minIMUMotion << ", " << this->initParams.minMeanParallax <<
        ", " << this->initParams.minCovisibleNum << std::endl;

    // 读取紧耦合 VIO 滑动窗口参数
    this->optimizeParams.minMeanParallax = file["vio_min_mean_parallax"];
    this->optimizeParams.minCovisibleNum = file["vio_min_covisible_num"];
    this->optimizeParams.maxImuSumTime = file["vio_max_imu_integrate_time"];
    std::cout << "<Config> Estimatior params : " << this->optimizeParams.minMeanParallax << ", " << this->optimizeParams.minCovisibleNum <<
        ", " << this->optimizeParams.maxImuSumTime << std::endl;

    return true;
}


/* 将配置结果写入新的 config 文件 */
bool Config::CreateConfigFile(std::string &filePath) {
    // TODO
    return true;
}