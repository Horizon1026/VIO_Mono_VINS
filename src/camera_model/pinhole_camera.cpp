#include <include/camera_model/pinhole_camera.hpp>

/* 相机投影模型参数初始化 */
void PinholeCamera::InitParams(float fx, float fy, float cx, float cy, int mapScale) {
    this->fx = fx;
    this->fy = fy;
    this->cx = cx;
    this->cy = cy;
    this->CreateUndistortedMap(mapScale);
}


/* 由像素平面坐标，变化为归一化平面坐标 */
Eigen::Vector2f PinholeCamera::PixelPlane_NormlizedPlane(cv::Point2f p) {
    return Eigen::Vector2f((p.x - this->cx) / this->fx,
                           (p.y - this->cy) / this->fy);
}


/* 由归一化平面坐标，变化为像素平面坐标 */
cv::Point2f PinholeCamera::NormlizedPlane_PixelPlane(Eigen::Vector2f p) {
    return cv::Point2f(p[0] * this->fx + this->cx,
                       p[1] * this->fy + this->cy);
}


/* 由非畸变像素平面坐标，变化为畸变像素平面坐标 */
cv::Point2f PinholeCamera::Undistortion_Distortion(cv::Point2f p) {
    //将输入的像素坐标转化到归一化平面上，p.x是u，p.y是v
    double x = (p.x - cx) / fx;
    double y = (p.y - cy) / fy;

    //根据畸变参数计算畸变后的归一化平面坐标
    cv::Point2f p_distorted;
    double r_2 = x * x + y * y;     //计算r的平方
    p_distorted.x = x * (1 + k1 * r_2 + k2 * r_2 * r_2) + 2 * p1 * x * y + p2 * (r_2 + 2 * x * x);
    p_distorted.y = y * (1 + k1 * r_2 + k2 * r_2 * r_2) + 2 * p2 * x * y + p1 * (r_2 + 2 * y * y);

    //将畸变后的归一化平面坐标转化成像素坐标
    p_distorted.x = fx * p_distorted.x + cx;
    p_distorted.y = fy * p_distorted.y + cy;

    return p_distorted;
}