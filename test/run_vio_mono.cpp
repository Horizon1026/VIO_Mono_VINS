#include <iostream>
#include <thread>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>

#include <include/estimator/estimator.hpp>

VIOMono vio_mono;
std::list<std::pair<double, Eigen::Vector3d>> gtPos;
double firstTimeStamp = 0;

/* 订阅图像信息的回调函数 */
void SubscribeImageCallBack(const sensor_msgs::ImageConstPtr &img_msg) {
    // 定义指向图像的指针ptr，从图像话题的消息中提取出图像信息
    cv_bridge::CvImagePtr imagePtr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        imagePtr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else {
        imagePtr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }

    // 消息类型转换，并压入到队列中
    std::shared_ptr<ImageMessage> imageMessage(new ImageMessage());
    imageMessage->timeStamp = img_msg->header.stamp.toSec();
    imageMessage->image = imagePtr->image.clone();
    vio_mono.PushImageMessage(imageMessage);
}

/* 订阅 IMU 信息的回调函数 */
void SubscribeIMUCallBack(const sensor_msgs::ImuConstPtr &imu_msg) {
    // 消息类型转换，并压入到队列中
    std::shared_ptr<IMUMessage> imuMessage(new IMUMessage());
    imuMessage->timeStamp = imu_msg->header.stamp.toSec();
    imuMessage->accel << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;
    imuMessage->gyro << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
    vio_mono.PushIMUMessage(imuMessage);
}

/* 订阅轨迹真值的回调函数 */
void SubscribeGroundTruthCallBack(const geometry_msgs::PointStampedConstPtr &gt_msg) {
    if (gtPos.empty()) {
        firstTimeStamp = gt_msg->header.stamp.toSec();
        gtPos.emplace_back(std::make_pair(0, Eigen::Vector3d(gt_msg->point.x, gt_msg->point.y, gt_msg->point.z)));
    } else {
        double time = gt_msg->header.stamp.toSec() - firstTimeStamp;
        gtPos.emplace_back(std::make_pair(time, Eigen::Vector3d(gt_msg->point.x, gt_msg->point.y, gt_msg->point.z)));
    }
}

/* 保存真值轨迹 */
void SaveGroundTruth() {
    std::ofstream file;
    std::string path = "/home/horizon/slam_ws/my_code_onROS/[workspace]/src/vio_mono/saved_data/gt.txt";
    file.open(path.c_str());
    if (file.is_open()) {
        for (auto it = gtPos.begin(); it != gtPos.end(); ++it) {
            file << it->first << " "
                 << it->second.x() << " "
                 << it->second.y() << " "
                 << it->second.z() << std::endl;
        }
    }
}


int main(int argc, char **argv) {
    // ROS 节点创建和初始化
    ros::init(argc, argv, "vio_mono");
    ros::NodeHandle n("~");

    // 订阅图像消息和 IMU 消息
    ros::Subscriber sub_image = n.subscribe<sensor_msgs::Image>("/cam0/image_raw", 1000, &SubscribeImageCallBack);
    ros::Subscriber sub_imu = n.subscribe<sensor_msgs::Imu>("/imu0", 10000, &SubscribeIMUCallBack);
    ros::Subscriber sub_gt = n.subscribe<geometry_msgs::PointStamped>("/leica/position", 100, &SubscribeGroundTruthCallBack);

    // 确定配置文件路径
    std::string configFile = "/home/horizon/slam_ws/my_code_onROS/[workspace]/src/vio_mono/config/euroc.yaml";

    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    bool visualizeFlag = true;

    // 初始化 VIO 并启动
    vio_mono.Initialize(configFile);
    if (visualizeFlag) {
        vio_mono.VisualizeStart(s_cam, d_cam);
    }
    ROS_INFO("test vio mono start.");
    size_t timeOut = 1000;
    size_t timeCnt = 0;
    size_t cnt = 0;
    while (timeCnt < timeOut) {
        bool res = vio_mono.RunOnce();
        
        if (visualizeFlag) {
            if (vio_mono.VisualizeShouldQuit() == false) {
                vio_mono.VisualizeOnce(s_cam, d_cam);
            } else {
                break;
            }
        }
        if (res == false) {
            usleep(5000);
            ++timeCnt;
        } else {
            timeCnt = 0;
            ++cnt;
        }

        ros::spinOnce();

        // TODO : 仅仅单步运行有限次数
        // if (cnt > 500) {
        //     break;
        // }
    }
    if (visualizeFlag) {
        while (vio_mono.VisualizeShouldQuit() == false) {
            vio_mono.VisualizeOnce(s_cam, d_cam);
            usleep(5000);
        }
        vio_mono.VisualizeStop();
    }
    vio_mono.SavePosesAsFile();
    vio_mono.SaveCostTimesAsFile();
    SaveGroundTruth();
    ROS_INFO("test vio mono timeout, stop.");

    ros::spinOnce();
    return 0;
}