#include <include/estimator/estimator.hpp>
#include <unistd.h>

/* 绘制所有特征点和所有关键帧之间的空间位置关系 */
void VIOMono::Visualize(void) {
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    this->VisualizeStart(s_cam, d_cam);
    while (this->VisualizeShouldQuit() == false) {
        this->VisualizeOnce(s_cam, d_cam);
        usleep(5000);
    }
    this->VisualizeStop();
}


/* 构建绘图窗口 */
void VIOMono::VisualizeStart(pangolin::OpenGlRenderState &s_cam, pangolin::View &d_cam) {
    // 创建 Pangolin 窗口
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        // pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
        pangolin::ModelViewLookAt(0.1, -0.5, 0.1, 0.2, 0, 0, 0.0, 0.0, 1.5)
    );

    d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
}


/* 读取数据，进行一次绘制 */
void VIOMono::VisualizeOnce(pangolin::OpenGlRenderState &s_cam, pangolin::View &d_cam) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
    glColor3f(0, 0, 1);
    pangolin::glDrawAxis(3);

    if (this->frameManager->frames.empty() || this->landmarkManager->landmarks.empty()) {
        pangolin::FinishFrame();
        return;
    }

    // 绘制关键帧的位姿点
    glColor3f(0, 1.0, 0);
    glPointSize(5);
    glBegin(GL_POINTS);
    for (auto &frame: this->frameManager->frames) {
        if (frame->ID == this->frameManager->frames.back()->ID) {
            continue;
        }
        auto pose = frame->t_wc;
        glVertex3d(pose.x(), pose.y(), pose.z());
    }
    glEnd();
    glColor3f(0, 1.0, 1.0);
    glPointSize(5);
    glBegin(GL_POINTS);
    for (auto &frame: this->frameManager->frames) {
        if (frame->ID == this->frameManager->frames.back()->ID) {
            continue;
        }
        auto pose = frame->t_wb;
        glVertex3d(pose.x(), pose.y(), pose.z());
    }
    glEnd();

    // 最新一帧的位姿点用红色
    glColor3f(1.0, 0, 0);
    glPointSize(10);
    glBegin(GL_POINTS);
    auto pose = this->frameManager->frames.back()->t_wc;
    glVertex3d(pose.x(), pose.y(), pose.z());
    pose = this->frameManager->frames.back()->t_wb;
    glVertex3d(pose.x(), pose.y(), pose.z());
    glEnd();

    // 绘制关键帧位姿的 XYZ 轴（分别对应 RGB）
    glColor3f(0, 0, 0);
    glLineWidth(1);
    glBegin(GL_LINES);
    for (auto it = this->frameManager->frames.begin(); it != this->frameManager->frames.end(); ++it) {
        Eigen::Vector3f temp;
        Eigen::Vector3f addx = (*it)->q_wc.toRotationMatrix() * Eigen::Vector3f(0.05, 0, 0);
        Eigen::Vector3f addy = (*it)->q_wc.toRotationMatrix() * Eigen::Vector3f(0, 0.05, 0);
        Eigen::Vector3f addz = (*it)->q_wc.toRotationMatrix() * Eigen::Vector3f(0, 0, 0.05);
        glColor3f(0.9, 0, 0);
        temp = (*it)->t_wc + addx;
        glVertex3d((*it)->t_wc.x(), (*it)->t_wc.y(), (*it)->t_wc.z());
        glVertex3d(temp.x(), temp.y(), temp.z());
        glColor3f(0, 0.9, 0);
        temp = (*it)->t_wc + addy;
        glVertex3d((*it)->t_wc.x(), (*it)->t_wc.y(), (*it)->t_wc.z());
        glVertex3d(temp.x(), temp.y(), temp.z());
        glColor3f(0, 0, 0.9);
        temp = (*it)->t_wc + addz;
        glVertex3d((*it)->t_wc.x(), (*it)->t_wc.y(), (*it)->t_wc.z());
        glVertex3d(temp.x(), temp.y(), temp.z());

        addx = (*it)->q_wb.toRotationMatrix() * Eigen::Vector3f(0.05, 0, 0);
        addy = (*it)->q_wb.toRotationMatrix() * Eigen::Vector3f(0, 0.05, 0);
        addz = (*it)->q_wb.toRotationMatrix() * Eigen::Vector3f(0, 0, 0.05);
        glColor3f(0.9, 0, 0);
        temp = (*it)->t_wb + addx;
        glVertex3d((*it)->t_wb.x(), (*it)->t_wb.y(), (*it)->t_wb.z());
        glVertex3d(temp.x(), temp.y(), temp.z());
        glColor3f(0, 0.9, 0);
        temp = (*it)->t_wb + addy;
        glVertex3d((*it)->t_wb.x(), (*it)->t_wb.y(), (*it)->t_wb.z());
        glVertex3d(temp.x(), temp.y(), temp.z());
        glColor3f(0, 0, 0.9);
        temp = (*it)->t_wb + addz;
        glVertex3d((*it)->t_wb.x(), (*it)->t_wb.y(), (*it)->t_wb.z());
        glVertex3d(temp.x(), temp.y(), temp.z());
    }
    glEnd();

    // 绘制特征点
    glColor3f(0, 0, 1.0);
    glPointSize(4);
    glBegin(GL_POINTS);
    for (auto &item : this->landmarkManager->landmarks) {
        auto &landmark = item.second;
        if (landmark->isSolved != Landmark::SolveStatus::YES) {
            glColor3f(1.0, 0, 0);
        } else {
            glColor3f(0, 0, 1.0);
        }
        glVertex3d(landmark->p_w.x(), landmark->p_w.y(), landmark->p_w.z());
    }
    glEnd();

    // 绘制滑动窗口内的轨迹
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
    for (auto it = this->frameManager->frames.begin(); std::next(it) != this->frameManager->frames.end(); ++it) {
        auto next = std::next(it);
        glVertex3d((*it)->t_wc.x(), (*it)->t_wc.y(), (*it)->t_wc.z());
        glVertex3d((*next)->t_wc.x(), (*next)->t_wc.y(), (*next)->t_wc.z());
        glVertex3d((*it)->t_wb.x(), (*it)->t_wb.y(), (*it)->t_wb.z());
        glVertex3d((*next)->t_wb.x(), (*next)->t_wb.y(), (*next)->t_wb.z());
    }
    glEnd();

    // 绘制全局轨迹
    glColor3f(0.2, 0.2, 0.2);
    glLineWidth(2);
    glBegin(GL_LINES);
    for (auto it = this->poses.begin(); std::next(it) != this->poses.end(); ++it) {
        auto next = std::next(it);
        glVertex3d((*it)->t_wb.x(), (*it)->t_wb.y(), (*it)->t_wb.z());
        glVertex3d((*next)->t_wb.x(), (*next)->t_wb.y(), (*next)->t_wb.z());
    }
    glEnd();

    pangolin::FinishFrame();
}


/* 关闭绘图窗口 */
void VIOMono::VisualizeStop(void) {
    pangolin::DestroyWindow("Trajectory Viewer");
}


/* 判断绘图窗口是否要关闭 */
bool VIOMono::VisualizeShouldQuit(void) {
    return pangolin::ShouldQuit();
}