# cmake工程最低版本为3.10
cmake_minimum_required( VERSION 3.10 )
# cmake工程名
project(vio_mono)
# 设置编译类型为发布
set(CMAKE_BUILD_TYPE "Release")
# 添加c++ 17标准的支持
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

# 添加 catkin 库
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  std_msgs
  cv_bridge
)
catkin_package()
include_directories( ${catkin_INCLUDE_DIRS} )

# 寻找Eigen库并添加其路径
include_directories( "/usr/include/eigen3" )

# 寻找 TBB 库
find_package( TBB REQUIRED )

# 寻找到OpenCV的库并添加OpenCV的路径
set(OpenCV_INCLUDE_DIRS "/home/horizon/opencvLib/opencv_3.4.14/include")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 寻找 Pangolin 库并添加其路径
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# 添加工程路径
include_directories( ${PROJECT_SOURCE_DIR} )

# 添加输出的可执行文件
add_executable( test_vio_mono
    test/run_vio_mono.cpp

    src/estimator/estimator.cpp
    src/estimator/estimator_support.cpp
    src/estimator/estimator_visualize.cpp
    src/estimator/support/epipolar_geometry.cpp
    src/estimator/support/perspective_n_point.cpp
    src/estimator/initialize/initialize.cpp
    src/estimator/initialize/initialize_visual_only.cpp
    src/estimator/initialize/initialize_align.cpp
    src/estimator/optimize/optimize.cpp
    src/estimator/optimize/marginalize.cpp

    src/interface/interface.cpp

    src/config_parameters/config.cpp

    src/camera_model/camera_base.cpp
    src/camera_model/pinhole_camera.cpp

    src/measurement/feature_tracker.cpp
    src/measurement/frame_manager.cpp
    src/measurement/imu_preintegration.cpp
    src/measurement/landmark_manager.cpp
)

# 为可执行文件链接相关的库
target_link_libraries( test_vio_mono
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS} 
    pangolin GL GLEW    # pangolin和OpenGL的库
    TBB::tbb
)