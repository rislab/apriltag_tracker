cmake_minimum_required(VERSION 2.8.3)
project(apriltag_tracker)

find_package(catkin REQUIRED COMPONENTS
  roscpp image_transport cv_bridge image_geometry
  apriltags
  std_msgs tf2_ros rospy geometry_msgs tf
  )
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(cmake_modules)
find_package(Eigen3)

catkin_python_setup()

add_message_files(DIRECTORY msg FILES Apriltag.msg Apriltags.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ffast-math -Ofast -Wall")

catkin_package(
  INCLUDE_DIRS
  include ${EIGEN3_INCLUDE_DIR}
  CATKIN_DEPENDS
  roscpp
  cv_bridge
  sensor_msgs
  image_transport
  std_msgs
  geometry_msgs
  apriltags
  rospy
  tf
  tf2_ros
  )


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  )

# detector node
add_executable(apriltag_tracker_node
  src/apriltag_tracker_node.cpp)
target_link_libraries(apriltag_tracker_node
  ${catkin_LIBRARIES}
  yaml-cpp
  )

add_executable(apriltag_detector_node
  src/apriltag_detector_node.cpp)
target_link_libraries(apriltag_detector_node
  ${catkin_LIBRARIES}
  yaml-cpp
  )

# dependencies
add_dependencies(apriltag_tracker_node
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_gencpp
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  )
add_dependencies(apriltag_detector_node
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_gencpp
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  )


## Python wrapper
add_library(_AprilTagTracker src/apriltag_tracker_python_wrapper.cpp)
target_link_libraries(_AprilTagTracker ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(_AprilTagTracker PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
  )

## INSTALL


install(TARGETS apriltag_tracker_node apriltag_detector_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)
