#include <apriltag_tracker/AprilTagTrackerRosWrapper.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "apriltag_detector_node");
  ros::NodeHandle nh("~");
  float rate;
  nh.param<float>("rate", rate, 60);
  std::string params_file;
  nh.param<std::string>("params_file", params_file, "../config/params.yaml");
  std::cout << "[AprilTagDetectorRosWrapper] Rate : " << rate << "\n";
  AprilTagTrackerRosWrapper wrapper(nh, params_file);
  ros::Rate r(rate);
  do {
    wrapper.publishTags();
    ros::spinOnce();
    r.sleep();
  } while (ros::ok());

  return EXIT_SUCCESS;
}
