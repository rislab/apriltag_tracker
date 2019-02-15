#include <apriltag_tracker/AprilTagTracker.h>

bool AprilTagTracker::TrackTags(
    const cv::Mat &curr_img,
    std::vector<std::pair<Eigen::Matrix4f, uint>> &tag_tuples,
    cv::Mat &debug_img) {
  tag_tuples.clear();
  std::vector<AprilTags::TagDetection> detections = tag_detector_.extractTags(
      curr_img);
  if (detections.empty()) {
    return false;
  }
  if (params_.do_debug) {
    cv::cvtColor(curr_img, debug_img, CV_GRAY2RGB);
  }
  for (AprilTags::TagDetection &detection : detections) {
    Eigen::Matrix4f theta = Eigen::Matrix4f::Identity();
    cv::Mat rvec, tvec;
    // Get the relative transform of the camera in the tag frame
    detection.getRelativeRT(params_.tag_size, params_.K, params_.D, rvec, tvec);
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Map<Eigen::Matrix3f> R(reinterpret_cast<float *>(R_cv.data),
                                  R_cv.rows, R_cv.cols);
    Eigen::Map<Eigen::Vector3f> t(reinterpret_cast<float *>(tvec.data), 3);
    theta.block<3, 3>(0, 0) = R.cast<float>();
    theta.block<3, 1>(0, 3) = t.cast<float>();
    tag_tuples.push_back(std::make_pair(theta, detection.id));

    if (params_.do_debug) {
      detection.draw(debug_img);
    }
  }
  return true;
}

bool AprilTagTracker::DetectTags(
    const cv::Mat &curr_img,
    std::map<uint, std::vector<cv::Point2f>> &detected_points,
    cv::Mat &debug_img) {
  detected_points.clear();

  std::vector<AprilTags::TagDetection> detections = tag_detector_.extractTags(
      curr_img);
  if (detections.empty()) {
    return false;
  }
  if (params_.do_debug) {
    cv::cvtColor(curr_img, debug_img, CV_GRAY2RGB);
  }
  for (AprilTags::TagDetection &detection : detections) {
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(detection.p[0].first, detection.p[0].second));
    points.push_back(cv::Point2f(detection.p[1].first, detection.p[1].second));
    points.push_back(cv::Point2f(detection.p[2].first, detection.p[2].second));
    points.push_back(cv::Point2f(detection.p[3].first, detection.p[3].second));

    if (detection.good) {
      detected_points[detection.id] = points;
    }

    // Check if any of the points are beyond the dimensions of the image!
    cv::Rect rect(cv::Point(), curr_img.size());
    for (const auto &point : points) {
      if (!rect.contains(point)) {
        return false;
      }
    }

    if (params_.do_debug) {
      detection.draw(debug_img);
    }
  }
  return true;
}
