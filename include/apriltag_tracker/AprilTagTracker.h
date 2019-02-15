#pragma once

#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <apriltag_tracker/Apriltags.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <apriltags/TagDetector.h>

#include <apriltags/Tag36h11.h>


#include <yaml-cpp/yaml.h>

struct apriltag_params {
  float tag_size;
  int id;
  cv::Mat K;
  cv::Mat D;
  bool do_debug;
  Eigen::MatrixXf cam_in_body, cam_in_body_inv;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  apriltag_params()
      : tag_size(1.0),
        id(0),
        K(cv::Matx33f::eye()),
        D(cv::Mat::zeros(4, 1, CV_32FC1)),
        do_debug(false),
        cam_in_body(Eigen::Matrix4f::Identity()),
        cam_in_body_inv(Eigen::Matrix4f::Identity()) {
  }
  void read_config(std::string params_file) {
    std::cout << "[AprilTagParams] Reading from file " << params_file << "\n";
    try {
      YAML::Node params = YAML::LoadFile(params_file);
      YAML::Node apriltag_params_node = params["apriltag_tracker_params"];
      if (!apriltag_params_node) {
        std::cerr << "[AprilTagParams] Could not read apriltag_tracker_params!";
        exit(-1);
      } else {
        tag_size = apriltag_params_node["tag_size"].as<float>();
        id = apriltag_params_node["id"].as<int>();
        do_debug = apriltag_params_node["do_debug"].as<bool>();

        YAML::Node K_node = apriltag_params_node["K"];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            K.at<float>(i, j) = K_node[i * 3 + j].as<float>();
          }
        }
        YAML::Node D_node = apriltag_params_node["D"];
        for (int i = 0; i < 4; ++i) {
          D.at<float>(i) = D_node[i].as<float>();
        }
        YAML::Node cam_in_body_node = apriltag_params_node["cam_in_body"];
        if (cam_in_body_node) {
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              cam_in_body(i, j) =
                  cam_in_body_node[i * 4 + j].as<float>();
            }
          }
          cam_in_body_inv = cam_in_body.inverse();
        }
      }
    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      exit(-1);
    }
  }
};

std::ostream &operator<<(std::ostream &o, const apriltag_params &params) {
  o << "tag_size: " << params.tag_size << "\nid: " << params.id << "\nK: "
    << params.K << "\nD: " << params.D << "\n"
    << "\ndo_debug: "
    << params.do_debug << "\ncam_in_body: \n"
    << params.cam_in_body << "\n";
  return o;
}

class AprilTagTracker {
 private:
  AprilTags::TagDetector tag_detector_;
  apriltag_params params_;

 public:
  AprilTagTracker(std::string params_file)
      : tag_detector_(AprilTags::tagCodes36h11) {
    params_.read_config(params_file);
    std::cout << "[AprilTagParams] AprilTagTracker params : \n"
              << params_;
  }

  void ReadConfig(std::string params_file) {
    params_.read_config(params_file);
    std::cout << "[AprilTagParams] AprilTagTracker params : \n"
              << params_;
  }

  template <typename T>
  bool TrackTag(const cv::Mat &curr_img, Eigen::MatrixBase<T> &theta,
                cv::Mat &debug_img);

  template <typename T>
  bool TrackTagInBody(const cv::Mat &curr_img, Eigen::MatrixBase<T> &theta,
                      cv::Mat &debug_img);

  bool TrackTags(const cv::Mat &curr_img,
                 std::vector<std::pair<Eigen::Matrix4f, uint>> &detections,
                 cv::Mat &debug_img);

  bool DetectTags(const cv::Mat &curr_img,
                  std::map<uint, std::vector<cv::Point2f>> &detected_points,
                  cv::Mat &debug_img);

  apriltag_params get_params() {
    return params_;
  }
};

template <typename T>
bool AprilTagTracker::TrackTagInBody(const cv::Mat &curr_img,
                                     Eigen::MatrixBase<T> &theta,
                                     cv::Mat &debug_img) {
  Eigen::Matrix4f theta_in_cam = Eigen::Matrix4f::Identity();
  bool return_val = TrackTag(curr_img, theta_in_cam, debug_img);
  theta = params_.cam_in_body * theta_in_cam;
  return return_val;
}

template <typename T>
bool AprilTagTracker::TrackTag(const cv::Mat &curr_img,
                               Eigen::MatrixBase<T> &theta,
                               cv::Mat &debug_img) {
  if (params_.do_debug) {
    cv::cvtColor(curr_img, debug_img, CV_GRAY2RGB);
  }
  // Detect tags
  std::vector<AprilTags::TagDetection> detections = tag_detector_.extractTags(
      curr_img);
  // Process detection
  if (!detections.empty()) {
    // Actual processing
    for (AprilTags::TagDetection &detection : detections) {
      if (detection.id == params_.id) {
        if (params_.do_debug) {
          detection.draw(debug_img);
        }
        cv::Mat rvec, tvec;
        detection.getRelativeRT(params_.tag_size, params_.K, params_.D, rvec,
                                tvec);
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Map<Eigen::Matrix3f> R(reinterpret_cast<float *>(R_cv.data),
                                      R_cv.rows, R_cv.cols);
        Eigen::Map<Eigen::Vector3f> t(reinterpret_cast<float *>(tvec.data), 3);
        theta.template block<3, 3>(0, 0) = R.cast<typename T::Scalar>();
        theta.template block<3, 1>(0, 3) = t.cast<typename T::Scalar>();
        return true;
      }
    }
  }
  return false;
}

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
