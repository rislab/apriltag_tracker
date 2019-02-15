#pragma once

#include <apriltag_tracker/AprilTagTracker.h>

#include <ros/publisher.h>
#include <ros/ros.h>
#include <ros/subscriber.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>

#include <apriltag_tracker/Apriltags.h>

#include <memory>

class AprilTagTrackerRosWrapper {
 private:
  std::shared_ptr<ros::NodeHandle> nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  ros::Subscriber img_sub_;
  ros::Publisher odom_pub_, tags_pub_;
  std::string cam_frame_id_;

  image_transport::Publisher image_pub_;

  std::shared_ptr<AprilTagTracker> tracker_;

  cv::Mat curr_img_, prev_img_;
  double curr_t_;
  Eigen::MatrixXf prev_theta_;
  bool initialized_, new_data_;
  tf::TransformBroadcaster br_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AprilTagTrackerRosWrapper(ros::NodeHandle nh, std::string params_file)
      : curr_t_(0.0),
        prev_theta_(Eigen::Matrix4f::Identity()),
        new_data_(false) {
    nh_ = std::make_shared<ros::NodeHandle>(nh);
    it_ = std::make_shared<image_transport::ImageTransport>(nh);
    tracker_ = std::make_shared<AprilTagTracker>(params_file);
    InitializeParams();
    InitializeCallbacks();
    InitializePublishers();
    std::cout << "[AprilTagTrackerRosWrapper] Initialization done!\n";
  }

  AprilTagTrackerRosWrapper(std::string params_file)
      : curr_t_(0.0),
        prev_theta_(Eigen::Matrix4f::Identity()),
        new_data_(false) {
    tracker_ = std::make_shared<AprilTagTracker>(params_file);
    InitializeParams();
  }

  void InitializeParams() {
    cam_frame_id_ = std::string("apriltag_detector");
  }

  void InitializeCallbacks() {
    img_sub_ = nh_->subscribe("image_topic", 100,
                              &AprilTagTrackerRosWrapper::callback, this);
  }

  void InitializePublishers() {
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>("odom", 10, false);
    tags_pub_ = nh_->advertise<apriltag_tracker::Apriltags>("tags", 10, false);
    image_pub_ = it_->advertise("debug_image", 1);
  }

  apriltag_params get_params() {
    return tracker_->get_params();
  }

  void callback(const sensor_msgs::Image::ConstPtr &image_msg) {
    curr_img_.copyTo(prev_img_);
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(image_msg, "mono8");
    } catch (cv_bridge::Exception &e) {
      throw std::runtime_error(
          std::string("cv_bridge exception: ") + std::string(e.what()));
    }
    cv_ptr->image.copyTo(curr_img_);
    curr_t_ = image_msg->header.stamp.toSec();
    if (!initialized_ && prev_img_.cols > 0) {
      initialized_ = true;
      std::cout << "[AprilTagTrackerRosWrapper] Initialization done!\n";
    }
    new_data_ = true;
  }

  bool get_current_synchronized_data() {
    //@todo: Do this with a lock?
    return initialized_ && new_data_;
  }

  void trackInBody() {
    cv::Mat debug_img;
    Eigen::Matrix4f theta = Eigen::Matrix4f::Identity();
    apriltag_params params = tracker_->get_params();
    // theta.block<3,3>(0,0) = (prev_rot_.inverse() * rot_).matrix();
    if (get_current_synchronized_data()) {
      if (tracker_->TrackTagInBody(curr_img_, theta, debug_img)) {
        new_data_ = false;
        publishOdom(theta);
        if (params.do_debug) {
          sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                             "rgb8", debug_img)
                                              .toImageMsg();
          image_pub_.publish(img_msg);
        }
      }
    }
  }

  void track() {
    cv::Mat debug_img;
    Eigen::Matrix4f theta = Eigen::Matrix4f::Identity();
    apriltag_params params = tracker_->get_params();
    // theta.block<3,3>(0,0) = (prev_rot_.inverse() * rot_).matrix();
    if (get_current_synchronized_data()) {
      if (tracker_->TrackTag(curr_img_, theta, debug_img)) {
        new_data_ = false;
        publishOdom(theta);
        if (params.do_debug) {
          sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                             "rgb8", debug_img)
                                              .toImageMsg();
          image_pub_.publish(img_msg);
        }
      }
    }
  }

  void trackTags(std::vector<std::pair<Eigen::Matrix4f, uint>> &tag_tuples) {
    cv::Mat debug_img;
    if (get_current_synchronized_data()) {
      tracker_->TrackTags(curr_img_, tag_tuples, debug_img);
      apriltag_params params = tracker_->get_params();
      if (params.do_debug) {
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                           "rgb8", debug_img)
                                            .toImageMsg();
        image_pub_.publish(img_msg);
      }
    }
  }

  void detectTags(std::map<uint, std::vector<cv::Point2f>> &detections,
                  cv::Mat &debug_img) {
    if (get_current_synchronized_data()) {
      tracker_->DetectTags(curr_img_, detections, debug_img);
    }
  }

  void publishTags() {
    std::map<uint, std::vector<cv::Point2f>> detections;
    cv::Mat debug_img;
    if (get_current_synchronized_data()) {
      tracker_->DetectTags(curr_img_, detections, debug_img);
      apriltag_tracker::Apriltags tags_msg;
      tags_msg.header.stamp = ros::Time(curr_t_);
      for (auto tag : detections) {
        apriltag_tracker::Apriltag tag_msg;
        tag_msg.id = tag.first;
        for (auto corner : tag.second) {
          geometry_msgs::Point point;
          point.x = corner.x;
          point.y = corner.y;
          tag_msg.corners.push_back(point);
        }
        tags_msg.apriltags.push_back(tag_msg);
      }
      tags_pub_.publish(tags_msg);
      apriltag_params params = tracker_->get_params();
      if (params.do_debug) {
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                           "rgb8", debug_img)
                                            .toImageMsg();
        image_pub_.publish(img_msg);
      }
    }
  }

  void publishOdom(const Eigen::Matrix4f &theta) {
    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time(curr_t_);
    msg.header.frame_id = cam_frame_id_;
    msg.pose.pose.position.x = theta(0, 3);
    msg.pose.pose.position.y = theta(1, 3);
    msg.pose.pose.position.z = theta(2, 3);
    Eigen::Quaternionf q(theta.block<3, 3>(0, 0));
    msg.pose.pose.orientation.w = q.w();
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    msg.header.frame_id = "tag";
    msg.child_frame_id = "camera";
    odom_pub_.publish(msg);

    tf::Transform rel_transform;
    rel_transform.setOrigin(tf::Vector3(theta(0, 3), theta(1, 3), theta(2, 3)));
    rel_transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    br_.sendTransform(
        tf::StampedTransform(rel_transform, msg.header.stamp, "camera", "tag"));
  }
};
