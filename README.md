A simple ROS tracker wrapper for the catkinized AprilTag library [https://github.com/icoderaven/apriltags_catkin]

Publishes all detected AprilTags in the subscribed image stream in custom messages defined in msg/ folder.

Also includes a Boost Python Wrapper for the same to use within python seamlessly.

Tag parameters need to be specified in a yaml format. An example is included in config/params.yaml

Two executables exist:
apriltag_detector_node : To only publish the tag messages.
apriltag_tracker_node : To publish the PnP output of the tag in the camera frame.

Enabling the debug parameter publishes a debug image with an overlay of the tags.