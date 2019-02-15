#!/usr/bin/env python2.7
from __future__ import division
import roslib
import rospy
import tf
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import pdb

from message_filters import Subscriber, ApproximateTimeSynchronizer
class GT_cleaner:
    def __init__(self):

        
        self.init = [False, False]
        self.broadcaster = tf.TransformBroadcaster()
        self.mocap_pub = rospy.Publisher(
            '/gt_clean_odom', Odometry, queue_size=10)
        self.april_pub = rospy.Publisher(
            '/april_clean_odom', Odometry, queue_size=10)
        
        
        self.first_quat = None
        self.first_pos = np.array([0, 0, 0])
        self.prev_frame = [np.eye(4), np.eye(4)]
        self.first_frame = [np.eye(4),np.eye(4)]
        self.first_frame_inv = [np.eye(4),np.eye(4)]
        self.last_time = [rospy.Time.now(),rospy.Time.now()]

        self.sub = ApproximateTimeSynchronizer([Subscriber("/mocap/odom", Odometry),Subscriber("/apriltag_tracker/odom", Odometry)],100, 0.05)
        self.sub.registerCallback(self.callback)
        

    def callback(self, mocap_msg, odom_msg):
        for i,msg in enumerate([mocap_msg, odom_msg]):
            q = msg.pose.pose.orientation
            p = msg.pose.pose.position
            quat = np.array([q.x, q.y, q.z, q.w])
            pos = np.array([p.x, p.y, p.z])
            frame = tf.transformations.quaternion_matrix(quat)
            frame[:3, 3] = pos

            if i==1:
                frame = np.linalg.inv(frame) # Because track tag in body is the other way around

            if self.init[i] == False:
                self.last_time[i] = msg.header.stamp
                self.init[i] = True

                self.first_frame[i] = frame
                self.first_frame_inv[i] = np.linalg.inv(frame)
                continue

            dt = (msg.header.stamp - self.last_time[i]).to_sec()
            self.last_time[i] = msg.header.stamp
            frame_in_first = np.dot(self.first_frame_inv[i], frame)
            

            # add to path
            odom = Odometry()

            odom.header.frame_id = msg.header.frame_id
            odom.pose.pose.position.x = frame_in_first[0, 3]
            odom.pose.pose.position.y = frame_in_first[1, 3]
            odom.pose.pose.position.z = frame_in_first[2, 3]
            q = tf.transformations.quaternion_from_matrix(frame_in_first)
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]
            odom.header.stamp = msg.header.stamp

            #Now time for the velocities
            # Get the delta transform to obtain the velocities
            delta_frame = np.dot(np.linalg.inv(self.prev_frame[i]), frame_in_first)
            self.prev_frame[i] = frame_in_first
            # Linear part is easy
            odom.twist.twist.linear.x = delta_frame[0,3]/dt
            odom.twist.twist.linear.y = delta_frame[1,3]/dt
            odom.twist.twist.linear.z = delta_frame[2,3]/dt
            # For the angular velocity, we compute the angle axis
            result = tf.transformations.rotation_from_matrix(delta_frame)
            angle = result[0]
            direction = result[1]
            omega = direction * angle/dt
            odom.twist.twist.angular.x = omega[0]
            odom.twist.twist.angular.y = omega[1]
            odom.twist.twist.angular.z = omega[2]

            if i == 0:
                self.mocap_pub.publish(odom)
            else:
                self.april_pub.publish(odom)


if __name__ == '__main__':
    rospy.init_node('gt_cleaner', anonymous=True)
    cleaner_obj = GT_cleaner()
    rospy.spin()
