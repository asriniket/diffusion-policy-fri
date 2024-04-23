import rospy
import argparse
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped, Pose
import PyKDL
from tf_conversions import posemath
from intera_interface import Limb
import intera_interface
import numpy as np 
import math

class SawyerEnv():
    def __init__(self) -> None:
        rospy.init_node('go_to_cartesian_pose_py')
        self.limb = Limb()
        self.tip_name = "right_hand"
    
    def reset(self):
        pass

    # def go_to_cartestian(self, x, y, z, q1, q2, q3, qw, tip_name="right_hand"):
    def go_to_cartesian(self, x1, y1, z1, q1, q2, q3, q4, tip_name="right_hand"):        
        # TEST: go to current pose

        # TODO: make this go to commanded pose (pass args)

        # TODO: make this pose relative (x,y,z) should be relative to the current (x,y,z) 

        current_pose = self.limb.endpoint_pose()
        x = current_pose['position'].x
        y = current_pose['position'].y
        z = current_pose['position'].z
        qx = current_pose['orientation'].x
        qy = current_pose['orientation'].y
        qz = current_pose['orientation'].z
        qw = current_pose['orientation'].w

        # computing the current frame so that later we can compute relative frame
        current_position_PYKDL = PyKDL.Vector(x, y, z)
        current_orientation_PYKDL = PyKDL.Rotation.Quaternion(qw, qx, qy, qz) # assuming order is w, x, y,z
        current_frame = PyKDL.Frame(current_orientation_PYKDL, current_position_PYKDL)
  
        # set up traj options
        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.JOINT
        traj = MotionTrajectory(trajectory_options=traj_options, limb=self.limb)
        
        # Configure waypoint options
        wpt_opts = MotionWaypointOptions(
            max_linear_speed=0.5,
            max_linear_accel=0.2,
            max_rotational_speed=1.0,
            max_rotational_accel=0.5,
            max_joint_speed_ratio=1.0
        )
        
        # Create waypoint
        waypoint = MotionWaypoint(options=wpt_opts, limb=self.limb)
        pose = Pose()
        # pose.position.x = x1 
        # pose.position.y = y1
        # pose.position.z = z1
        # pose.orientation.x = q1
        # pose.orientation.y = q2
        # pose.orientation.z = q3
        # pose.orientation.w = q4

        # computing target pose

        target_rot = PyKDL.Rotation.Quaternion(q4, q1, q2, q3)
        target_pos = PyKDL.Vector(x1, y1, z1)
        target_frame = PyKDL.Frame(target_rot, target_pos)

        # Computing the relative frame
        relative_frame = current_frame.Inverse() * target_frame
        pose = posemath.toMsg(relative_frame)
        
        # Set the Cartesian pose
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose
        joint_angles = self.limb.joint_ordered_angles()  # Get current joint angles for context
        waypoint.set_joint_angles(joint_angles, tip_name)

        waypoint.set_cartesian_pose(pose_stamped, self.tip_name, joint_angles)

        # Append and send trajectory
        traj.append_waypoint(waypoint.to_msg())
        result = traj.send_trajectory(timeout=None)  # Consider specifying a timeout
        print("Trajectory result:", result)
        return result

    def step(self, action):
        # TODO : here apply the action, use go_to_cartesian 
        # TODO: return the observation
        new_coords = self.go_to_cartesian(action[0], action[1], action[2], action[3], action[4], action[5])
        return new_coords
        
def run_episode(policy, env):
    pass


if __name__ == '__main__':
    env = SawyerEnv()
    while True:
        env.go_to_cartesian()