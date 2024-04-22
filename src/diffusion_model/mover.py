import rospy

class Mover():
    def __init__(self, limb):
        self.limb = limb
        self.x = limb.get_pose()
    
    def go_to_cartesian_pose(x, y, z, qx, qy, qz, qw):
