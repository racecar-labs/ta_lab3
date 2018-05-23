#!/usr/bin/env python

import time
import sys
import rospy
import rosbag
import numpy as np
import utils as Utils

import torch
import torch.utils.data
from torch.autograd import Variable

from nav_msgs.srv import GetMap
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped
#from threading import Lock

class MPPIController:

  def __init__(self, T, K, sigma=0.5, _lambda=0.5):
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset")) # Offset conversion param from rpm to speed
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain"))   # Gain conversion param from rpm to speed
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset")) # Offset conversion param from servo position to steering angle
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain")) # Gain conversion param from servo position to steering angle
    self.CAR_LENGTH = 0.33 

    self.last_pose = None
    # MPPI params
    self.sigma = sigma
    self.T = T
    self.K = K # number of sample rollouts
    self._lambda = _lambda

    self.goal = None

    self.costs = torch.zeros(self.K).cuda() # K
    self.weights = torch.zeros(self.K).cuda() # K
    self.ctrl = torch.zeros(self.T, 1, 2).cuda() # Tx2
    self.noise = None
    self.msgid = 0

    self.num_viz_paths = 40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    #self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this
      
    # This subscriber just caches the most recent servo position command
    self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
            PoseStamped, self.clicked_goal_cb, queue_size=1) # TODO get the correct topic

    #self.pose_sub  = rospy.Subscriber(rospy.get_param("~pose_topic","/pf/ta/viz/inferred_pose"),
    #        PoseStamped, self.mppi_cb, queue_size=1)

    self.ctrl_pub  = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav0"),
            AckermannDriveStamped, queue_size=2)
    self.path_pub  = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)

    # torch NN model
    model_name = rospy.get_param("~nn_model", "my_2layer.pt")
    #model_name = rospy.get_param("~nn_model", "mymodel2.pt")
    #model_name = rospy.get_param("~nn_model", "my_1layer.pt")
    self.model = torch.load(model_name)
    self.model.cuda()
    self.dtype = torch.cuda.FloatTensor
    print("Loading:", model_name)
    print("Model:\n",self.model)
    print("Torch Datatype:", self.dtype)

    # map
    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use    
    print("Map Information:\n",self.map_info)

    # Create numpy array representing map for later use
    self.map_height = map_msg.info.height
    self.map_width = map_msg.info.width
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible
    self.permissible_region = np.negative(self.permissible_region) # 0 is permissible, 1 is not
                                              
    print(np.sum(self.permissible_region))
    print("Percent Accessible Map:", 1.0-np.mean(self.permissible_region))
    print("making callback")

    self.pose_sub  = rospy.Subscriber("/pf/ta/viz/inferred_pose",
            PoseStamped, self.mppi_cb, queue_size=1)
    
  def clicked_goal_cb(self, msg):
    #self.state_lock.acquire()
    self.goal = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)])
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal)
    #self.state_lock.release()
    
  def running_cost(self, pose, goal, ctrl, noise):
    np_pose = pose.cpu().numpy() # Kx3
    Utils.world_to_map(np_pose, self.map_info)
    np_pose = np.round(np_pose).astype(int)
    np_pose[:,0] = np.clip(np_pose[:,0], 0, self.map_height-1)
    np_pose[:,1] = np.clip(np_pose[:,1], 0, self.map_width-1)

    bounds_check = 100*self.permissible_region[np_pose[:,0], np_pose[:,1]] # TODO check this bounds checks
    bounds_check = torch.from_numpy( bounds_check ).type(self.dtype)

    #delta = pose[:,0:2] - torch.from_numpy(goal[0:2]).type(self.dtype)
    delta = pose - torch.from_numpy(goal).type(self.dtype)
    pose_cost = 2.5*torch.sum(delta**2, 1)

    #pose_cost = 2.5*torch.sum((pose[:,0:2]-torch.from_numpy(goal[0:2]).type(self.dtype))**2, 1)
    ctrl_cost = torch.abs( self._lambda*( (noise*(1/self.sigma)).mm(ctrl.expand(1,2).transpose(0,1) )) )

    #print("costs:",torch.mean(pose_cost), torch.mean(ctrl_cost), torch.mean(bounds_check))

    return pose_cost + torch.squeeze(ctrl_cost) + bounds_check

  def final_cost(self, pose, state):
    # TODO make a better final_cost (running tally of bound check)
    np_pose = pose.cpu().numpy()
    Utils.world_to_map(np_pose, self.map_info)
    np_pose = np.round(np_pose).astype(int)
    bounds_check = 10000*self.permissible_region[np_pose[:,0], np_pose[:,1]]
    return torch.from_numpy(bounds_check).type(self.dtype)

  def mppi(self, model, ctrl, costs, init_pose, init_state):
    t0 = time.time()
    T = self.T
    K = self.K
    nn_input = torch.from_numpy( init_state ).type(self.dtype)
    nn_input = Variable(nn_input.repeat(K).resize_(K,5), requires_grad=False)

    ############################################# MPPI STUFF
    costs.zero_()
    noise = torch.randn(T, K, 2).cuda()*self.sigma # TxKx2
    noise[:,:,0].clamp_(-1.0, 1.0) # speed clamp
    noise[:,:,1].clamp_(-0.34, 0.34) # steering clamp
    noise_u = Variable(noise + ctrl, requires_grad=False)
    noise_u[:,:,0].data.clamp_(-1.0, 1.0) # speed clamp
    noise_u[:,:,1].data.clamp_(-0.34, 0.34) # steering clamp
    pose = torch.from_numpy( init_pose.astype('float32') ).type(self.dtype)
    pose = pose.repeat(K).resize_(K,3)
    next_pose = torch.zeros(pose.shape).cuda()

    poses = np.zeros((self.num_viz_paths, T, 3)) # visualization

    for i in range(0, T):
      #print("T:",i)
      nn_input[:,3:] = noise_u[i,:,:]
      nn_output = model(nn_input)
      #nn_output.div_(3) # TODO remove this scale
      pose.add_(nn_output.data) # variable to tensor
      if i==0:
          next_pose.set_(pose)
      nn_input[:,0:3] = nn_output

      poses[:,i,:] = pose[0:self.num_viz_paths, :]

      costs.add_( self.running_cost(pose, self.goal, ctrl[i,0,:], noise[i,:,:]) )

    #costs.add_( self.final_cost(pose, nn_input) )
    baseline, idx = torch.min(costs, 0)
    #print(baseline, torch.max(costs) )
    self.weights = costs - baseline
    self.weights.div_(-self._lambda)
    self.weights.exp_()
    eta = torch.sum(self.weights)
    #costs.sub_(baseline)
    #costs.div_(-self._lambda)
    #costs.exp_() # in place
    #eta = torch.sum(costs)
    #weights = costs.div(eta) # costs is now K weights
    self.weights.div_(eta)
    weights = self.weights
    self.noise = noise
    ctrl.add_( weights.expand(T,1,K).bmm(noise) ) # Tx2 = TxKx2 * K

    ctrl[:,:,0].clamp_(-1.0, 1.0) # speed clamp
    ctrl[:,:,1].clamp_(-0.34, 0.34) # steering clamp

    #costs.div_(eta) # costs is now K weights
    #self.ctrl.add_( costs.expand(T,1,K).bmm(noise) ) # Tx2 = TxKx2 * K

    # cost needs to be 1x1xK; 1x1xK * TxKx2 = Tx1x2
    run_ctrl = ctrl[0,:].cpu().numpy()[0]
    #print(run_ctrl)
    ctrl[0:-1] = ctrl[1:] # shift controls
    print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))

    return run_ctrl, poses, next_pose[idx,:].cpu().numpy()

  def mppi_cb(self, msg):
    #print("callback")
    #self.state_lock.acquire()
    if self.last_pose is None:
      self.last_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      self.goal = self.last_pose
      return

    curr_pose = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)])
    pose_dot = curr_pose - self.last_pose # get state
    self.last_pose = curr_pose

    estimate = np.array([pose_dot[0], pose_dot[1], pose_dot[2], 0.0, 0.0])

    run_ctrl, poses, pred_pose = mppi(self.model, self.ctrl, self.costs, curr_pose, estimate)

    self.send_controls( run_ctrl[0], run_ctrl[1] )

    self.visualize(poses)

    #self.state_lock.release()
  
  def send_controls(self, speed, steer):
    print("Speed:", speed, "Steering:", steer)
    ctrlmsg = AckermannDriveStamped()
    ctrlmsg.header.seq = self.msgid
    ctrlmsg.drive.steering_angle = steer 
    ctrlmsg.drive.speed = speed
    self.ctrl_pub.publish(ctrlmsg)
    self.msgid += 1

    # visualize
  def visualize(self, poses):
    if self.path_pub.get_num_connections() > 0:
      frame_id = 'map'
      for i in range(0, self.num_viz_paths):
        pa = Path()
        pa.header = Utils.make_header(frame_id)
        pa.poses = map(Utils.particle_to_posestamped, poses[i,:,:], [frame_id]*self.T)
        #print(pa)
        self.path_pub.publish(pa)

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):
  init_state = np.array([0.,0.,0.,0.,0.])
  init_pose = np.array([0.,0.,0.])
  mp.goal = goal
  print("Start:", init_pose)
  mp.ctrl.zero_()
  for i in range(0,N):
    run, poses, next_pose = mp.mppi(mp.model, mp.ctrl, mp.costs, init_pose, init_state)
    init_pose = next_pose[0,:]
    init_state[0:3] = next_pose[0,:]
    print("Now:", init_pose)
  print("End:", init_pose)
     
# Suggested main 
if __name__ == '__main__':

  T = 50
  K = 500
  sigma = 0.3
  _lambda = 0.1 #1.0

  # run
  #rospy.init_node("mppi_control", anonymous=True) # Initialize the node
  #mp = MPPIController(T, K, sigma, _lambda)
  #rospy.spin()

  # test
  mp = MPPIController(T, K, sigma, _lambda)
  init_pose = np.array([0.,0.,0.])
  init_state = np.array([0.,0.,0.,0.,0.])
  mp.goal = init_pose

