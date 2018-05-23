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

SPEED_TO_ERPM_OFFSET     = 0.0
SPEED_TO_ERPM_GAIN       = 4614.0
STEERING_TO_SERVO_OFFSET = 0.5304
STEERING_TO_SERVO_GAIN   = -1.2135

bag = rosbag.Bag(sys.argv[1])
tt = bag.get_type_and_topic_info()
t1='/vesc/sensors/core'
t2='/vesc/sensors/servo_position_command'
t3='/pf/ta/viz/inferred_pose'
t5='/imu'
c1='/vesc/commands/motor/speed'
c2='/vesc/commands/servo/position'
topics = [t1,t2,t3,t5,c1,c2]
max_datas = tt[1][t1][1] # number of t1 messages
min_datas = tt[1][t3][1] # number of t3 messages

INPUT_SIZE=5

poses   = np.zeros((min_datas,3))
raw_datas = np.zeros((min_datas,INPUT_SIZE))
x_datas = np.zeros((min_datas,6))
y_datas = np.zeros((min_datas,3))

time0 = bag.get_start_time()
last_servo = None
last_vel = None
last_t = time0
idx=0
print("INPUT SIZE: ", INPUT_SIZE)
if INPUT_SIZE == 5:
    for topic, msg, t in bag.read_messages(topics=topics):
        #if idx > 10000:
        #    break
        dt = t.to_sec() - last_t
        last_t = t.to_sec()
        if topic == t1:
            last_vel = msg.state.speed
        elif topic == t2:
            last_servo = msg.data
        elif topic == t3 and last_vel is not None and last_servo is not None:
            last_vel   = (last_vel - SPEED_TO_ERPM_OFFSET) / SPEED_TO_ERPM_GAIN
            last_servo = (last_servo-STEERING_TO_SERVO_OFFSET) / STEERING_TO_SERVO_GAIN
            orientation = Utils.quaternion_to_angle(msg.pose.orientation)
            data = np.array([msg.pose.position.x,
                             msg.pose.position.y,
                             orientation,
                             last_vel,
                             last_servo])
            #print(data)
            #poses[idx,:] = data[0:3]
            raw_datas[idx,:] = data
            idx = idx+1
            if idx % 1000==0:
                print('.')
            #print(msg)
elif INPUT_SIZE == 4:
    ctrl1 = None
    ctrl2 = None
    for topic, msg, t in bag.read_messages(topics=topics):
        dt = t.to_sec() - last_t
        last_t = t.to_sec()
        if topic == t1:
            last_vel = (msg.state.speed - SPEED_TO_ERPM_OFFSET) / SPEED_TO_ERPM_GAIN
        elif topic == t5:
            last_servo = msg.angular_velocity.z / 100.0
        elif topic == c1:
            ctrl1 = (msg.data - SPEED_TO_ERPM_OFFSET) / SPEED_TO_ERPM_GAIN
        elif topic == c2:
            ctrl2 = (msg.data - STEERING_TO_SERVO_OFFSET) / STEERING_TO_SERVO_GAIN

        if (topic == t3) and \
                (last_vel is not None) and \
                (last_servo is not None) and \
                (ctrl1 is not None) and \
                (ctrl2 is not None):
            print(last_vel, last_servo, ctrl1, ctrl2)
            raw_datas[idx,:] = np.array([last_vel, last_servo,
                             ctrl1, ctrl2])
            orientation = Utils.quaternion_to_angle(msg.pose.orientation)
            poses[idx,:] = np.array([msg.pose.position.x,
                             msg.pose.position.y,
                             orientation])
            idx = idx+1
            if idx % 1000==0:
                print('.')
bag.close()


raw_datas = raw_datas[ np.abs(raw_datas[:,3]) < 0.75 ]
raw_datas = raw_datas[ np.abs(raw_datas[:,4]) < 0.36 ]

poses = raw_datas[:,0:3]

poses   = poses[:-10,:] # trim the last few in case of missing data
raw_datas = raw_datas[:-10,:] # trim the last few in case of missing data
x_datas = np.zeros((raw_datas.shape[0],6))
y_datas = np.zeros((raw_datas.shape[0],3))

pose_dot = scipy.signal.savgol_filter( np.diff(poses, axis=0), 15, 3, axis=0)

# controls
x_datas[:,3:6] = raw_datas[:,2:5] # theta and controls

# Outlier filtering
mask     = np.abs(pose_dot[:,0]) < 0.5
pose_dot = pose_dot[ mask ]
poses    = poses[ mask ]

mask     = np.abs(pose_dot[:,1]) < 0.5
pose_dot = pose_dot[ mask ]
poses    = poses[ mask ]

mask     = np.abs(pose_dot[:,2]) < 1 
pose_dot = pose_dot[ mask ]
poses    = poses[ mask ]

CAR_FRAME = True
if CAR_FRAME == True:
    # Car FRAME
    print("GIMMA DAT CAR FRAM")
    car_frame = np.zeros((pose_dot.shape[0], 3))
    for i in range(0,pose_dot.shape[0]):
        rot = Utils.rotation_matrix(-poses[i,2])
        delta = pose_dot[i,0:2] #.transpose()
        local_delta = rot*delta[:,np.newaxis] #).transpose()
        car_frame[i,0] = local_delta[0]
        car_frame[i,1] = local_delta[1]
        car_frame[i,2] = pose_dot[i,2]
    x_datas[0:car_frame.shape[0]-1,0:3] = car_frame[0:car_frame.shape[0]-1,:]
    y_datas[0:car_frame.shape[0]-1,:]   = car_frame[1:car_frame.shape[0],:]
else:
    print("Da WRULD IS URZ")
    # WORLD FRAME
    x_datas[0:pose_dot.shape[0]-1,0:3] = pose_dot[0:pose_dot.shape[0]-1,:]
    y_datas[0:pose_dot.shape[0]-1,:]   = pose_dot[1:pose_dot.shape[0],:]

i = pose_dot.shape[0]-1
print(x_datas.shape)
x_datas = x_datas[0:i-1,:] # trim unused bits
y_datas = y_datas[0:i-1,:] # trim unused bits
print(x_datas.shape)
print(y_datas.shape)

print("Xdot  ", np.min(x_datas[:,0]), np.max(x_datas[:,0]))
print("Ydot  ", np.min(x_datas[:,1]), np.max(x_datas[:,1]))
print("Tdot  ", np.min(x_datas[:,2]), np.max(x_datas[:,2]))
print("Thta  ", np.min(x_datas[:,3]), np.max(x_datas[:,3]))
print("vel   ", np.min(x_datas[:,4]), np.max(x_datas[:,4]))
print("delt  ", np.min(x_datas[:,5]), np.max(x_datas[:,5]))
print()
print("y Xdot", np.min(y_datas[:,0]), np.max(y_datas[:,0]))
print("y Ydot", np.min(y_datas[:,1]), np.max(y_datas[:,1]))
print("y Tdot", np.min(y_datas[:,2]), np.max(y_datas[:,2]))

######### NN stuff
# x_dot, y_dot, theta_dot, [vel, delta]*1
# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
N, D_in, H, D_out = 64, 6, 32, 3

#x = Variable(torch.from_numpy(x_datas.astype('float32')).type(dtype))
#y = Variable(torch.from_numpy(y_datas.astype('float32')).type(dtype), requires_grad=False)
num_samples = x_datas.shape[0]
rand_idx = np.random.permutation(num_samples)
x_datas = x_datas[rand_idx,:]
y_datas = y_datas[rand_idx,:]
x_tr = x_datas[:int(0.9*num_samples)]
y_tr = y_datas[:int(0.9*num_samples)]
x_tt = x_datas[int(0.9*num_samples):]
y_tt = y_datas[int(0.9*num_samples):]

x = torch.from_numpy(x_tr.astype('float32')).type(dtype)
y = torch.from_numpy(y_tr.astype('float32')).type(dtype)
x_val = torch.from_numpy(x_tt.astype('float32')).type(dtype)
y_val = torch.from_numpy(y_tt.astype('float32')).type(dtype)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)
model = model.cuda()

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-3
#opt = torch.optim.RMSprop(model.parameters(), lr=1e-3) #learning_rate)
opt = torch.optim.Adam(model.parameters(), lr=1e-3) #learning_rate)
#opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

mydataset = torch.utils.data.TensorDataset(x, y)
BATCH = 2048
trainloader=torch.utils.data.DataLoader(mydataset, batch_size=BATCH, shuffle=True)

def dowork(filename, optimizer, MINIBATCH=False, N=5000):
    if MINIBATCH == True:
        for t in range(N):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data 
                inputs = Variable(inputs)
                labels = Variable(labels, requires_grad=False)
                y_pred = model(inputs)
                loss = loss_fn(y_pred, labels)
        
                if t % 10 == 0 and i == 0:
                    val = model(Variable(x_val))
                    vloss = loss_fn(val, Variable(y_val, requires_grad=False))
                    print(t, loss.data[0]/BATCH, vloss.data[0]/BATCH)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    else:
        for t in range(N):
            y_pred = model(Variable(x))
            loss = loss_fn(y_pred, Variable(y, requires_grad=False))
            if t % 10 == 0:
                val = model(Variable(x_val))
                vloss = loss_fn(val, Variable(y_val, requires_grad=False))
                print(t, loss.data[0]/x.shape[0], vloss.data[0]/x_val.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, filename)


def rollout(m, nn_input, N):
    init = torch.zeros(3).cuda()
    print(init.cpu().numpy())
    for i in range(N):
        out = m(Variable(nn_input))
        init.add_(out.data)
        if init[2] > 3.14:
            init[2] -= 3.14
        if init[2] < -3.14:
            init[2] += 3.14
        nn_input[0] = out.data[0]
        nn_input[1] = out.data[1]
        nn_input[2] = out.data[2]
        nn_input[3] = init[2]
        print(init.cpu().numpy())
 
def test_model(m, N):
    s = 6
    print("Nothing")
    nn_input = torch.zeros(s).cuda()
    rollout(m, nn_input, N)

    print("Forward")
    nn_input = torch.zeros(s).cuda()
    nn_input[4] = 0.7 #1.0
    rollout(m, nn_input, N)

    print("Backwards")
    nn_input = torch.zeros(s).cuda()
    nn_input[4] = -0.7 #1.0
    rollout(m, nn_input, N)

    print("RIGHT")
    nn_input = torch.zeros(s).cuda()
    nn_input[4] = 0.7 
    nn_input[5] = -0.26
    rollout(m, nn_input, N)

    print("Left")
    nn_input = torch.zeros(s).cuda()
    nn_input[4] = 0.7
    nn_input[5] = 0.31
    rollout(m, nn_input, N)


