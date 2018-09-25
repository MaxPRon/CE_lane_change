import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning
import world
import lateral_agent
import car

#### Environment parameters ####

num_of_cars = 12
num_of_lanes = 2
track_length = 300
speed_limit = 120
random_seed = 0
random.seed(random_seed)

#### Ego parameters ####

ego_lane_init = 1
ego_pos_init = 0
ego_speed_init = 0.5*speed_limit

#### Network parameters ####

input_dim = (num_of_cars+1)*3
output_dim = 10
hidden_units = 100
layers = 5
clip_value = 300
learning_rate = 0.001
buffer_size = 50000
batch_size = 32
update_freq = 6000

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 100000

#### Learning Parameters ####

max_train_episodes = 100000
pre_strain_steps = 100000


#### Environment ####

done = False
dt = 0.1
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,dt,random_seed)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(2) # acc/steer
timestep = 0

#### plot variables ####
x_ego_list = []
y_ego_list = []
y_acc_list = []
v_ego_list = []
x_acc_list = []

while done == False:

    ego_x, ego_y, ego_x_dot, ego_y_dot, ego_v = env.get_ego() # x, y, x_dot, y_dot, v
    x_ego_list.append(ego_x)
    y_ego_list.append(ego_y)
    v_ego_list.append(ego_v)
    goal_x = ego_x + 5 * ego_v

    if timestep % 100 == 0: # and goal_lane == goal_lane_prev
        goal_lane = int((random.randint(1,num_of_lanes) - 1) * env.road_width + env.road_width * 0.5)
        goal_x = 3*ego_v
        print("Goal:",goal_lane)


    if goal_lane != goal_lane_prev:
        _, _ = lateral_controller.solve(ego_y, goal_lane, 0, goal_x)
        x_base = ego_x

    if goal_lane ==  ego_y:
        action[0] = env.dist_control(0)
        action[1] = goal_lane
        y_acc_list.append(0)
        x_acc_list.append(action[0])
    else:
        action[0] = env.dist_control(0)
        action[1],ydot,yddot = lateral_controller.function_output(ego_x-x_base)
        y_acc_list.append(lateral_controller.y_acceleration(ego_x-x_base, ego_v))
        x_acc_list.append(action[0])

    if ego_x >= 5000:
        done = True

    env.step(action)
    env.render()
    goal_lane_prev = goal_lane
    timestep+= 1


plt.figure(1)
plt.plot(x_ego_list,y_ego_list)
plt.title('Driven Path')
plt.xlabel('x-pos in [m]')
plt.ylabel('y-pos in [m]')
plt.show(block=False)


plt.figure(2)
ax1 = plt.subplot(3,1,1)
ax1.plot(x_ego_list,y_acc_list)
ax1.set_xlabel('x_pos in [m]')
ax1.set_ylabel('acc in [m/s^2]')
ax1.set_title('y-acceleration')
ax2 = plt.subplot(3,1,3)
ax2.plot(x_ego_list,v_ego_list)
ax2.set_xlabel('x_pos in [m]')
ax2.set_ylabel('velocity in [m/s]')
ax2.set_title('Velocity')
ax3 = plt.subplot(3,1,2)
ax3.plot(x_ego_list,x_acc_list)
ax3.set_xlabel('x_pos in [m]')
ax3.set_ylabel('acc in [m/s^2]')
ax3.set_title('x-acceleration')
plt.tight_layout()
plt.show()


print(max(v_ego_list))