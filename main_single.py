import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning
import world
import lateral_agent
import car

#### Environment parameters ####

num_of_cars = 5
num_of_lanes = 2
track_length = 300
speed_limit = 120
random_seed = 0
random.seed(random_seed)
x_range = 10


#### Ego parameters ####

ego_lane_init = 1
ego_pos_init = 0
ego_speed_init = 0.5*speed_limit

#### Network parameters ####

input_dim = (num_of_cars+1)*3
output_dim = x_range*num_of_lanes
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
pre_train_steps = 100000
random_sweep = 5
tau = 1


#### Environment ####

done = False
dt = 0.05
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,dt,random_seed,x_range)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(2) # acc/steer

for r_seed in range(0,random_sweep):

    random.seed(r_seed)

    #### Start training process ####

    states = []
    actions = []
    reward_time = []

    folder_path = './initial_testing'

    path = folder_path+ "model_initial"

    if r_seed == 1:  # Only write for first time

        file = open(path + 'params' + str(id) + '.txt', 'w')
        # file = open(complete_file, 'w')
        file.write('NETWORK PARAMETERS: \n\n')
        file.write('Layers: ' + str(layers) + '\n')
        file.write('Hidden units: ' + str(hidden_units) + '\n')
        file.write('Learning rate: ' + str(learning_rate) + '\n')
        file.write('Buffer size: ' + str(buffer_size) + '\n')
        file.write('Pre_train_steps: ' + str(pre_train_steps) + '\n')
        file.write('Batch_size: ' + str(batch_size) + '\n')
        file.write('Update frequency: ' + str(update_freq) + '\n')
        file.write('Tau: ' + str(tau) + '\n\n')

        file.write('RL PARAMETERS: \n\n')
        file.write('Gamma: ' + str(gamma) + '\n')
        file.write('Epsilon start: ' + str(eStart) + '\n')
        file.write('Epsilon end: ' + str(eEnd) + '\n')
        file.write('Epsilon steps: ' + str(estep) + '\n')

        file.write('SCENARIO PARAMETERS: \n\n')
        file.write('Cars: ' + str(num_of_cars) + '\n')
        file.write('Lanes: ' + str(num_of_lanes) + '\n')
        file.write('Ego speed init: ' + str(ego_speed_init) + '\n')
        file.write('Ego pos init: ' + str(ego_pos_init) + '\n')
        file.write('Ego lane init: ' + str(ego_lane_init) + '\n')

        file.close()

        for x in range(10000):

            if x % 150 == 0:
                action = random.randint(0,29)
            env.step(action)
            env.render()







