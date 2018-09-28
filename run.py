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

def vectorize_state(state):
    v_x = []
    v_y = []
    v_v = []

    for id in range(len(state)):
        v_x.append(state[id].x)
        v_y.append(state[id].y)
        v_v.append(state[id].v)


    state_v = np.concatenate((v_x,v_y))
    state_v = np.concatenate((state_v, v_v))
    return state_v



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
ego_speed_init = speed_limit

#### Network parameters ####

input_dim = (num_of_cars+1)*3
output_dim = x_range*num_of_lanes
hidden_units = 100
layers = 5
clip_value = 300
learning_rate = 0.001
buffer_size = 50000
batch_size = 32
update_freq = 10000

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 100000

#### Learning Parameters ####

max_train_episodes = 100000
pre_train_steps = 100000
random_sweep = 10
tau = 1


#### Environment ####

done = False
dt = 0.05
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,ego_speed_init,dt,random_seed,x_range)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(2) # acc/steer


f.reset_default_graph()

mainQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)
targetQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf. trainable_variables()

targetOps = q_learning.updateNetwork(trainables,tau)

random_sweep= 5

## Init environment ##

states = []
actions = []
reward_time = []
reward_average = []
reward_episode = 0
total_steps = 0

done = False

final_save_path = "./results/bayes_constant/model_random_61065/random_1_Final.ckpt"
num_tries = 100

for t in range(0,num_tries):
    if t % 100 == 0:
        print("Number of tries:" + str(t))
    with tf.Session() as sess:
        done = False
        sess.run(init)
        saver.restore(sess,final_save_path)
        env = world.World(num_of_cars, num_of_lanes, track_length, speed_limit, ego_pos_init, ego_lane_init,
                          ego_speed_init, dt, random_seed, x_range)

        state,_,_ = env.get_state()
        state_v = vectorize_state(state)
        rewards = []
        test = 0
        flag = 0
        while done == False:
            action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state_v]})

            #action = random.randint(0,22)
            state1,reward,done, success = env.step(action)
            rewards.append(reward)
            test += reward

            env.render()
            #if test < -10:
            #    env.render()
            state1_v = vectorize_state(state1)
            state_v = state1_v

        reward_time.append(sum(rewards))
        if success == True:
            finished += 1


average_reward = sum(reward_time)/num_tries

print("Average reward for " + str(num_tries) + " is:" + str(average_reward))
print("Finished succesfully: %s/%s" % (finished, num_tries))