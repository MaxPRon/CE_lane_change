import numpy as np
import car
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data




class World:

    def __init__(self,num_of_cars, lanes,length,speed_limit,init_pos,init_lane):

        self.n_lanes = lanes
        self.n_cars = num_of_cars
        self.length = length
        self.speed_limit = speed_limit/3.6
        self.T = 2
        self.s0 = self.T * self.speed_limit
        self.delta = 4
        self.road_width = 4
        self.lane_init = (init_lane-1)*self.road_width + self.road_width*0.5


        self.vehicle_list = []
        self.vehicle_list.append(car.Car(init_pos,self.lane_init,self.speed_limit,0,speed_limit))
        for n in range(0,self.n_cars):
            y_random = random.randint(0,self.n_lanes-1)*self.road_width + self.road_width*0.5
            x_random = np.random.choice(np.arange(0.4,2,0.1))*self.length
            v_random = random.random()*self.speed_limit
            yaw_init = 0 #if lane change scenario
            self.vehicle_list.append(car.Car(x_random,y_random,v_random,0,speed_limit))


    def render(self):
        image_path_ego = get_sample_data('car-red.png')
        image_path_cars = get_sample_data('car-blue.png')

        plt.figure(1)
        ax1 = plt.subplot(1, 1, 1)
        ego = self.vehicle_list[0]
        ax1.plot([ego.x], [ego.y], 'ro')
        # Car Pictures
        imscatter([ego.x], [ego.y],image_path_ego,zoom=0.03,ax=ax1)

        ax2 = plt.step
        for lane in range(1,self.n_lanes+1):
            ax1.axhline((lane-1)*self.road_width + self.road_width*0.5, color="g", linestyle="-", linewidth=1)
        x_cars = []
        y_cars = []
        for n in range(1,self.n_cars+1):
            vehicle = self.vehicle_list[n]
            ax1.plot([vehicle.x], [vehicle.y], color = 'b',marker = 'x' )
            x_cars.append(vehicle.x)
            y_cars.append(vehicle.y)
        # Car Pictures

        imscatter(x_cars, y_cars, image_path_cars, zoom=0.03, ax=ax1)
        #ax1.grid()
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()

    def step(self,action):

        # Ego Action



        acc = action[0]
        steer = action[1]

        ego = self.vehicle_list[0]

        #ego.ego_motion(acc,steer)
        ego.y = steer
        ego.motion(acc, 0)
        self.vehicle_list[0] = ego

        # Non-Ego Action
        for n in range(1,self.n_cars+1):
            vehicle = self.vehicle_list[n]
            #acc, dist = self.IDM(n)
            acc = self.dist_control(n)
            vehicle.motion(acc,0)
            self.vehicle_list[n] = vehicle

    def IDM(self, id):

        # Get vehicles on same lane
        lane = self.vehicle_list[id].y
        x_pos = self.vehicle_list[id].x


        # Add Field of view
        vehicle_list_sec = [vehicle for vehicle in self.vehicle_list if vehicle.y <= lane + self.road_width*0.5 and vehicle.y >= lane - self.road_width*0.5 and vehicle != self.vehicle_list[id] and self.vehicle_list[id].x < vehicle.x]

        # Calculate distance to car in front
        if len(vehicle_list_sec) == 0:
            sa = self.s0*10
            v_delta =self.vehicle_list[id].v-self.speed_limit

        else:
            x_front = min(vehicle.x for vehicle in vehicle_list_sec)
            v_front = [vehicle.v for vehicle in vehicle_list_sec if vehicle.x ==x_front]
            sa = x_front - x_pos - 2*self.vehicle_list[id].lf
            v_delta = self.vehicle_list[id].v - v_front[0]

        # Implementation of IDM
        va = self.vehicle_list[id].v
        s_star = self.s0 + va*self.T + np.divide(va*v_delta,2*np.sqrt(self.vehicle_list[id].a*self.vehicle_list[id].b))
        acc = self.vehicle_list[id].a*(1 - (np.divide(va, self.speed_limit))**(self.delta) - (np.divide(s_star, sa))**2)

        return acc, sa

    def get_ego(self):

        ego = self.vehicle_list[0]

        return ego.x, ego.y, ego.x_dot, ego.y_dot, ego.v


    def dist_control(self,id):

        alpha =0.5
        lane = self.vehicle_list[id].y
        x_pos = self.vehicle_list[id].x

        vehicle_list_sec = [vehicle for vehicle in self.vehicle_list if
                            vehicle.y <= lane + self.road_width * 0.5 and vehicle.y >= lane - self.road_width * 0.5 and vehicle !=
                            self.vehicle_list[id] and self.vehicle_list[id].x < vehicle.x]

        # Calculate distance to car in front
        if len(vehicle_list_sec) == 0:

            acc = alpha*(self.speed_limit - self.vehicle_list[id].v)

        else:
            x_front = min(vehicle.x for vehicle in vehicle_list_sec)
            v_front = [vehicle.v for vehicle in vehicle_list_sec if vehicle.x == x_front]
            v_ego = self.vehicle_list[id].v
            dist = x_front - self.vehicle_list[id].x
            acc = alpha*(v_front[0] - v_ego) + 0.25*(alpha**2)*(dist-self.s0)


        if acc >= 0:
            acc = min(self.vehicle_list[id].a,acc)
        else:
            acc = max(-self.vehicle_list[id].b,acc)

        return acc


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists




