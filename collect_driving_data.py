import os
import shutil
import subprocess


import gym
import pygame
from PIL import Image
from pygame.locals import *

#custom wrapper classes
from hud2 import HUD
from wrappers2 import *

import sys
import glob
import argparse

import numpy as np

import automatic_control
import agents

import weakref
import math

from synchronous_mode import CarlaSyncMode

from matplotlib import cm

#look for automatic agents
try:
    sys.path.insert(0,'/home/lunet/cokm2/CARLA/PythonAPI/carla/agents')
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import LocalPlanner

from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

#look for carla file
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

'''This class contains the IMU sensor class. Taken from https://github.com/Ashwin-Rajesh/Kalman_filter_carla'''
class imu_integrate:
    # State : A column vector with [x_pos, y_pos, yaw, x_vel, y_vel]
    def __init__(self, init_state, init_time):
        self.state  = init_state
        self.prev_time   = init_time

        self.states = []

    # Input : A column vector with [x_accel, y_accel, yaw_vel]
    def update(self, inp, time):
        dt = time - self.prev_time

        # Transition matrix :
        #
        # | 1   0   0   dt  0  |
        # | 0   1   0   0   dt |
        # | 0   0   1   0   0  |
        # | 0   0   0   1   0  |
        # | 0   0   0   0   1  |
        #
        A = np.asarray([\
            [1, 0, 0, dt,0], \
            [0, 1, 0, 0, dt],\
            [0, 0, 1, 0, 0], \
            [0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 1]  \
            ])

        # Input influence matrix
        #
        # | 0   0   0  |
        # | 0   0   0  |
        # | 0   0   dt |
        # | dt  0   0  |
        # | 0   dt  0  |
        #
        B = np.asarray([\
            [0, 0, 0], \
            [0, 0, 0], \
            [0, 0, dt],\
            [dt,0, 0], \
            [0, dt,0], \
            ])

        yaw      = self.state[2]
        accel_xl = inp[0]
        accel_yl = inp[1]
        accel_xg = accel_xl * np.cos(yaw) - accel_yl * np.sin(yaw)
        accel_yg = accel_xl * np.sin(yaw) + accel_yl * np.cos(yaw)

        inp[0]  = accel_xg
        inp[1]  = accel_yg

        # State updation with input
        self.state = A.dot(self.state) + B.dot(inp)

        if(self.state[2] > np.pi):
            self.state[2] = self.state[2] - 2 * np.pi
        elif(self.state[2] < -np.pi):
            self.state[2] = self.state[2] + 2 * np.pi

        # Append to states
        self.states.append([self.state, time])

        # Update previous time
        self.prev_time = time

    # Return position
    def get_pos(self):
        return (self.states[len(self.states)-1])

'''This class contains the kalman filtering https://github.com/Ashwin-Rajesh/Kalman_filter_carla'''
class kalman_filter:
    # State : A column vector with [x_pos, y_pos, yaw, x_vel, y_vel]
    def __init__(self, init_state, init_time, accel_var, yaw_var, meas_var):
        self.state          = np.asarray(init_state).reshape(5,1)
        self.prev_time      = init_time
        self.covar          = np.zeros((5,5))

        self.Q              = np.diag([accel_var, accel_var, yaw_var])
        self.R              = np.diag([meas_var, meas_var])

        self.states = []
        self.covars = []

    # Input : A column vector with [x_accel, y_accel, yaw_vel]
    def update(self, inp, time):

        inp = np.asarray(inp).reshape(3,1)

        dt = time - self.prev_time

        # Transition matrix :
        #
        # | 1   0   0   dt  0  |
        # | 0   1   0   0   dt |
        # | 0   0   1   0   0  |
        # | 0   0   0   1   0  |
        # | 0   0   0   0   1  |
        #
        A = np.asarray([\
            [1, 0, 0, dt,0], \
            [0, 1, 0, 0, dt],\
            [0, 0, 1, 0, 0], \
            [0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 1]  \
            ])

        # Input influence matrix
        #
        # | 0   0   0  |
        # | 0   0   0  |
        # | 0   0   dt |
        # | dt  0   0  |
        # | 0   dt  0  |
        #
        B = np.asarray([\
            [0, 0, 0], \
            [0, 0, 0], \
            [0, 0, dt],\
            [dt,0, 0], \
            [0, dt,0], \
            ])

        # L = np.asarray([\
        #     [0, 0, 0,], \
        #     [0, 0, 0,], \
        #     [0, 0, 1,], \
        #     [1, 0, 0],  \
        #     [0, 1, 0],  \
        #     ])

        yaw      = self.state[2][0]
        accel_xl = inp[0][0]
        accel_yl = inp[1][0]
        accel_xg = accel_xl * np.cos(yaw) - accel_yl * np.sin(yaw)
        accel_yg = accel_xl * np.sin(yaw) + accel_yl * np.cos(yaw)

        dxvel_dyaw = -dt * (inp[0][0] * np.sin(self.state[2][0]) + inp[1][0] * np.cos(self.state[2][0]))
        dyvel_dyaw =  dt * (inp[0][0] * np.cos(self.state[2][0]) - inp[1][0] * np.sin(self.state[2][0]))

        dxvel_din1 =  dt * np.cos(self.state[2][0])
        dxvel_din2 = -dt * np.sin(self.state[2][0])
        dyvel_din1 =  dt * np.sin(self.state[2][0])
        dyvel_din2 =  dt * np.cos(self.state[2][0])

        g_inp = np.asarray([accel_xg, accel_yg, inp[2][0]]).reshape(3,1)
        # State updation with input
        self.state = A.dot(self.state) + B.dot(g_inp)
        #self.state = np.asarray([x_new, y_new, yaw_new, xvel_new, yvel_new]).reshape(5,1)

        if(self.state[2][0] > np.pi):
            self.state[2][0] = self.state[2][0] - 2 * np.pi
        elif(self.state[2][0] < -np.pi):
            self.state[2][0] = self.state[2][0] + 2 * np.pi

        # x_new    = self.state[0][0] + dt * self.state[3][0]
        # y_new    = self.state[1][0] + dt * self.state[4][0]
        # yaw_new  = self.state[2][0] + dt * inp[2][0]
        # xvel_new = self.state[3][0] + dt * (inp[0][0] * np.cos(self.state[2][0]) - inp[1][0] * np.sin(self.state[2][0]))
        # yvel_new = self.state[4][0] + dt * (inp[0][0] * np.sin(self.state[2][0]) + inp[1][0] * np.cos(self.state[2][0]))

        A = np.asarray([\
            [1, 0, 0,           dt,0], \
            [0, 1, 0,           0, dt],\
            [0, 0, 1,           0, 0], \
            [0, 0, dxvel_dyaw,  1, 0], \
            [0, 0, dyvel_dyaw,  0, 1]  \
            ])

        B = np.asarray([\
            [0,             0,          0], \
            [0,             0,          0], \
            [0,             0,          dt],\
            [dxvel_din1,    dxvel_din2, 0], \
            [dyvel_din1,    dyvel_din2, 0], \
            ])

        # Covariance update
        self.covar = A.dot(self.covar.dot(A.T)) + B.dot(self.Q.dot(B.T))

        # Append to trajectory
        self.states.append([self.state, time, 0])
        self.covars.append([self.covar, time, 0])

        # Update previous time
        self.prev_time = time

    def measure(self, measurement, time):
        # How to find expected measurement from state?
        H = np.asarray([\
                        [1, 0, 0, 0, 0], \
                        [0, 1, 0, 0, 0], \
                        ])

        measurement = np.asarray(measurement).reshape(2,1)

        # Error of measurement from expected measurement
        V = measurement - H.dot(self.state)

        S = H.dot(self.covar.dot(H.T)) + self.R

        K = self.covar.dot(H.T.dot(np.linalg.inv(S)))

        self.state = self.state + K.dot(V)

        self.covar = self.covar - K.dot(S.dot(K.T))

        # Append to trajectory
        self.states.append([self.state, time, 1])
        self.covars.append([self.covar, time, 1])

    # Return position
    def get_pos(self):
        return (self.states[len(self.states)-1])

'''This class constructs an automatic driving agent to collect data'''
class driveCar(object):
    def __init__(self, args, client):
        pygame.init()
        pygame.font.init()

#Pygame is initalised but not displayed on purpose
        start, finish = [int(x) for x in args.route.split(',')]
        #print(start, finish)

        self.width, self.height = [int(x) for x in args.viewer_res.split('x')]
        self.out_width, self.out_height = [int(x) for x in args.obs_res.split('x')]
        args.dot_extent -=1

        # Setup gym environment
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.out_width, self.out_height, 3), dtype=np.float32)
        self.action_smoothing = 0.9

        #self.vehicle = vehicle
        self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

#This sets the recording flags and buffers to store sensor  data
        self.done = False
        self.recording = False
        self.extra_info = []
        self.num_saved_observations = 0
        #self.num_images_to_save = (args.num_images*4)
        self.num_images_to_save = args.num_images
        self.observation = {key: None for key in ["segmentation", "lidar","logDepth","GNSS", "IMU"]}        # Last received observations
        #self.observation_sensor
        self.observation_buffer = {key: None for key in ["segmentation", "lidar","logDepth","GNSS","IMU"]}
        self.toSave = {key: None for key in ["Interlaced", "logDepth"]}


        # Remove existing output directory
        if os.path.isdir(args.output_dir):
           shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

        self.output_dir = args.output_dir
        os.makedirs(os.path.join(self.output_dir, "logDepth"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "Interlaced"), exist_ok=True)

#setting spawn locations
        self.args = args
        self.client = client
        self.world = World(self.client)
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.spawnLoc = carla.Location(self.spawn_points[start].location)
        spawnRot = carla.Rotation(pitch=0.0, yaw=0, roll=0.000000)
        spawnTrans = carla.Transform(self.spawnLoc, spawnRot)
        self.destination = carla.Location(self.spawn_points[finish].location)

#constructing vehicle object for agent to drive
        self.vehicle = Vehicle(self.world, spawnTrans,
        on_collision_fn=lambda e: self._on_collision(e),
       on_invasion_fn=lambda e: self._on_invasion(e))

#Create heads up display, outputs information but not video in real time
        self.hud = HUD(self.width, self.height)
        self.hud.set_vehicle(self.vehicle)
        try:
            self.world.on_tick(self.hud.on_world_tick)
        except Exception:
            print("cant do")

#Dash cam for segmented output
        self.dashcam_seg = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("segmentation", e),
                                      camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)

#dashcam for logarithmic video
        self.dashcam_depth = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("logDepth", e),
                                      camera_type="sensor.camera.depth", color_converter=carla.ColorConverter.LogarithmicDepth)

#Lidar wrapper for dashboard
        self.lidar_sen = Lidar(self.world, args, self.dashcam_seg, self.out_width, self.out_height, transform=camera_transforms["dashboard"],
                        attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("lidar", e))

#gnss sensor wrapper
        self.gnss_sensor = GNSScheat(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_image("GNSS", e))

#imu sensor wrapper
        self.imu_sensor = IMU(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_image("IMU", e))

#this code is used to interlace lidar points on to field of view
        fov = self.dashcam_seg.fov
        focal = self.out_width / (2.0 * np.tan(fov * np.pi /360.0))
        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = focal
        self.K[0, 2] = self.out_width / 2.0
        self.K[1, 2] = self.out_height / 2.0

        self.VIRIDIS = np.array(cm.get_cmap('viridis').colors)
        self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])

#called when the escape button is pressed
    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

#This method lets you display lidar points on to the field of view
    def interlace(self, seg_image, lidar_data, intensity):
        im_array = seg_image
        sensor_points = lidar_data
        point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(self.K, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        #discard unnesscary points
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < self.out_width) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < self.out_height) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(int)
        v_coord = points_2d[:, 1].astype(int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 2]) * 255.0]).astype(int).T

        if args.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                im_array[
                    v_coord[i]-args.dot_extent : v_coord[i]+args.dot_extent,
                    u_coord[i]-args.dot_extent : u_coord[i]+args.dot_extent] = color_map[i]
        #print(im_array.shape)
        return im_array

    def gnss_to_xyz(self, data):
        lat = deg_to_rad(data[0])
        lon = deg_to_rad(data[1])
        alt = data[2]

        rad_y  = 6.357e6
        rad_x  = 6.378e6

        x = (lon - self.vehicle.geo_centre_lon) * np.cos(self.vehicle.geo_centre_lat) * rad_x
        y = (self.vehicle.geo_centre_lat - lat) * rad_y
        z = alt - self.vehicle.geo_centre_alt

        return x, y, z

#This function saves an data for a time step for all sensors
    def save_observation(self):
        object = self.interlace(self.observation["segmentation"], self.observation["lidar"][0],self.observation["lidar"][1])
        self.toSave.update({'Interlaced' : object})
        self.toSave.update({'logDepth' : self.observation["logDepth"]})

        init_vel = self.vehicle.get_velocity()
        init_loc = self.vehicle.get_location()
        init_rot = self.vehicle.get_transform().rotation

        timestamp = self.observation["IMU"].timestamp
        init_state = np.asarray([init_loc.x, init_loc.y, init_rot.yaw * np.pi/180, init_vel.x, init_vel.y]).reshape(5,1)
        int_obj = imu_integrate(init_state,timestamp)

#This is part of the IMU sensor class
        int_rvel_list = []
        int_rpos_list = []
        int_ryaw_list = []
        rvel = init_vel
        rloc = init_loc
        rrot = init_rot

        int_rvel_list.append(((rvel.x, rvel.y), timestamp))
        int_rpos_list.append(((rloc.x, rloc.y), timestamp))
        int_ryaw_list.append((rrot.yaw * np.pi / 180, timestamp))

        yaw_vel = self.observation["IMU"].gyroscope.z
        accel_x = self.observation["IMU"].accelerometer.x
        accel_y = self.observation["IMU"].accelerometer.y
        #print(yaw_vel, accel_x, accel_y)
        prev_posIMU = carla.Location(int_obj.state[0][0], int_obj.state[1][0], rloc.z+1)
        int_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3,1), timestamp)
        posIMU = carla.Location(int_obj.state[0][0], int_obj.state[1][0], rloc.z+1)

        imu_var_a   = 0.05
        imu_var_g   = 0.01
        gnss_var    = 30
        kal_obj = kalman_filter(init_state, timestamp, imu_var_a, imu_var_g, gnss_var)

        kal_rvel_list  = []
        kal_rpos_list  = []
        kal_ryaw_list  = []
        kal_gnss_list  = []
        kal_gact_list  = []

        kal_rvel_list.append(((rvel.x, rvel.y), timestamp))
        kal_rpos_list.append(((rloc.x, rloc.y), timestamp))
        kal_ryaw_list.append((rrot.yaw * np.pi / 180, timestamp))

        prev_posKal = carla.Location(kal_obj.state[0][0], kal_obj.state[1][0], rloc.z+1)
        kal_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3,1), timestamp)
        posKal = carla.Location(kal_obj.state[0][0], kal_obj.state[1][0], rloc.z+1)

        x = self.vehicle.get_transform().location.x
        y = self.vehicle.get_transform().location.y
        z = self.vehicle.get_transform().location.z

        kal_gnss_list.append(((x, y), timestamp))
        kal_gact_list.append(((rloc.x, rloc.y), timestamp))
        prev_posGNSS = carla.Location(kal_obj.state[0][0], kal_obj.state[1][0], rloc.z+1)
        kal_obj.measure(np.asarray([x, y]).reshape(2,1), timestamp)
        posGNSS = carla.Location(kal_obj.state[0][0], kal_obj.state[1][0], rloc.z+1)

#when saved the folder, increment the counter
        if self.recording:
            for obs_type, obs in self.toSave.items():
                img = Image.fromarray(obs)
                img.save(os.path.join(self.output_dir, obs_type, "{}.png".format(self.num_saved_observations)))
            self.num_saved_observations += 0.5
            if self.num_saved_observations >= self.num_images_to_save:
                #print(self.num_saved_observations,self.num_images_to_save,"Over HEre")
                self.done = True

        # Render HUD
        self.extra_info.extend([
            "Images: %i/%i" % (self.num_saved_observations, self.num_images_to_save),
            "Progress: %.2f%%" % (self.num_saved_observations / self.num_images_to_save * 100.0)
        ])
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

#reset the imu lists
        int_rvel_list = None
        int_rpos_list = None
        int_ryaw_list = None

        kal_rvel_list  = None
        kal_rpos_list  = None
        kal_ryaw_list  = None
        kal_gnss_list  = None
        kal_gact_list  = None

#for a time step, excecute a certain number of functions
    def step(self, action):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]
            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)


        # Tick game
        self.clock.tick()
        self.hud.tick(self.world, self.clock)
        self.world.tick()
        try:
            self.world.wait_for_tick(seconds=0.5)
        except RuntimeError as e:
            pass # Timeouts happen for some reason, however, they are fine to ignore

#saving an obsevation in abuffer accofding to a key from the callback function
        self.observation["segmentation"] = self._get_observation("segmentation")
        self.observation["logDepth"] = self._get_observation("logDepth")
        self.observation["lidar"] = self._get_observation_lidar("lidar")
        self.observation["GNSS"] = self._get_observation_gnss("GNSS")
        self.observation["IMU"] = self._get_observation_imu("IMU")

#press escape to stop simulation
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            self.done = True
        if keys[K_SPACE]:
            self.recording = True

#funcition that is flag for
    def is_done(self):
        return self.done

#Callback function for sensors to use
    def _get_observation(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name].copy()
        self.observation_buffer[name] = None
        return obs

#observation buffer for IMU
    def _get_observation_imu(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name]
        self.observation_buffer[name] = None
        return obs

#observation buffer for lidar, gets the sensor and intenstiy points
    def _get_observation_lidar(self, name):
        while self.observation_buffer[name][0] is None:
            pass
        obs = self.observation_buffer[name][0].copy()
        self.observation_buffer[name][0] = None

        while self.observation_buffer[name][1] is None:
            pass
        intensity = self.observation_buffer[name][1].copy()
        self.observation_buffer[name][1] = None
        return obs, intensity

#observation buffer for the GNNS sensor
    def _get_observation_gnss(self, name):
        while self.observation_buffer[name][0] is None:
            pass
        lat = self.observation_buffer[name][0]
        self.observation_buffer[name][0] = None

        while self.observation_buffer[name][1] is None:
            pass
        long = self.observation_buffer[name][1]
        self.observation_buffer[name][1] = None

        while self.observation_buffer[name][2] is None:
            pass
        alt = self.observation_buffer[name][2]
        self.observation_buffer[name][2] = None

        return lat, long, alt

#collision notifcation
    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

#invasion notifaction
    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

#image buffer for heads up display, not used
    def _set_observation_image(self, name, image):
        self.observation_buffer[name] = image

#main routine that calls a basic logic agent from the carla folder to driver around a route
    def main_loop(self):
        action = np.zeros(self.action_space.shape[0])
        #print(self.world.get_settings().synchronous_mode)
        try:
            agent = BasicAgent(self.vehicle, target_speed=17)

            agent.set_destination((self.destination.x,
                                   self.destination.y,
                                   self.destination.z))

            self.recording = False
            while self.num_saved_observations < 500:
                self.clock.tick_busy_loop(60)
                self.world.tick()
                control = agent.run_step()
                control.manual_gear_shift = False
                self.vehicle.apply_control(control)
                self.world.tick()
                action[0] = np.clip(control.steer, -1, 1)
                action[1] = control.throttle
                self.step(action)
                self.save_observation()

        finally:
            if self.world is not None:
                self.world.destroy()
            pygame.quit()

#this stand alone function allows different towns to be loaded
def preset(i):
    switcher={
        'Town01':carla.WeatherParameters.ClearNoon,
        'Town02':carla.WeatherParameters.HardRainNoon,
        'Town03':carla.WeatherParameters.CloudySunset,
        'Town04':carla.WeatherParameters.SoftRainNoon,
        'Town05':carla.WeatherParameters.HardRainSunset
    }
    return switcher.get(i,'Invalid, left for validating model')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument("--viewer_res", default="400x500", type=str, help="Window resolution (default: 1280x720)")
    argparser.add_argument("--obs_res", default="160x80", type=str, help="Output resolution (default: same as --res)")
    argparser.add_argument("--output_dir", default="images", type=str, help="Directory to save images to")
    argparser.add_argument("--num_images", default=1000, type=int, help="Number of images to collect")
    argparser.add_argument(
            '-d', '--dot-extent',
            metavar='SIZE',
            default=2,
            type=int,
            help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    argparser.add_argument(
        '--Town',
        default='Town07',
        help='Select Map')
    argparser.add_argument(
        '--route',
        default='1,100',
        help='Select spawn locationss')

#connect to carla client with a town argument
    args = argparser.parse_args()
    client = carla.Client(args.host, args.port)
    try:
        weather = preset(args.Town)
    except Exception as e:
        print(e)

    try:
        client.set_timeout(2.0)
        client.load_world(args.Town)
        world = client.get_world()
    except Exception as e:
        print(e, "Could not find server")

#run main loop
    try:
        obj2 = driveCar(args,client)
        obj2.main_loop()

    except RuntimeError:
            print("Not Valid")
