import os
import subprocess
import time

import carla
import gym
import pygame
from gym.utils import seeding
from pygame.locals import *

from hud2 import HUD
from planner import RoadOption, compute_route_waypoints
from wrappers2 import *

from matplotlib import cm

class CarlaLapEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, args, reward_fn=None, encode_state_fn=None,
                 synchronous=True, fps=30, action_smoothing=0.9):
        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        self.args = args
        width, height = [int(x) for x in self.args.viewer_res.split('x')]
        self.out_width, self.out_height = [int(x) for x in self.args.obs_res.split('x')]
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.out_width, self.out_height, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(60.0)
            self.client.load_world('Town07')

            # Create world wrapper
            self.world = World(self.client)

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                self.world.apply_settings(settings)

            # Get spawn location
            #lap_start_wp = self.world.map.get_waypoint(carla.Location(x=-180.0, y=110))
            lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[1].location)
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)

            self.model_3 = "vehicle.tesla.model3"

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, spawn_transform, vehicle_type=self.model_3,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            self.dashcam = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                      sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
                                      camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)

            self.dashcam_seg = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image_depth(e),
                                      sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
                                      camera_type="sensor.camera.depth", color_converter=carla.ColorConverter.LogarithmicDepth)


            self.lidar_sen = Lidar(self.world, self.args, self.dashcam, self.out_width, self.out_height, transform=camera_transforms["dashboard"],
                        attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_sensor("Lidar", e))

            self.gnss_sensor = GNSScheat(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_sensor("GNSS", e))

            self.imu_sensor = IMU(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_sensor("IMU", e))


            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

            fov = self.dashcam.fov
            focal = self.out_width / (2.0 * np.tan(fov * np.pi /360.0))
            self.K = np.identity(3)
            self.K[0, 0] = self.K[1, 1] = focal
            self.K[0, 2] = self.out_width / 2.0
            self.K[1, 2] = self.out_height / 2.0

            self.VIRIDIS = np.array(cm.get_cmap('viridis').colors)
            self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])

            self.observation_momentum =  {key: None for key in ["GNSS","IMU","Lidar"]}
            self.observation_momentum_buffer = {key: None for key in ["GNSS", "IMU","Lidar"]}


        except Exception as e:
            self.close()
            raise e

        # Generate waypoints along the lap
        self.route_waypoints = compute_route_waypoints(self.world.map, lap_start_wp, lap_start_wp, resolution=1.0,
                                                       plan=[RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5)
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0

        # Reset env to set initial state
        self.reset()

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

        if self.args.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                im_array[
                    v_coord[i]-self.args.dot_extent : v_coord[i]+self.args.dot_extent,
                    u_coord[i]-self.args.dot_extent : u_coord[i]+self.args.dot_extent] = color_map[i]
        #print(im_array.shape)
        return im_array

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=True):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()
        if is_training:
            # Teleport vehicle to last checkpoint
            waypoint, _ = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
            self.current_waypoint_index = self.checkpoint_waypoint_index
        else:
            # Teleport vehicle to start of track
            waypoint, _ = self.route_waypoints[0]
            self.current_waypoint_index = 0
        transform = waypoint.transform
        transform.location += carla.Location(z=1.0)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)

        time.sleep(2.0)

        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD

        self.observation_lidar = self.observation_buffer_lidar = None
        #self.observation_momentum =  {key: None for key in ["GNSS","IMU"]}
        self.observation_momentum_buffer = {key: None for key in ["GNSS", "IMU"]}
        #self.observation_momentum = self.observation_momementum_buffer = None

        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training
        self.start_waypoint_index = self.current_waypoint_index

        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0


        return self.step(None)[0]

    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW: maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:     maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:    maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT: maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.VOID:     maneuver = "VOID"
        else:                                                   maneuver = "INVALID(%i)" % self.current_road_maneuver

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s"       % maneuver,
            "Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            "Distance traveled: % 7d m"    % self.distance_traveled,
            "Center deviance:   % 7.2f m"  % self.distance_from_center,
            "Avg center dev:    % 7.2f m"  % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h"  % (3.6 * self.speed_accum / self.step_count),
            "Environment: Lap "
        ])

        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)

            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5
        #print(action)
        # Take action
        brake = 0
        if action is not None:
            steer, throttle = [float(a) for a in action]
            #steer, throttle, brake = [float(a) for a in action]
            if throttle > 0:
                throttle = throttle
            elif throttle <= 0:
                brake = -1*(throttle)
                throttle = 0
            else:
                pass

            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)

        # Tick game
        self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation_DUV()
        self.viewer_image = self._get_viewer_image()
        self.observation_momentum["GNSS"] = self._get_observation_gnss("GNSS")
        self.observation_momentum["IMU"] = self._get_observation_imu("IMU")
        self.observation_momentum["Lidar"] = self._get_observation_lidar("Lidar")


        encoded_state = self.encode_state_fn(self)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Calculate deviation from center of the lane
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
        self.next_waypoint, self.next_road_maneuver       = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # DEBUG: Draw current waypoint
        #self.world.debug.draw_point(self.current_waypoint.transform.location, color=carla.Color(0, 255, 0), life_time=1.0)

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Get lap count
        self.laps_completed = (self.current_waypoint_index - self.start_waypoint_index) / len(self.route_waypoints)
        if self.laps_completed >= 3:
            # End after 3 laps
            self.terminal_state = True

        # Update checkpoint for training
        if self.is_training:
            checkpoint_frequency = 50 # Checkpoint frequency in meters
            self.checkpoint_waypoint_index = (self.current_waypoint_index // checkpoint_frequency) * checkpoint_frequency

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        return encoded_state, self.last_reward, self.terminal_state, { "closed": self.closed }

    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints)-1, skip+1):
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i+1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _get_observation_interlace(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None

        while self.observation_buffer_lidar is None:
            pass
        sensor_points, intensity = self.observation_buffer_lidar.copy()
        self.observation_buffer_lidar = None

        obs = self.interlace(seg,sensor_points,intensity)

        return obs

    def _get_observation_DUV(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None
        UV = self.toYUV_preprocess(seg)

        while self.observation_buffer_depth is None:
            pass
        depth = self.observation_buffer_depth.copy()
        self.observation_buffer_depth = None
        Y_ = self.preprocess_depth_frame(depth)

        DUV = self.final_representation(Y_,UV)

        return DUV

    def _get_observation_DUV_lidar(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None
        UV = self.toYUV_preprocess(seg)

        while self.observation_buffer_depth is None:
            pass
        depth = self.observation_buffer_depth.copy()
        self.observation_buffer_depth = None
        Y_ = self.preprocess_depth_frame(depth)

        DUV = self.final_representation(Y_,UV)

        while self.observation_buffer_lidar is None:
            pass
        sensor_points, intensity = self.observation_buffer_lidar.copy()
        self.observation_buffer_lidar = None

        obs = self.interlace(DUV,sensor_points,intensity)

        return obs

    def final_representation(self, Y_images, U_V_images):
        arr = np.dstack((Y_images, U_V_images))
        return arr

    def preprocess_depth_frame(self, frame):
        frame = frame[:, :, :1]                 # RGBA -> R
        return frame

    def preprocess_UV(self, frame):
        frame = frame[:, :, 1:3]
        #frame = frame.astype(np.float32) / 12.0 # [0, 12=num_classes] -> [0, 1]
        return frame

    def RGB2YUV(self, rgb):

        m = np.array([[ 0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [ 0.11400, 0.50000, -0.08131]])

        yuv = np.dot(rgb,m)
        yuv[:,:,1:]+=128.0
        return yuv.astype(int)

    def toYUV_preprocess(self, image):
        stepOne = self.RGB2YUV(image)
        stepTwo = self.preprocess_UV(stepOne)
        return stepTwo

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_observation_image_depth(self, image):
        self.observation_buffer_depth = image

    def _set_observation_image_lidar(self, image):
        self.observation_buffer_lidar = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_observation_sensor(self, name, image):
        self.observation_momentum_buffer[name] = image

    def _get_observation_gnss(self, name):
        while self.observation_momentum_buffer[name][0] is None:
            pass
        lat = self.observation_momentum_buffer[name][0]
        self.observation_momentum_buffer[name][0] = None

        while self.observation_momentum_buffer[name][1] is None:
            pass
        long = self.observation_momentum_buffer[name][1]
        self.observation_momentum_buffer[name][1] = None

        while self.observation_momentum_buffer[name][2] is None:
            pass
        alt = self.observation_momentum_buffer[name][2]
        self.observation_momentum_buffer[name][2] = None

        return lat, long, alt

    def _get_observation_lidar(self, name):
        while self.observation_momentum_buffer[name][0] is None:
            pass
        obs = self.observation_momentum_buffer[name][0].copy()
        self.observation_momentum_buffer[name][0] = None

        while self.observation_momentum_buffer[name][1] is None:
            pass
        intensity = self.observation_momentum_buffer[name][1].copy()
        self.observation_momentum_buffer[name][1] = None

        #final = np.append(obs, intensity)

        return obs

    def _get_observation_imu(self, name):
        while self.observation_momentum_buffer[name] is None:
            pass
        obs = self.observation_momentum_buffer[name]
        self.observation_momentum_buffer[name] = None
        return obs

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


class CarlaRouteEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, args, reward_fn=None, encode_state_fn=None,
                 synchronous=True, fps=30, action_smoothing=0.9):
        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        self.args = args
        width, height = [int(x) for x in self.args.viewer_res.split('x')]
        self.out_width, self.out_height = [int(x) for x in self.args.obs_res.split('x')]
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.out_width, self.out_height, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000 # m

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(60.0)
            self.client.load_world('Town07')

            # Create world wrapper
            self.world = World(self.client)

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                self.world.apply_settings(settings)

            # Get spawn location
            self.model_3 = "vehicle.tesla.model3"

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0], vehicle_type=self.model_3,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            self.dashcam = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                      sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
                                      camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)

            self.dashcam_seg = Camera(self.world, self.out_width, self.out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image_depth(e),
                                      sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
                                      camera_type="sensor.camera.depth", color_converter=carla.ColorConverter.LogarithmicDepth)


            self.lidar_sen = Lidar(self.world, self.args, self.dashcam, self.out_width, self.out_height, transform=camera_transforms["dashboard"],
                        attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_sensor("Lidar", e))

            self.gnss_sensor = GNSScheat(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_sensor("GNSS", e))

            self.imu_sensor = IMU(self.world, attach_to=self.vehicle, transform=camera_transforms["dashboard"], on_recv_image=lambda e: self._set_observation_sensor("IMU", e))


            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

            fov = self.dashcam.fov
            focal = self.out_width / (2.0 * np.tan(fov * np.pi /360.0))
            self.K = np.identity(3)
            self.K[0, 0] = self.K[1, 1] = focal
            self.K[0, 2] = self.out_width / 2.0
            self.K[1, 2] = self.out_height / 2.0

            self.VIRIDIS = np.array(cm.get_cmap('viridis').colors)
            self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])

            self.observation_momentum =  {key: None for key in ["GNSS","IMU","Lidar"]}
            self.observation_momentum_buffer = {key: None for key in ["GNSS", "IMU","Lidar"]}


        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()

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

        if self.args.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                im_array[
                    v_coord[i]-self.args.dot_extent : v_coord[i]+self.args.dot_extent,
                    u_coord[i]-self.args.dot_extent : u_coord[i]+self.args.dot_extent] = color_map[i]
        #print(im_array.shape)
        return im_array

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=True):
        # Create new route
        self.num_routes_completed = -1
        self.new_route()
        # Do a soft reset (teleport vehicle)

        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD

        self.observation_lidar = self.observation_buffer_lidar = None

        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training
        self.start_waypoint_index = self.current_waypoint_index

        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0

        return self.step(None)[0]

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()

        # Generate waypoints along the lap
        self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)]
        self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)

        self.world.tick()


    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW: maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:     maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:    maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT: maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.VOID:     maneuver = "VOID"
        else:                                                   maneuver = "INVALID(%s)" % self.current_road_maneuver

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s"       % maneuver,
            "Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            "Distance traveled: % 7d m"    % self.distance_traveled,
            "Center deviance:   % 7.2f m"  % self.distance_from_center,
            "Avg center dev:    % 7.2f m"  % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h"  % (3.6 * self.speed_accum / self.step_count),
            "Environment: Route"
        ])

        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Create new route on route completion
        if self.current_waypoint_index >= len(self.route_waypoints)-1:
            self.new_route()

        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)

            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5
        #print(action)
        # Take action
        brake = 0
        if action is not None:
            steer, throttle = [float(a) for a in action]
            if throttle > 0:
                throttle = throttle
            elif throttle <= 0:
                brake = -1*(throttle)
                throttle = 0
            else:
                pass

            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)

        # Tick game
        self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation_DUV()
        self.viewer_image = self._get_viewer_image()
        self.observation_momentum["GNSS"] = self._get_observation_gnss("GNSS")
        self.observation_momentum["IMU"] = self._get_observation_imu("IMU")
        self.observation_momentum["Lidar"] = self._get_observation_lidar("Lidar")

        encoded_state = self.encode_state_fn(self)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints)-1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance:
            self.terminal_state = True

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1
        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        return encoded_state, self.last_reward, self.terminal_state, { "closed": self.closed }

    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints)-1, skip+1):
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i+1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _get_observation_interlace(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None

        while self.observation_buffer_lidar is None:
            pass
        sensor_points, intensity = self.observation_buffer_lidar.copy()
        self.observation_buffer_lidar = None

        obs = self.interlace(seg,sensor_points,intensity)

        return obs

    def _get_observation_DUV(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None
        UV = self.toYUV_preprocess(seg)

        while self.observation_buffer_depth is None:
            pass
        depth = self.observation_buffer_depth.copy()
        self.observation_buffer_depth = None
        Y_ = self.preprocess_depth_frame(depth)

        DUV = self.final_representation(Y_,UV)

        return DUV

    def _get_observation_DUV_lidar(self):
        while self.observation_buffer is None:
            pass
        seg = self.observation_buffer.copy()
        self.observation_buffer = None
        UV = self.toYUV_preprocess(seg)

        while self.observation_buffer_depth is None:
            pass
        depth = self.observation_buffer_depth.copy()
        self.observation_buffer_depth = None
        Y_ = self.preprocess_depth_frame(depth)

        DUV = self.final_representation(Y_,UV)

        while self.observation_buffer_lidar is None:
            pass
        sensor_points, intensity = self.observation_buffer_lidar.copy()
        self.observation_buffer_lidar = None

        obs = self.interlace(DUV,sensor_points,intensity)

        return obs

    def final_representation(self, Y_images, U_V_images):
        arr = np.dstack((Y_images, U_V_images))
        return arr

    def preprocess_depth_frame(self, frame):
        frame = frame[:, :, :1]                 # RGBA -> R
        return frame

    def preprocess_UV(self, frame):
        frame = frame[:, :, 1:3]
        #frame = frame.astype(np.float32) / 12.0 # [0, 12=num_classes] -> [0, 1]
        return frame

    def RGB2YUV(self, rgb):

        m = np.array([[ 0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [ 0.11400, 0.50000, -0.08131]])

        yuv = np.dot(rgb,m)
        yuv[:,:,1:]+=128.0
        return yuv.astype(int)

    def toYUV_preprocess(self, image):
        stepOne = self.RGB2YUV(image)
        stepTwo = self.preprocess_UV(stepOne)
        return stepTwo

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_observation_image_depth(self, image):
        self.observation_buffer_depth = image

    def _set_observation_image_lidar(self, image):
        self.observation_buffer_lidar = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_observation_sensor(self, name, image):
        self.observation_momentum_buffer[name] = image

    def _get_observation_gnss(self, name):
        while self.observation_momentum_buffer[name][0] is None:
            pass
        lat = self.observation_momentum_buffer[name][0]
        self.observation_momentum_buffer[name][0] = None

        while self.observation_momentum_buffer[name][1] is None:
            pass
        long = self.observation_momentum_buffer[name][1]
        self.observation_momentum_buffer[name][1] = None

        while self.observation_momentum_buffer[name][2] is None:
            pass
        alt = self.observation_momentum_buffer[name][2]
        self.observation_momentum_buffer[name][2] = None

        return lat, long, alt

    def _get_observation_lidar(self, name):
        while self.observation_momentum_buffer[name][0] is None:
            pass
        obs = self.observation_momentum_buffer[name][0].copy()
        self.observation_momentum_buffer[name][0] = None

        while self.observation_momentum_buffer[name][1] is None:
            pass
        intensity = self.observation_momentum_buffer[name][1].copy()
        self.observation_momentum_buffer[name][1] = None

        #final = np.append(obs, intensity)

        return obs

    def _get_observation_imu(self, name):
        while self.observation_momentum_buffer[name] is None:
            pass
        obs = self.observation_momentum_buffer[name]
        self.observation_momentum_buffer[name] = None
        return obs

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
