import random
import time
import collections
import math
import numpy as np
import weakref
import pygame
from queue import Queue
from queue import Empty
import numpy as np

import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll
        )
    )

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate-1] + u"\u2026") if len(name) > truncate else name

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi: angle -= 2 * np.pi
    elif angle <= -np.pi: angle += 2 * np.pi
    return angle

def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

def deg_to_rad(val):
    return val * np.pi / 180

camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}



#===============================================================================
# CarlaActorBase
#===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)

#===============================================================================
# CollisionSensor
#===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


#===============================================================================
# LaneInvasionSensor
#===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)


class GNSS(CarlaActorBase):
    def __init__(self, world, attach_to=None, transform=carla.Transform(), on_recv_image=None):
        self.on_recv_image = on_recv_image
        std_dev = 0.1
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute('sensor_tick', '0.1')
        gnss_bp.set_attribute('noise_lat_stddev', str(std_dev))
        gnss_bp.set_attribute('noise_lon_stddev', str(std_dev))
        gnss_tf = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))

        weak_self = weakref.ref(self)
        self.period = 0.1
        self.gnss_time = 0

        self.gnss = world.try_spawn_actor(gnss_bp, gnss_tf, attach_to.get_carla_actor())
        print("Spawned actor \"{}\"".format(self.gnss.type_id))
        self.gnss.listen(lambda data: GNSS.gnss_listen(weak_self, data))

        super().__init__(world, self.gnss)

    @staticmethod
    def gnss_listen(weak_self, data):
        self = weak_self()
        if not self:
            return
        if(data.timestamp - self.gnss_time < self.period):
            return
        if callable(self.on_recv_image):
            lat = data.latitude
            lon = data.longitude
            alt = data.altitude

            output = [lat, lon, alt]
            self.on_recv_image(output)

class GNSScheat(CarlaActorBase):
    def __init__(self, world, attach_to=None, transform=carla.Transform(), on_recv_image=None):
        self.on_recv_image = on_recv_image
        std_dev = 0.1
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute('sensor_tick', '0.1')
        gnss_bp.set_attribute('noise_lat_stddev', str(std_dev))
        gnss_bp.set_attribute('noise_lon_stddev', str(std_dev))
        gnss_tf = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))

        weak_self = weakref.ref(self)
        self.period = 0.1
        self.gnss_time = 0
        self.vehicle = attach_to.get_carla_actor()

        self.gnss = world.try_spawn_actor(gnss_bp, gnss_tf, attach_to.get_carla_actor())
        print("Spawned actor \"{}\"".format(self.gnss.type_id))
        self.gnss.listen(lambda data: GNSS.gnss_listen(weak_self, data))

        super().__init__(world, self.gnss)

    @staticmethod
    def gnss_listen(weak_self, data):
        self = weak_self()
        if not self:
            return
        if(data.timestamp - self.gnss_time < self.period):
            return
        if callable(self.on_recv_image):
            t = self.vehicle.get_transform()
            lat = t.location.x
            lon = t.location.y
            alt = t.location.z

            output = [lat, lon, alt]
            self.on_recv_image(output)

    def destroy(self):
        super().destroy()

class IMU(CarlaActorBase):
    def __init__(self, world, attach_to=None, transform=carla.Transform(), on_recv_image=None):
        self.on_recv_image = on_recv_image
        self.period = 0.1
        accel_std_dev = 0
        gyro_std_dev = 0

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute('sensor_tick', '0.1')
        imu_bp.set_attribute('noise_gyro_stddev_y',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_gyro_stddev_x',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_gyro_stddev_z',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_y', str(accel_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_x', str(accel_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_z', str(accel_std_dev))
        imu_tf = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        weak_self = weakref.ref(self)
        self.imu_time = 0
        self.imu_per = 0.1

        try:
            self.imu = world.try_spawn_actor(imu_bp, imu_tf, attach_to.get_carla_actor())
        except Exception as e:
            print(e)
        print("Spawned actor \"{}\"".format(self.imu.type_id))
        self.imu.listen(lambda data: IMU.imu_listen(weak_self, data))

        super().__init__(world, self.imu)

    @staticmethod
    def imu_listen(weak_self, data):
        self = weak_self()
        if not self:
            return
        if(data.timestamp - self.imu_time < self.imu_per):
            return
        if callable(self.on_recv_image):
            self.on_recv_image(data)

    def destroy(self):
        super().destroy()

class Lidar(CarlaActorBase):
    def __init__(self, world, args, dashcam, width, height, transform=carla.Transform(),
    attach_to=None, on_recv_image=None):
        self.on_recv_image = on_recv_image
        self.on_recv_intensity = None
        self.dashcamObj = dashcam
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")

        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        weak_self = weakref.ref(self)
        self.actor = world.try_spawn_actor(lidar_bp, transform, attach_to.get_carla_actor())
        print("Spawned actor \"{}\"".format(self.actor.type_id))
        self.actor.listen(lambda image: Lidar.process_camera_input(weak_self, image))

        super().__init__(world, self.actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            p_cloud_size = len(image)
            p_cloud = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            #cloud shape intensity
            intensity = np.array(p_cloud[:,3])

            #print(intensity)

            #point cloud in lidar sensor array
            local_lidar_points = np.array(p_cloud[:, :3]).T

            #manipulate so it can be multiplied by a 4x4 matrix
            local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

            #get coordinates
            lidar_2_world = self.actor.get_transform().get_matrix()

            #4x4 matrix transform lidar space to worldspace
            world_points = np.dot(lidar_2_world, local_lidar_points)

            #use 4x4 matrix for camera transformation
            world_2_camera = np.array(self.dashcamObj.get_carla_actor().get_transform().get_inverse_matrix())

            #transform points from world space to camera space
            sensor_points = np.dot(world_2_camera, world_points)
            output = [sensor_points, intensity]
            self.on_recv_image(output)
            #self.on_recv_intensity(intensity)

    def destroy(self):
        super().destroy()
#===============================================================================
# Camera
#===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))
        self.fov = camera_bp.get_attribute("fov").as_float()

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

class CameraLidar(CarlaActorBase):
    def __init__(self, world, args, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None):
        self.on_recv_image = on_recv_image
        self.color_converter = carla.ColorConverter.CityScapesPalette
        weak_self = weakref.ref(self)
        actors = []

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))
        self.fov = camera_bp.get_attribute("fov").as_float()

        # Create and setup camera actor
        cam_seg = world.try_spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())

        print("Spawned actor \"{}\"".format(cam_seg.type_id))
        actors.append(cam_seg)

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        lidar_sen = world.try_spawn_actor(lidar_bp, transform, attach_to=attach_to.get_carla_actor())
        print("Spawned actor \"{}\"".format(lidar_sen.type_id))
        actors.append(lidar_sen)
        lidar_queue = Queue()
        lidar_sen.listen(lambda data: CameraLidar.sensor_callback(data, lidar_queue))
        cam_seg.listen(lambda image: Camera.process_camera_input(weak_self, image))

        super().__init__(world, cam_seg)

    def sensor_callback(data, queue):
        queue.put(data)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()




#===============================================================================
# Vehicle
#===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform,
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz2017"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().filter(vehicle_type)[0]

        #vehicle_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print(actor)
        print("Spawned actor \"{}\"".format(actor.type_id))

        try:
            map_geo = world.get_map().transform_to_geolocation(carla.Location(0,0,0))
        except Exception as e:
            print(e)

        self.geo_centre_lat = deg_to_rad(map_geo.latitude)
        self.geo_centre_lon = deg_to_rad(map_geo.longitude)
        self.geo_centre_alt = map_geo.altitude

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

#===============================================================================
# World
#===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)
