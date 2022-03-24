import os
import sys
import glob
import argparse

#import the agents folder
try:
    sys.path.insert(0,'/home/lunet/cokm2/CARLA/PythonAPI/carla/agents')
except IndexError:
    pass

from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

#connnect to carla
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

#Class that shows the waypoints on the map
class WayPoints:
    def __init__(self, client, start, finish):
        self.sampling_resolution = 2
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.dao = GlobalRoutePlannerDAO(self.map, self.sampling_resolution)
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.numLocations = len(self.spawn_points)
        self.a = 0
        self.b = 0

#function displays routes, every point is denoted with an 'O'. Then every 5 points is an x
    def view(self):
        grp = GlobalRoutePlanner(self.dao)
        grp.setup()
        self.a = carla.Location(self.spawn_points[start].location)
        self.b = carla.Location(self.spawn_points[finish].location)
        w1 = grp.trace_route(self.a, self.b)
        i = 0
        for w in w1:
            if i % 10 == 0:
                self.world.debug.draw_string(w[0].transform.location,'O',draw_shadow=False,
                color=carla.Color(r=255,g=0,b=0), life_time=120.0,
                persistent_lines=True)
            else:
                self.world.debug.draw_string(w[0].transform.location,'X',draw_shadow=False,
                color = carla.Color(r=0, g=0, b=255), life_time=1000.0)
            i += 1

#display spawnpoints as 3D objects in the map
    def spawnPoints(self):
        j = 0
        while j < self.numLocations:
            i = self.spawn_points[j].location
            world.debug.draw_string(i,'{}'.format(j),draw_shadow=False,
                color=carla.Color(r=255,g=0,b=0), life_time=120.0,
               persistent_lines=True)
            j += 1

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
        '--Town',
        default='Town07',
        help='Select Map')
    argparser.add_argument(
        '--route',
        default='1,100',
        help='Select spawn locations')
    argparser.add_argument(
        '--waypoints',
        type=bool,
        default=False,
        help='Select true to view waypoints')
    args = argparser.parse_args()

#This script will run with commands.
    client = carla.Client(args.host, args.port)
    start, finish = [int(x) for x in args.route.split(',')]

    try:
        client.set_timeout(2.0)
        client.load_world(args.Town)
        world = client.get_world()
    except Exception as e:
        print(e, "Could not find server")

    try:
        obj1 = WayPoints(client, start, finish)
        if args.waypoints == True:
            print("view route")
            obj1.view()
        else:
            print("view spawnpoints")
            obj1.spawnPoints()


    except RuntimeError:
            print("Not Valid")
