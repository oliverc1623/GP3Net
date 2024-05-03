import carla
import random
import math
import cv2
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from bird_view_manager import BirdviewSensor
import matplotlib.pyplot as plt
from agents.navigation.global_route_planner import GlobalRoutePlanner
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from chauffeurnet import ObsManager
from carla_gym.core.task_actor.common.criteria import blocked, collision, outside_route_lane, route_deviation, run_stop_sign
from carla_gym.core.task_actor.common.criteria import encounter_light, run_red_light
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)

class CustomAgent(BasicAgent):
    def __init__(self, vehicle, route_plan, debug=False):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        super().__init__(vehicle, debug)
        self.vehicle=vehicle
        self.criteria_blocked = blocked.Blocked()
        self.criteria_collision = collision.Collision(self.vehicle, world)
        self.criteria_light = run_red_light.RunRedLight(self._map)
        self.criteria_encounter_light = encounter_light.EncounterLight()
        self.criteria_stop = run_stop_sign.RunStopSign(world)
        self.criteria_outside_route_lane = outside_route_lane.OutsideRouteLane(self._map, self.vehicle.get_location())
        self.criteria_route_deviation = route_deviation.RouteDeviation()
        self.route_plan = route_plan

    def run_step(self):
        """Execute one step of navigation."""
        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 1.0
        control.brake = 0.0
        control.hand_brake = False
        return control

class MotionPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, town):
        super().__init__()
        # Define action space
        self.action_space = spaces.Box(
            low = np.array([0.0, -1.0, 0.0]),
            high = np.array([1.0, 1.0, 1.0]),
            dtype = np.float32
        )
        # Define observation space
        self.observation_space = spaces.Dict({
            "bev": spaces.Box(
                low = 0,
                high = 255,
                shape = (192, 192, 3),
                dtype = np.uint8
            ),
            "odometrics": spaces.Dict({
                'throttle': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'steer': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                'brake': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'vel_xy': spaces.Box(low=-1e2, high=1e2, shape=(2,), dtype=np.float32),
            })
        })
        # Connect to the client and retrieve the world object
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.client.load_world('Town01')
        ego_bp = self.world.get_blueprint_library().find('vehicle.dodge.charger_2020')
        ego_bp.set_attribute('role_name','ego')
        route, point_a, point_b = self._get_route()
        self.ego_vehicle = self.world.spawn_actor(ego_bp, point_a)
        print('Ego is spawned')
        self._follow_vehicle()
        self.myAgent = CustomAgent(vehicle=self.ego_vehicle, route_plan=route)
        destination = point_b.location
        self.myAgent.set_destination(destination)
        self.myAgent.ignore_traffic_lights(active=True)

    def _follow_vehicle(self, offset=carla.Location(x=-6, z=2), pitch=-15):
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        location = transform.location + transform.get_forward_vector() * offset.x + carla.Location(z=offset.z)
        rotation = carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw + 0, roll=0)
        spectator.set_transform(carla.Transform(location, rotation))
        
    def _get_route(self):
        amap = self.world.get_map()
        sampling_resolution = 2
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        spawn_points = self.world.get_map().get_spawn_points()
        # Randomly sample Points A and B
        point_a, point_b = random.sample(spawn_points, 2)
        a = carla.Location(point_a.location)
        b = carla.Location(point_b.location)
        w1 = grp.trace_route(a, b)
        i = 0
        for w in w1:
            if i % 10 == 0:
                self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                persistent_lines=True)
            else:
                self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                persistent_lines=True)
            i += 1
        # Return route, point A, point B 
        return w1, point_a, point_b
    
    def step(self, action):
        pass

    def reset(seflf):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass
