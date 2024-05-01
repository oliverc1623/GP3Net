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


class MotionPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low = np.array([0.0, -1.0, 0.0]),
            high = np.array([1.0, 1.0, 1.0]),
            dtype = np.float32
        )
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