
################################################################################
#
# Script to run an obstacle avoidance example for the TwoPlayerUnicycle4D.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from two_player_unicycle_4d import TwoPlayerUnicycle4D
from ilq_solver import ILQSolver
from point import Point
from proximity_cost import ProximityCost
from obstacle_cost import ObstacleCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from player_cost import PlayerCost
from box_constraint import BoxConstraint
from visualizer import Visualizer
from logger import Logger
import sys
import signal

from unicycle_4d import Unicycle4D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from multi_goal_cost import MultiGoalCost

# General parameters.
TIME_HORIZON = 3   # s
TIME_RESOLUTION = 0.1 # s, default: 0.1
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/one_player_dynamic_goal_obs/"
# MAX_V = 15.0 # m/s
MAX_V = 5.0 # m/s

# Create dynamics.
car1 = Unicycle4D(T=TIME_RESOLUTION)
dynamics = ProductMultiPlayerDynamicalSystem(
    [car1], T=TIME_RESOLUTION)

# Choose an initial state and control laws.
theta0 = np.pi / 3 # 60 degree heading
v0 = 5.0             # 5 m/s initial speed


theta0 = np.pi / 3
v0 = 5.0

# x0 = np.array([[30.0],
#                [30.0],
#                [theta0],
#                [v0]])

# New start
x0 = np.array([[20.0],
               [20.0],
               [theta0],
               [v0]])

# New start
x0 = np.array([[20.0],
               [20.0],
               [theta0],
               [v0]])

mult = 0.0
P1s =  [mult * np.array([[0, 0, 0, 0],
                         [0, 0, 1, 0]])] * HORIZON_STEPS

alpha1s = [np.zeros((dynamics._u_dims[0], 1))] * HORIZON_STEPS


# Create the example environment. It will have a couple of circular obstacles
# laid out like this:
#                           x goal
#
#                      ()
#               ()
#                            ()
#
#          x start

###################################
# goal = Point(55.0, 70.0)
# goal_centers = [Point(50.0, 50.0), Point(40.0, 20.0)]
# goal_centers = [Point(40.0, 20.0)]


# goal_cost = ProximityCost(position_indices=(0, 1),
#                           point=goal,
#                           max_distance=np.inf,
#                           apply_after_time=HORIZON_STEPS - 1,
#                           # apply_after_time=0,
#                           name="goal")

# goal_costs = [ProximityCost(position_indices=(0, 1),
#                             point=p,
#                             max_distance=np.inf,
#                             apply_after_time=HORIZON_STEPS - 1,
#                             # apply_after_time=0,
#                             name="goal_%f_%f" % (p.x, p.y))
#               for p in goal_centers]

# Test Multi goal cost
# goal_centers = [Point(50.0, 50.0), Point(40.0, 20.0)]
# goal_times = [[28, 29], [14, 15]]

# # Test consecutive goal 1
# goal_centers = [Point(30.0, 20.0), Point(40.0, 30.0), Point(50, 40)]
# goal_times = [[9, 10], [19, 20], [29, 30]]

# Test consecutive goal 2
# We can try different MAX_V: 15, reach for 2nd goal, 30 reach for 1st goal

goal_centers = [Point(30.0, 40.0), Point(40.0, 30.0), Point(50, 20)]
# goal_times = [[9, 10], [19, 20], [29, 30]]
# goal_times = [[9, 10], [14, 15], [29, 30]]
goal_times = [[29, 30], [14, 15], [9, 10]]

# goal_centers = [Point(40.0, 30.0), Point(50, 20)]
# goal_times = [[14, 15], [29, 30]]

# Test consecutive goal 3
# goal_centers = [Point(30.0, 40.0), Point(35, 35), Point(40.0, 30.0), Point(45.0, 25.0), Point(50, 20)]
# # goal_times = [[0, 10], [10, 15], [15, 20], [20, 30]]
# goal_times = [[29, 30], [24, 25], [19, 20], [14, 15], [9, 10]]

# # Original goal
# goal_centers = [Point(55.0, 70.0)]
# goal_times = [[29, 30]]

goal_cost = MultiGoalCost(position_indices=(0, 1),
                          point_list=goal_centers,
                          time_list=goal_times,
                          max_distance=np.inf,
                          name="goal")

###################################
# Original obstacles
# obstacle_centers = [Point(80.0, 15.0),
#                     Point(45.0, 45.0), Point(15.0, 60.0)]
# obstacle_radii = [8.0, 8.0, 8.0]
# obstacle_time = [[0, 30], [0, 30], [0, 30]]

# Current test obstacles
obstacle_centers = [Point(30, 30), Point(40, 40), Point(50, 30)]
obstacle_radii = [6.0, 6.0, 6.0]
obstacle_time = [[0, 10], [10, 20], [20, 30]]

# for center in range(30, 60, 10):
#     obstacle_centers.append(Point(center+0.0, 80.0 - center))
#     obstacle_radii.append(6.0)
# obstacle_time = [[0, 10], [10, 20], [20, 30]]

# obstacle_centers = [Point(45.0, 45.0)]
# obstacle_radii = [8.0]


obstacle_costs = [ObstacleCost(
    position_indices=(0, 1), point=p, max_distance=r,
    name="obstacle_%f_%f" % (p.x, p.y),
    start_after_time=t[0], start_before_time=t[1])
                  for p, r, t in zip(obstacle_centers, obstacle_radii, obstacle_time)]

w_cost = QuadraticCost(dimension=0, origin=0, name="w_cost")
a_cost = QuadraticCost(dimension=1, origin=0, name="a_cost")

dvx_cost = QuadraticCost(dimension=0, origin=0, name="dvx_cost")
dvy_cost = QuadraticCost(dimension=1, origin=0, name="dvy_cost")

# Add light quadratic around original values for theta/v.
v_cost_upper = SemiquadraticCost(
    dimension=3, threshold=MAX_V, oriented_right=True, name="v_cost_upper")
v_cost_lower = SemiquadraticCost(
    dimension=3, threshold=0, oriented_right=False, name="v_cost_lower")

# control constraint
player1_constraint = BoxConstraint(np.array([[0]]).T, np.array([[MAX_V]]).T)

# OBSTACLE_WEIGHT = 100.0 # HJI: 100
OBSTACLE_WEIGHT = 200.0 # HJI: 100
GOAL_WEIGHT = 100.0 # HJI: 400
D_WEIGHT = 1000.0 # HJI: 1000
U_WEIGHT = 1.0 # HJI: 100

V_WEIGHT = 1.0
# V_WEIGHT = 50.0 # HJI: 50

# Build up total costs for both players. This is basically a zero-sum game.
player1_cost = PlayerCost()
player1_cost.add_cost(goal_cost, "x", -GOAL_WEIGHT)
for cost in obstacle_costs:
    player1_cost.add_cost(cost, "x", OBSTACLE_WEIGHT)

player1_cost.add_cost(v_cost_upper, "x", V_WEIGHT)
player1_cost.add_cost(v_cost_lower, "x", V_WEIGHT)
player1_cost.add_cost(w_cost, 0, U_WEIGHT)
player1_cost.add_cost(a_cost, 0, U_WEIGHT)


# Visualizer.
# visualizer = None
visualizer = Visualizer(
    [(0, 1)],
    [goal_cost] + obstacle_costs,
    [".-b"],
    1,
    False,
    plot_lims=[0, 100, 0, 100])

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

path_to_logfile = os.path.join(LOG_DIRECTORY, "dynamic_goal_obs_example.pkl")
if len(sys.argv) > 1:
    path_to_logfile = os.path.join(LOG_DIRECTORY, sys.argv[1])

print("Saving log file to {}...".format(path_to_logfile))
logger = Logger(path_to_logfile)

# Set up ILQSolver.
solver = ILQSolver(dynamics,
                   [player1_cost],
                   x0,
                   [P1s],
                   [alpha1s],
                   # 0.01, # 0.01
                   0.1,
                   2000,
                   None,
                   logger,
                   visualizer,
                   # u_constraints=[player1_constraint],
                   TOLERANCE_PERCENTAGE=1e-7)

def handle_sigint(sig, frame):
    print("SIGINT caught! Saving log data...")
    logger.dump()
    print("...done, exiting!")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

solver.run()