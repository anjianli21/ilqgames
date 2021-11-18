
################################################################################
#
# Script to run an obstacle avoidance example for the TwoPlayerUnicycle4D.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

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
from reach_avoid_solver import ReachAvoidSolver

# General parameters.
TIME_HORIZON = 3   # s
TIME_RESOLUTION = 0.1 # s, default: 0.1
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/one_player_dynamic_goal_obs/"
MAX_V = 25.0 # m/s
MAX_A = 10.0 # m/s^2
MAX_W = 1.0 # rad/s

# Solver parameter
alpha_scaling = 0.01
max_iteration = 400

# Create dynamics.
car1 = Unicycle4D(T=TIME_RESOLUTION)
dynamics = ProductMultiPlayerDynamicalSystem(
    [car1], T=TIME_RESOLUTION)

# Choose an initial state and control laws.
theta0 = np.pi / 3 # 60 degree heading
v0 = 5.0             # 5 m/s initial speed

# New start
x0 = np.array([[20.0],
               [20.0],
               [theta0],
               [v0]])


mult = 0.0
P1s =  [mult * np.array([[0, 0, 0, 0],
                         [0, 0, 1, 0]])] * HORIZON_STEPS

alpha1s = [np.zeros((dynamics._u_dims[0], 1))] * HORIZON_STEPS

###################################
# Test Multi goal cost
# goal_centers = [Point(30.0, 40.0), Point(40.0, 30.0), Point(50, 20)]
# goal_times = [[9, 10], [19, 20], [29, 30]]

# Test single goal cost
goal_centers = [Point(50.0, 60.0)]
goal_times = [[29, 30]]

goal_cost = MultiGoalCost(position_indices=(0, 1),
                          point_list=goal_centers,
                          time_list=goal_times,
                          max_distance=np.inf,
                          name="goal")

###################################
# Current test obstacles
# obstacle_centers = [Point(30, 30), Point(40, 40), Point(50, 30)]
# obstacle_radii = [6.0, 6.0, 6.0]
# obstacle_time = [[0, 10], [10, 20], [20, 30]]

# Single obstacles
obstacle_centers = [Point(40, 40)]
obstacle_radii = [10.0]
obstacle_time = [[0, 30]]

obstacle_costs = [ObstacleCost(
    position_indices=(0, 1), point=p, max_distance=r,
    name="obstacle_%f_%f" % (p.x, p.y),
    start_after_time=t[0], start_before_time=t[1])
                  for p, r, t in zip(obstacle_centers, obstacle_radii, obstacle_time)]

w_cost = QuadraticCost(dimension=0, origin=0, name="w_cost")
a_cost = QuadraticCost(dimension=1, origin=0, name="a_cost")
w_cost_upper = SemiquadraticCost(dimension=0, threshold=MAX_W, oriented_right=True, name="w_cost_upper")
w_cost_lower = SemiquadraticCost(dimension=0, threshold=-MAX_W, oriented_right=False, name="w_cost_lower")
a_cost_upper = SemiquadraticCost(dimension=1, threshold=MAX_A, oriented_right=True, name="a_cost_upper")
a_cost_lower = SemiquadraticCost(dimension=1, threshold=-MAX_A, oriented_right=False, name="a_cost_lower")

# Add light quadratic around original values for theta/v.
v_cost_upper = SemiquadraticCost(
    dimension=3, threshold=MAX_V, oriented_right=True, name="v_cost_upper")
v_cost_lower = SemiquadraticCost(
    dimension=3, threshold=0, oriented_right=False, name="v_cost_lower")


# OBSTACLE_WEIGHT = 100.0 # HJI: 100
OBSTACLE_WEIGHT = 200.0 # HJI: 100
GOAL_WEIGHT = 100.0 # HJI: 400
U_WEIGHT = 1.0 # HJI: 100

V_WEIGHT = 100.0

# Build up total costs for both players. This is basically a zero-sum game.
player1_cost = PlayerCost()
player1_cost.add_cost(goal_cost, "x", -GOAL_WEIGHT)
for cost in obstacle_costs:
    player1_cost.add_cost(cost, "x", OBSTACLE_WEIGHT)

player1_cost.add_cost(v_cost_upper, "x", V_WEIGHT)
player1_cost.add_cost(v_cost_lower, "x", V_WEIGHT)
player1_cost.add_cost(w_cost, 0, U_WEIGHT)
player1_cost.add_cost(a_cost, 0, U_WEIGHT)
# player1_cost.add_cost(a_cost_upper, 0, U_WEIGHT)
# player1_cost.add_cost(a_cost_lower, 0, U_WEIGHT)
# player1_cost.add_cost(w_cost_upper, 0, U_WEIGHT)
# player1_cost.add_cost(w_cost_lower, 0, U_WEIGHT)
#################################
# Control constraints
player1_u_constraint = BoxConstraint(np.array([[-MAX_W, -MAX_A]]).T, np.array([[MAX_W, MAX_A]]).T)


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
solver = ReachAvoidSolver(dynamics,
                        [player1_cost],
                        x0,
                        [P1s],
                        [alpha1s],
                        alpha_scaling,
                        max_iteration,
                        None,
                        logger,
                        visualizer,
                        u_constraints=[player1_u_constraint],
                        TOLERANCE_PERCENTAGE=1e-7)

def handle_sigint(sig, frame):
    print("SIGINT caught! Saving log data...")
    logger.dump()
    print("...done, exiting!")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

solver.run()