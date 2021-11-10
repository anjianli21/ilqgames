import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import sys
import signal
import time
from scipy.linalg import expm

from ilq_solver import ILQSolver
from visualizer import Visualizer
from logger import Logger

class RHCPlanner(object):
    def __init__(self,
                 dynamics,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling=0.05,
                 max_iteration=50,
                 reference_deviation_weight=None,
                 logger=None,
                 visualizer=None,
                 u_constraints=None,
                 simulation_horizon=50):
        """
        Receding horizon control planner using the LQ policy.
        """
        self._dynamics = dynamics
        self._player_costs = player_costs
        self._x0 = x0
        self._Ps = Ps
        self._alphas = alphas
        self._alpha_scaling = alpha_scaling
        self._max_iteration = max_iteration
        self._reference_deviation_weight = reference_deviation_weight
        self._u_constraints = u_constraints
        self._simulation_horizon = simulation_horizon
        self._num_players = len(player_costs)

        # Set up trajectory vectors
        self._xs = [self._x0]
        self._us = [[] for ii in range(self._num_players)]

        # Set up visualizer.
        self._visualizer = visualizer
        self._logger = logger

        # Log some of the paramters.
        if self._logger is not None:
            self._logger.log("_simulation_horizon", self._simulation_horizon)
            self._logger.log("x0", self._x0)

    def run(self, verbose):
        """ Run the algorithm for the specified parameters. """
        iteration = 0

        while iteration <= self._simulation_horizon:

            if verbose['planner']:
                print('****** Simulation time = ', iteration, ' ******')

            # Initialize the solver.
            solver = ILQSolver(self._dynamics,
                               self._player_costs,
                               self._x0,
                               self._Ps,
                               self._alphas,
                               self._alpha_scaling,
                               self._max_iteration,
                               self._reference_deviation_weight,
                               logger=None,
                               visualizer=None,
                               u_constraints=self._u_constraints)

            # Solve ILQ Game.
            solver.run(verbose['solver'])

            # Simulating the system forward.
            _, us, _ = solver._compute_operating_point()
            u = [us[ii][1] for ii in range(self._num_players)]
            self._x0 = self._dynamics.integrate(self._x0, u)
            self._xs.append(self._x0)
            for ii in range(self._num_players):
                self._us[ii].append(u[ii])

            # Visualization.
            if (self._visualizer is not None) and \
                    ((np.mod(iteration, 1) == 0) or (iteration == self._simulation_horizon)):
                traj = {"xs": self._xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = self._us[ii]

                self._visualizer.add_trajectory(iteration, traj)
                for ii in range(self._num_players):
                    self._visualizer.plot_controls(ii + 1)
                    plt.pause(0.001)
                    plt.clf()
                self._visualizer.plot()
                plt.pause(0.001)
                plt.clf()

            # Log everything.
            if self._logger is not None:
                self._logger.log("xs", self._xs)
                self._logger.log("us", self._us)
                self._logger.dump()

            # Update the member variables (for initialization).
            # Ps_tmp = solver._Ps[::-1]
            # Ps_tmp = Ps_tmp[1:]
            # Ps_tmp.append(Ps_tmp[-1])
            # self._Ps = Ps_tmp
            self._Ps = np.zeros_like(solver._Ps)
            # alphas_tmp = solver._alphas[::-1]
            # alphas_tmp = alphas_tmp[1:]
            # alphas_tmp.append(alphas_tmp[-1])
            # self._alphas = alphas_tmp
            self._alphas = np.zeros_like(solver._alphas)

            iteration += 1