# Anjian
# This cost is designed for multiple goals with different effective time

import torch
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost
from point import Point

class MultiGoalCost(Cost):

    def __init__(self, position_indices,
                 point_list, time_list,
                 max_distance, outside_weight=0.1,
                 name=""):
        self._x_index, self._y_index = position_indices
        self._point_list = point_list # may contain multiple point
        # may contain different effective time, format: [[start_time, end_time]]
        self._time_list = time_list
        self._max_squared_distance = max_distance ** 2
        self._outside_weight = outside_weight

        # Define if the robot has already catch the goal
        self._catch = False

        super(MultiGoalCost, self).__init__(name)

    def __call__(self, x, k=0):
        if k == 0:
            self._catch = False
            self._print = False

        relative_squared_distance = np.inf

        if self._catch:
            if self._print == False:
                print("trajectory has reach the goal at time", k)
            self._print = True
            # return torch.zeros(
            #     1, 1, requires_grad=True).double()

        # First check in the current time, which goal is in effect
        for ii in range(len(self._time_list)):
            if k >= self._time_list[ii][0] and k < self._time_list[ii][1]:
                # Compute relative distance.
                dx = x[self._x_index, 0] - self._point_list[ii].x
                dy = x[self._y_index, 0] - self._point_list[ii].y
                curr_relative_squared_distance = dx * dx + dy * dy
                relative_squared_distance = min(curr_relative_squared_distance,
                                                relative_squared_distance)

        if relative_squared_distance < 2:
            self._catch = True

        if relative_squared_distance == np.inf:
            # Means no goal is in effect
            # print("there is no goal at this time!")
            # exit()
            return torch.zeros(
                1, 1, requires_grad=True).double()
        else:
            return -relative_squared_distance * torch.ones(
                1, 1, requires_grad=True).double()


    def render(self, ax=None):
        """ Render this obstacle on the given axes. """
        if np.isinf(self._max_squared_distance):
            radius = 1.0
        else:
            radius = np.sqrt(self._max_squared_distance)

        for ii in range(len(self._point_list)):
            circle = plt.Circle(
                (self._point_list[ii].x, self._point_list[ii].y), radius,
                color="g", fill=True, alpha=0.75)
            ax.add_artist(circle)
            ax.text(self._point_list[ii].x + 1.25, self._point_list[ii].y + 1.25,
                    "goal in [{:.0f}, {:.0f}]".format(self._time_list[ii][0], self._time_list[ii][1]),
                    fontsize=10)