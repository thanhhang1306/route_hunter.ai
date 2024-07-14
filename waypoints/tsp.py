import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

class WaypointGenerator:

    """
    Generate waypoint given a 2x2 grid of cells with meta-values.

    Inputs:
    - cells: 2x2 grid of cells with integer? values representing the meta-values of the cells

    """

    def __init__(self, cells):
        self.cells = cells
        self.top_points = None

    def _generate_n_top_points(self, n, cells):
        pass

    def generate_waypoints(self, n):

        """
        Generate waypoints given the for WaypointGenerator object.

        Inputs:
        - n: number of waypoints

        Outputs:
        - waypoints: array of waypoints generated from the cells

        """

        # generate top points
        self.top_points = self._generate_n_top_points(n, self.cells)

        # sample N waypoints from the top points


        pass

distance_matrix = np.array([
    [0,  5, 4, 10],
    [5,  0, 8,  5],
    [4,  8, 0,  3],
    [10, 5, 3,  0]
])
permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
