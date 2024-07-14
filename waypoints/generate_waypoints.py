import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from collections import defaultdict

def generate_waypoints(matrix, partitions, start, k):
    def get_top_k_values_indices(matrix, k):
        flat_indices = np.argpartition(matrix.flatten(), -k)[-k:]
        return np.unravel_index(flat_indices, matrix.shape)

    def sample_waypoints(top_k_indices, start, k):
        if start not in top_k_indices:
            top_k_indices = [start] + top_k_indices
        np.random.shuffle(top_k_indices)
        selected_indices = top_k_indices[:k]
        if start not in selected_indices:
            selected_indices[0] = start
        return selected_indices

    def calculate_distance_matrix(points):
        dist_matrix = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(len(points)):
                dist_matrix[i, j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
        return dist_matrix

    def process_partition(matrix, partition, start, k):
        partition_indices = np.argwhere(partition)
        partition_values = matrix[partition]
        top_k_indices = get_top_k_values_indices(partition_values, k)
        local_top_k_indices = [tuple(partition_indices[idx]) for idx in zip(*top_k_indices)]
        waypoints = sample_waypoints(local_top_k_indices, start, k)
        distance_matrix = calculate_distance_matrix(waypoints)
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
        ordered_waypoints = [(int(p[0]), int(p[1])) for p in [waypoints[i] for i in permutation]]
        return ordered_waypoints

    N = np.max(partitions) + 1
    local_starts = defaultdict(list)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            local_starts[partitions[i, j]].append((i, j))

    region_waypoints = {}
    for region in range(N):
        local_start = start if start in local_starts[region] else local_starts[region][0]
        partition_mask = (partitions == region)
        region_waypoints[region] = process_partition(matrix, partition_mask, local_start, k)

    return region_waypoints

if __name__ == '__main__':
    matrix = np.array([
        [0,  2, 9, 10],
        [1,  0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ])

    partitions = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ])

    start = (0, 0)
    k = 3

    region_waypoints = generate_waypoints(matrix, partitions, start, k)

    print("Region Waypoints:")
    for region, waypoints in region_waypoints.items():
        print(f"Region {region}: {waypoints}")
