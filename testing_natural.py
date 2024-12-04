import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

def compute_natural_neighbor_weights(electrode_positions, query_point):
    """
    Compute Natural Neighbor weights for a query point based on electrode positions.

    :param electrode_positions: Array of electrode positions (Nx2 for 2D coordinates).
    :param query_point: The point at which to compute weights.
    :return: Dictionary with electrode indices as keys and weights as values.
    """

    # Combine electrode positions with the query point
    all_points = np.vstack([electrode_positions, query_point])
    vor = Voronoi(all_points)

    # Find the region of the query point (last point added)
    query_region_index = vor.point_region[-1]
    query_region = vor.regions[query_region_index]

    if -1 in query_region or len(query_region) < 3:
        # Unbounded region or not enough vertices
        raise ValueError("Query point lies outside the convex hull or is unbounded.")

    # Compute areas of the query point's Voronoi region and neighbors
    weights = {}
    query_cell_vertices = vor.vertices[query_region]

    for i, region_index in enumerate(vor.point_region[:-1]):  # Exclude query point
        neighbor_region = vor.regions[region_index]
        if -1 in neighbor_region or len(neighbor_region) < 3:
            continue  # Skip unbounded or invalid regions

        # Find shared vertices
        shared_vertices_indices = set(query_region).intersection(neighbor_region)
        if not shared_vertices_indices:
            continue  # No shared vertices, skip

        # Get shared vertices
        shared_vertices = vor.vertices[list(shared_vertices_indices)]

        # Calculate area of the shared region (assume 2D, use polygon area formula)
        if len(shared_vertices) < 3:
            continue  # Not enough points to form a polygon
        shared_area = 0.5 * np.abs(
            np.dot(shared_vertices[:, 0], np.roll(shared_vertices[:, 1], 1)) -
            np.dot(shared_vertices[:, 1], np.roll(shared_vertices[:, 0], 1))
        )
        weights[i] = shared_area

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {key: value / total_weight for key, value in weights.items()}
    else:
        raise ValueError("No valid weights found for the query point.")

    return weights


# Example electrode positions (x, y) and a query point
electrode_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
query_point = np.array([0.5, 0.5])

# Compute weights
weights = compute_natural_neighbor_weights(electrode_positions, query_point)
print("Natural Neighbor Weights:", weights)

vor = Voronoi(np.vstack([electrode_positions, query_point]))
fig, ax = plt.subplots()
voronoi_plot_2d(vor, ax=ax, show_vertices=False)

# Highlight query point and electrode positions
ax.plot(query_point[0], query_point[1], 'ro', label='Query Point')
ax.plot(electrode_positions[:, 0], electrode_positions[:, 1], 'bo', label='Electrodes')

# Label regions and points
for idx, pos in enumerate(electrode_positions):
    ax.text(pos[0], pos[1], str(idx), color="blue")

plt.legend()
plt.title("Natural Neighbor Debug Visualization")
plt.show()
