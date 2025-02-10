import networkx as nx
import osmnx as ox


def load_map(file_path):
    """Load the pre-downloaded map from a .graphml file."""
    return ox.load_graphml(file_path)


def find_nearest_node(graph, latitude, longitude):
    """Find the nearest graph node to given latitude and longitude."""
    return ox.distance.nearest_nodes(graph, longitude, latitude)


def find_shortest_route(graph, start_coords, dest_coords):
    """Find the shortest route using Dijkstra's algorithm."""
    start_node = find_nearest_node(graph, *start_coords)
    dest_node = find_nearest_node(graph, *dest_coords)

    # Compute shortest path
    shortest_path = nx.shortest_path(graph, start_node, dest_node, weight="length")
    return shortest_path


def generate_directions(graph, path):
    """Generate step-by-step directions from the path."""
    if not path:
        return ["No route found."]

    directions = []
    for i in range(len(path) - 1):
        start_point = (graph.nodes[path[i]]['y'], graph.nodes[path[i]]['x'])
        end_point = (graph.nodes[path[i + 1]]['y'], graph.nodes[path[i + 1]]['x'])
        directions.append(f"Move from {start_point} to {end_point}")

    return directions


def main(map_file, start_coords, dest_coords):
    graph = load_map(map_file)
    path = find_shortest_route(graph, start_coords, dest_coords)
    directions = generate_directions(graph, path)

    print("\n".join(directions))


# Example usage
if __name__ == "__main__":
    map_file = "offline_map.graphml"  # Pre-downloaded map file
    start_coords = (37.7749, -122.4194)  # Example: San Francisco (latitude, longitude)
    dest_coords = (37.8044, -122.2711)  # Example: Oakland (latitude, longitude)

    main(map_file, start_coords, dest_coords)
