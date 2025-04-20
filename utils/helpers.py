import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import seaborn as sns
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.cluster import AgglomerativeClustering, KMeans



##Solver for TSP (Travelling Salesman Problem); implemented
# in a bad way, but that does not matter for now; it's just a
# way to get a metric for new route confiugrations.
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def create_distance_matrix(points):
    size = len(points)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = euclidean_distance(points[i], points[j])
    return matrix

def solve_tsp(points):
    # Identify the most southwest point
    start_index = min(range(len(points)), key=lambda i: (points[i][1], points[i][0]))

    distance_matrix = create_distance_matrix(points)
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, start_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)  # scale to int

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        total_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(points[node_index])
            next_index = solution.Value(routing.NextVar(index))
            total_distance += distance_matrix[node_index][manager.IndexToNode(next_index)]
            index = next_index
        return total_distance, route
    else:
        return None, []
###TSP END

def calculate_centroid(coords):
    """Function that calculates the center of a small subset of points"""
    sum_x = sum(coord[0] for coord in coords)
    sum_y = sum(coord[1] for coord in coords)
    centroid_x = sum_x / len(coords)
    centroid_y = sum_y / len(coords)

    return (centroid_x, centroid_y)


def merge_partitions(partitions, max_stops_per_route, max_clusters=200):
    merged_partitions = partitions.copy()
    
    def find_closest_pair():
        """Looks for the other value with a closest center without exceeding the max_stops_per_route arg"""
        closest_pair = None
        min_distance = float('inf')
        
        # Check all pairs of partitions
        partition_ids = list(merged_partitions.keys())
        for i in range(len(partition_ids)):
            for j in range(i + 1, len(partition_ids)):
                id1, id2 = partition_ids[i], partition_ids[j]
                center1 = merged_partitions[id1]['partition_center']
                center2 = merged_partitions[id2]['partition_center']
                dist = euclidean_distance(center1, center2)
                
                # Calculate the new count if we were to merge these partitions
                new_count = merged_partitions[id1]['count'] + merged_partitions[id2]['count']
                
                # If they are closer and merging them won't exceed max_count
                if dist < min_distance and new_count <= max_stops_per_route:
                    min_distance = dist
                    closest_pair = (id1, id2)
        
        return closest_pair
    
    # Keep merging until we reach max_clusters or cannot merge further
    while len(merged_partitions) > max_clusters:
        print(len(merged_partitions), ' ', end='\r')
        # Find the closest pair of partitions
        closest_pair = find_closest_pair()
        
        if not closest_pair:
            break  # No more valid merges possible
        
        id1, id2 = closest_pair
        partition1 = merged_partitions[id1]
        partition2 = merged_partitions[id2]
        
        # Merge the partitions
        new_count = partition1['count'] + partition2['count']
        merged_points = list(partition1['partition_points']) + list(partition2['partition_points'])
        new_center = calculate_centroid(merged_points)

        # Create a new merged partition
        new_partition = {
            'partitionId': min(id1, id2),
            'count': new_count,
            'partition_points': merged_points,
            'partition_center': new_center
        }
        
        # Remove the old partitions and add the new one
        del merged_partitions[id1]
        del merged_partitions[id2]
        merged_partitions[new_partition['partitionId']] = new_partition
    
    return merged_partitions


def cluster_points(points, n_clusters):
    """
    Clusters 2D points into exactly n_clusters, good for non-spherical, bunched-up groups.

    Args:
        points (array-like): List or array of 2D points [(x, y), ...].
        n_clusters (int): Number of clusters to find.

    Returns:
        np.ndarray: Array of cluster labels for each point.
    """
    points = np.array(points)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='single',  # good for chaining tightly packed points
    )

    labels = model.fit_predict(points)
    return labels


def cluster_kmeans(points, n_clusters):
    points_np = np.array(points)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(points_np)
    return labels

def partition(coords, max_points=200):
    """
    Recursively partitions a list of (lat, lon) coordinates into clusters
    of at most `max_points` using axis-aligned splitting.

    Args:
        coords (list or np.ndarray): List/array of (lat, lon) points.
        max_points (int): Maximum number of points per cluster.

    Returns:
        List of np.ndarrays, each of shape (<=max_points, 2)
    """
    coords = np.array(coords)
    
    def split_recursive(points, depth=0):
        if len(points) <= max_points:
            return [points]

        # Alternate splitting axis: 0 for lat, 1 for lon
        axis = depth % 2
        sorted_points = points[points[:, axis].argsort()]
        median_index = len(sorted_points) // 2

        left = sorted_points[:median_index]
        right = sorted_points[median_index:]

        return split_recursive(left, depth + 1) + split_recursive(right, depth + 1)

    return split_recursive(coords)




def plot_centra_vs_full(centradata, fulldata, partitionsize, size = 2):
    partition_centra = []
    centra_representational_power = []
    for k, v in centradata.items(): 
        partition_centra.append(v['partition_center'])
        centra_representational_power.append(v['count'])
    x_coords = [coord[0] for coord in partition_centra]
    y_coords = [coord[1] for coord in partition_centra]
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(x_coords, y_coords, c=centra_representational_power, cmap='brg', s=size)
    plt.colorbar(scatter1, label='Representation count of center')
    plt.title(f'Partitioned plot ({len(partition_centra)} | {partitionsize})')
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(fulldata['lat'], fulldata['long'], c='blue', s=2)
    plt.tight_layout()







def read_request(route_id, ymd, idx_file):
    """
    Gepaste docstring in het goede formaat.
    Als je dit in het goede formaat doet kan je meteen documentatie maken
    https://realpython.com/python-project-documentation-with-mkdocs/

    Functie van Tim
    """
    
    folder_requests = os.path.join('..', 'data', 'input', 'requests', f"{route_id}-{ymd}")
    if not os.path.exists(folder_requests):
        return None

    file_request = os.listdir(folder_requests)[idx_file]
    file_path_request = os.path.join(folder_requests, file_request)

    
    with open(file_path_request, 'r') as f:
        request = json.load(f)

    rows = list()
    for task in request['tasks']:
        # TODO: Dit kan efficienter met een specifieke methode (pd.explode, json to dataframe zaken)
        if task['id'] == 'E1':
            continue ## TODO: Also retain this task!
        row = {'id' : int(task['id']),
                    'lat' : task['address']['latitude'],
                    'long' : task['address']['longitude'],
                    'start_time' : task['timeWindow']['from'],
                    'end_time' : task['timeWindow']['till']}
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by = 'id')


def visualize_request(route_id, ymd, idx_file):
    """
    This is actually a graph, could be plotted with NetworkX

    Functie van Tim
    """

    fig, ax = plt.subplots()
    df_request = read_request(route_id, ymd, idx_file)
    df_request = sort_request(df_request, route_id, ymd, idx_file)
    df_request.plot(x = 'lat', y = 'long', ax = ax, marker = '>', label = 'Route')

    df_request.iloc[[0], :].plot(x = 'lat', y = 'long', ax = ax, marker = 'o', color = 'r', ls = '', label = 'start')
    return fig, ax

def sort_request(df_request, route_id, ymd, idx_file):
    """
    Docstring!

    Functie van tim
    """
    folder_response = os.path.join('..', 'data', 'input', 'responses', f"{route_id}-{ymd}")
    file_response = os.listdir(folder_response)[idx_file]
    file_path_response = os.path.join(folder_response, file_response)

    with open(file_path_response, 'r') as f:
        content = f.read()
        if 'E1' in content: ## TODO: Deal with this better
            return None
        response = list(map(int, content.split('\n')))
    response.append(response[0])

    return (df_request.set_index('id')
                        .loc[response, :]
                        .reset_index())



######################    OUR UTILITIES:    ###########################

def routedatestring_to_date(string):
    """converts the datestring yyyymmdd into a date object."""
    return datetime.strptime(string, "%Y%m%d").date()

def get_route_dataframe(route_id, ymd, idx_file): 
    """
        Simple one linter that uses read_reques and sort_request to
        get the route dataframe from the JSON file, can be used as
        input for other functions. 
    """
    df_request = read_request(route_id, ymd, idx_file)
    df_request = sort_request(df_request, route_id, ymd, idx_file)
    return df_request

def get_route_center(route_id, ymd, idx_file):
    """
        Returns the center of the scatterplot by using mean for each
        as a tuple with (latitude, longitude)

        Parameters:
        - route_id: Id of the route
        - ymd: date in ymd formate
        - idx_file:


    """
    data = get_route_dataframe(route_id, ymd, idx_file)
    lat_center = data["lat"].mean()
    long_center = data["long"].mean()
    return (lat_center, long_center)

def find_clusters(df, eps=0.005, min_samples=3, latcol='lat', longcol='long'):
    """
    Identifies clusters in the lat-long data using DBSCAN and visualizes them with a heatmap.
    Ideally you run this on all JSON files.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'lat' and 'long' columns.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - df with cluster labels
    - cluster centers
    """
    coords = df[[latcol, longcol]].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coords)
    
    df['cluster'] = clustering.labels_  # Assign cluster labels
    
    # Compute cluster centers
    cluster_centers = df.groupby('cluster')[[latcol, longcol]].mean().reset_index()
    cluster_centers = cluster_centers[cluster_centers['cluster'] != -1]  # Ignore noise points (-1)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=longcol, y=latcol, hue='cluster', palette='tab10', s=50, alpha=0.6)
    plt.scatter(cluster_centers[longcol], cluster_centers[latcol], c='red', marker='X', s=100, label='Cluster Centers')
    plt.legend()
    plt.title("Identified Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    return df, cluster_centers




def highlight_route(df, id, h, w):
    """Highlight a specific route in green, others in steelblue."""
    df = df.copy()
    df['color'] = 'not in route'  # Set default value
    df.loc[df['route_id'] == id, 'color'] = 'in route'


    fig = px.scatter(
        df,
        x='lat',
        y='long',
        color='color',
        title='Spread in set for route id '+id,
        labels={'lat': 'Latitude', 'long': 'Longitude'},
        color_discrete_map={'in route': '#64ff33', 'not in route': '#0068ff'}, 
        height = h, 
        width = w
    )

    fig.show()


def train_test_val_splitter(df, trainratio, testratio, valratio, stratcols = False): 
    """takes a pandas df and splits it in three dfs according to the 
    given ratios. Optionally it uses tratified splitting columns defined in stratcols.
    arguments: 
    df (pandas dataframe)
    trainratio; (int): how many percent of df should end up in trainset
    testratio: (int): how many percent of df should end up in testset
    valratio: (int): how many percent of df should end up in valratio
    stratcols: (list): list of column names to perform a stratified split on. 
    """
    assert(trainratio+testratio+valratio == 100)
    if stratcols:
        train_data, remaining_data = train_test_split(
            df, test_size=(100 - trainratio) / 100, stratify=df[stratcols], random_state=42
        )
        test_data, val_data = train_test_split(
            remaining_data, test_size=valratio / (testratio + valratio), stratify=remaining_data[stratcols], random_state=42
        )
    else:
        train_data, remaining_data = train_test_split(
            df, test_size=(100 - trainratio) / 100, random_state=42
        )
        test_data, val_data = train_test_split(
            remaining_data, test_size=valratio / (testratio + valratio), random_state=42
        )
    return train_data, test_data, val_data


def get_X_y(df, target, drop = []): 
    drop.append(target)
    X = df.drop(columns=drop)
    y = df[target]
    return [X, y]