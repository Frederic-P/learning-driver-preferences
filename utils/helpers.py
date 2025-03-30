import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import seaborn as sns
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split



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