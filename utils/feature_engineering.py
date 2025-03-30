## put all functions to make features in here please
from tqdm import tqdm
import pandas as pd
import numpy as np

def calculate_distances(  
        df, 
        ordercol,
        step_prev_dst = 'dist_since_prev', 
        cumsum = 'dist_cumulative', 
        step_next_dst = 'dist_to_next_stop'        
        ):
    """
        Takes a pandas dataframe and returns distance metrics per routepoint
        based on euclidian distance. Dataframe gets sorted internally based on 
        file UUID and ascending stop_order
        NOTE: when updating order-logic in notebook 1, this needs to be reran!!
        
        ARGUMENTS:
            df = pandas df = (dataframe with lat and long column as well as file_uuid and stop_order.
            ordercol = column name where the order sequence information is stored as increment inteers.
            step_prev_dst = str (opt)= name of the column where interstop distances are stored.
                this column for point c should be interpreted as 'distance driven from b to get to c'
            cumsum = str (opt) = name of column where total route distance at point is stored.
                this column for point c shoul be interpreted as 'distance criven since start to get to c'
            step_next_dst = str (opt) =  name of the column where the interstop distance to the next stop is stored.
                tihs column for point c should be interpreted as 'how long is it from c to point d'
        
        RETURNS:
            df with more values in form step_pref_dst, cumsum, step_next_dst.
    """
    calculated_distances = []
    for route, route_df in tqdm(df.groupby('file_uuid')):
        route_df = route_df.sort_values(ordercol, ascending = True)
        d = np.sqrt((route_df['lat'].diff())**2 + (route_df['long'].diff())**2)
        route_df[step_prev_dst] = d
        route_df[step_prev_dst] = route_df[step_prev_dst].fillna(0)
        route_df[cumsum] = route_df[step_prev_dst].cumsum()
        route_df[step_next_dst] = route_df[cumsum].shift(-1) - route_df[cumsum].fillna(0)
        calculated_distances.append(route_df)  
    calculated_df = pd.concat(calculated_distances)
    return calculated_df


def stop_diffs(init_df, final_df):
    """
    ARGUMENTS:
        init_df = df representing the first version of the route. 
        final_df = df representing the final version of the route. 

    RETURNS:
        Method returns three lists:
        1) list of points that are in common
        2) lists of points that are in final_df but were not in the init_df
        3) list of points that are in init_df and are missing from final_df 
    """
    start_points = init_df.id
    stop_points = final_df.id

    common = list(set(start_points) & set(stop_points))
    added = list(set(stop_points) - set(start_points))
    dropped = list(set(start_points) - set(stop_points))

    return [common, added, dropped]


def get_route_center(df):
    """
    takes a dataframe and returns the center of each route_id
    a route_id is shared across multiple days - but we've seen
    in EDA that routes are sticking to specific zones of the 
    larger geographic area. 
    #TODO
    ==> WARNING: sensitive to outliers. consider making another algo
    """
    df = df.copy()
    res = []
    for _, df_route in df.groupby('route_id'):
        mean_lat = df_route['lat'].mean()
        mean_long = df_route['long'].mean()
        df_route['mean_lat'] = mean_lat
        df_route['mean_long'] = mean_long
        df_route['dst_point_to_center'] = np.sqrt((df_route['lat'] - mean_lat)**2 + (df_route['long'] - mean_long)**2)
        res.append(df_route)
    return pd.concat(res)
