## put all functions to make features in here please
from tqdm import tqdm
import pandas as pd
import numpy as np

def calculate_distances(  
        df, 
        step_prev_dst = 'dist_since_prev', 
        cumsum = 'dist_cumulative', 
        step_next_dst = 'dist_to_next_stop'        
        ):
    """
        Takes a pandas dataframe and returns distance metrics per routepoint
        based on euclidian distance. Dataframe gets sorted internally based on 
        file UUID and ascending stop_number
        NOTE: when updating order-logic in notebook 1, this needs to be reran!!
        //BUG: 28/3/2025: Order of stops is unclear see request id: 41931cd2-8975-4a64-9197-d16abe871bb7
        
        ARGUMENTS:
            df = pandas df = (dataframe with lat and long column as well as file_uuid and stop_number.
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
        route_df = route_df.sort_values('stop_number', ascending = True)
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
        TODO: this method should return two lists:
        1) list of points that are init_df and are missing from final_df 
        2) lists of points that are in final_df but were not in the init_df
        this is rather straightforward with using sets and comparing them
    """
    pass