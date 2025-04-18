import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.ops import triangulate
from shapely.geometry import MultiPoint
import math

import random

import os

#met dank aan chatgpt

class RoutePolygonManager:
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize with an optional DataFrame containing 'lat', 'long', and 'route_id'.
        """
        self.df = data
        self.polygons = {}  # route_id -> Polygon
        self.gdf = None     # GeoDataFrame of polygons
        self.unionized_poly = {}  #Polygon survaces with common areas
        self.reduced_poly = {}  #Polygon surfaces after shrinkage
        self.reduced_poly_centroid = {}  # after Polygon reduction set the center to the point that is closest to all other points envelopped by the cluster.

    def create_polygons(self, latcol, longcol, cluster_id):
        """
        Creates one polygon per route_id using convex hull of its points.
        Falls back to MultiPoint.convex_hull if Qhull fails.
        """
        if self.df is None:
            raise ValueError("DataFrame is not set.")

        polygons = {}
        for route_id, group in self.df.groupby(cluster_id):
            points = [Point(xy) for xy in zip(group[longcol], group[latcol])]
            coords = [(p.x, p.y) for p in points]

            try:
                if len(points) >= 3:
                    hull = ConvexHull(coords)
                    polygon = Polygon([coords[i] for i in hull.vertices])
                else:
                    polygon = MultiPoint(points).convex_hull
            except QhullError:
                polygon = MultiPoint(points).convex_hull  # Fallback for collinear or degenerate cases

            polygons[route_id] = polygon

        self.polygons = polygons
        self.gdf = gpd.GeoDataFrame({
            cluster_id: list(polygons.keys()),
            'geometry': list(polygons.values())
        }, geometry='geometry')

    def optimize_polygon_boundaries(self, buffer_amount=0.001):
        """
        Adjusts each polygon's boundary by buffering outward.
        This version avoids merging overlapping polygons.
        """
        if self.gdf is None:
            raise ValueError("Call create_polygons() first.")

        # Apply buffer individually per polygon
        self.gdf['geometry'] = self.gdf['geometry'].apply(lambda geom: geom.buffer(buffer_amount))


    def find_route_id(self, lat, long):
        """
        Given a (lat, long) point, returns the route_id it falls inside.
        """
        if self.gdf is None:
            raise ValueError("Polygons not yet created or loaded.")

        point = Point(long, lat)
        for idx, row in self.gdf.iterrows():
            if row['geometry'].contains(point):
                return row['route_id']
        return None

    def save_polygons_to_dir(self, directory):
        """
        Saves all polygons to a GeoJSON file in the given directory.
        """
        if self.gdf is None:
            raise ValueError("Polygons not yet created.")

        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "route_polygons.geojson")
        self.gdf.to_file(file_path, driver='GeoJSON')
        print(f"Saved polygons to: {file_path}")

    def load_polygons_from_dir(self, directory):
        """
        Loads polygons from a GeoJSON file in the given directory.
        """
        file_path = os.path.join(directory, "route_polygons.geojson")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")

        self.gdf = gpd.read_file(file_path)
        print(f"Loaded polygons from: {file_path}")

    def plot_polygons(self, figsize=(8, 8), show=True):
        """
        Plots just the polygons using matplotlib.
        """
        if self.gdf is None:
            raise ValueError("No polygons to plot. Run create_polygons() first.")

        fig, ax = plt.subplots(figsize=figsize)
        self.gdf.boundary.plot(ax=ax, color='blue')
        self.gdf.plot(ax=ax, alpha=0.3, edgecolor='black')
        ax.set_title("Route Polygons Only")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if show:
            plt.show()

    def plot_polygons_with_points(self, figsize=(8, 8), show=True):
        """
        Plots polygons and the original point coordinates.
        """
        if self.gdf is None or self.df is None:
            raise ValueError("Polygons or original data missing.")

        fig, ax = plt.subplots(figsize=figsize)
        self.gdf.boundary.plot(ax=ax, color='blue', linewidth=1)
        self.gdf.plot(ax=ax, alpha=0.3, edgecolor='black')

        # Plot the original coordinates
        ax.scatter(self.df['long'], self.df['lat'], c='red', s=2, label='Original Points')
        ax.set_title("Route Polygons with Original Points")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        if show:
            plt.show()


    def plot_random_polygons_grid(self, plot_dim=4, figsize=(12, 12), keyname = 'route_id', key_contains = False, plot_points=False):
        if self.gdf is None:
            raise ValueError("Polygons not yet created or loaded.")

        total_plots = plot_dim * plot_dim
        if key_contains != False:
            available_polygons = self.gdf[self.gdf[keyname].str.contains(key_contains)]
        else: 
            available_polygons = self.gdf.copy()

        # if len(available_polygons) < total_plots:
        #     raise ValueError(f"Not enough polygons ({len(available_polygons)}) to fill {total_plots} subplots.")
        if total_plots > len(available_polygons): 
            total_plots = len((available_polygons))
            print('less polygons available than requsted')
        sampled = available_polygons.sample(n=total_plots)

        fig, axs = plt.subplots(plot_dim, plot_dim, figsize=figsize)
        if key_contains: 
            fig.suptitle(f"{total_plots} Polygons for route where key contains {key_contains}", fontsize=16)
        else:
            fig.suptitle(f"{total_plots} Random Route Polygons", fontsize=16)
        axs = axs.flatten()

        for ax, (_, row) in zip(axs, sampled.iterrows()):
            geom = row.geometry
            route_id = row.get(keyname) or row.get('cluster_id', 'Route')

            # Plot polygon and boundary
            gpd.GeoSeries([geom]).boundary.plot(ax=ax, color='blue', linewidth=1)
            gpd.GeoSeries([geom]).plot(ax=ax, alpha=0.3, edgecolor='black')
            if plot_points: 
                points_df = self.df.query(f'{keyname}=="{route_id}"')
                # display(points_df)
                ax.scatter(points_df['long'], points_df['lat'], c='red', s=2)


            # Set title and axis labels
            ax.set_title(str(route_id))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()



    def overlay_polygons(self, overlap_frequency, key, route):
        """
        Overlays the polygons specified by the given list of IDs and the overlap frequency.

        Args:
        - ids (list): List of route_ids to overlay from the GeoDataFrame.
        - overlap_frequency (float): Percentage (between 0 and 1) of the surface that should overlap.

        Returns:
        - Polygon: A single polygon representing the overlapped area.
        """
        if self.gdf is None:
            raise ValueError("Polygons not yet created or loaded.")

        # Ensure overlap_frequency is between 0 and 1
        if not (0 <= overlap_frequency <= 1):
            raise ValueError("overlap_frequency must be between 0 and 1.")

        ids = [routeday for routeday in self.gdf[key] if routeday.startswith(route)]
        # Get the polygons by ID
        polygons_to_overlay = self.gdf[self.gdf[key].isin(ids)]['geometry']
        
        # Handle empty polygons case
        if polygons_to_overlay.empty:
            raise ValueError("No polygons found for the provided IDs.")

        # Union all selected polygons
        unioned_polygon = unary_union(polygons_to_overlay)

        # If overlap_frequency is not 1, we reduce the area of the resulting unioned polygon.
        # First, we'll buffer the resulting unioned polygon slightly to simulate a surface reduction.
        # Positive buffer reduces the area, negative increases the area.

        if overlap_frequency < 1:
            buffer_distance = (1 - overlap_frequency) * unioned_polygon.area
            unioned_polygon = unioned_polygon.buffer(-buffer_distance)  # Shrink the union polygon

        # Return the final polygon
        self.unionized_poly[route] = unioned_polygon




    
    def reduce_polygon(self, polygon, points, polyname, alpha=0.1, tolerance=0.001, eps=0.0075, min_samples=50):
        """
        Reduce the area of a polygon while optionally excluding some covered points.
        
        Args:
            polygon (Polygon or MultiPolygon): Original polygon.
            points (list of (x, y)): List of points.
            alpha (float): Alpha parameter for alpha shape (concavity).
            tolerance (float): Simplification tolerance.
            eps (float): DBSCAN epsilon for clustering.
            min_samples (int): DBSCAN minimum points per cluster.
        
        Returns:
            Polygon: Reduced polygon.
        """
        def alpha_shape(points, alpha):
            if len(points) < 4:
                return MultiPoint(points).convex_hull
            
            coords = [Point(p) for p in points]
            triangles = triangulate(MultiPoint(coords))
            
            def triangle_area(t):
                return t.area
            
            filtered = [t for t in triangles if t.length > 0 and t.area < (1.0 / alpha)]
            return unary_union(filtered).convex_hull
        inside_points = [p for p in points if polygon.contains(Point(p))]      
        if len(inside_points) == 0:
            return polygon

        coords = np.array(inside_points)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        clustered = [tuple(coords[i]) for i in range(len(coords)) if db.labels_[i] != -1]
        
        if len(clustered) < 3:
            clustered = inside_points  # fallback
        reduced = alpha_shape(clustered, alpha)

        simplified = reduced.simplify(tolerance, preserve_topology=True)

        self.reduced_poly[polyname] = simplified

        # Get polygon centroid
        covered_points = [p for p in inside_points if simplified.contains(Point(p))]
        if covered_points:
            distances = []
            if len(covered_points) > 1000:
                random.shuffle(covered_points)
                covered_points = covered_points[0:1000]
            for i, p1 in enumerate(covered_points):
                avg_dist = np.mean([np.linalg.norm(np.array(p1) - np.array(p2)) for j, p2 in enumerate(covered_points) if i != j])
                distances.append((avg_dist, p1))
            central_point = min(distances, key=lambda x: x[0])[1]
            self.reduced_poly_centroid[polyname] = central_point
        else:
            self.reduced_poly_centroid[polyname] = None

        return simplified



    def plot_optimized_polygon(self, raw_polygon: BaseGeometry, reduced_polygon: BaseGeometry, points, route_id):
        """
        Plots the original and reduced polygons along with the input points.
        
        Args:
            raw_polygon (Polygon or MultiPolygon): The original large polygon.
            reduced_polygon (Polygon or MultiPolygon): The reduced/simplified polygon.
            points (list of (x, y)): List of point coordinates.
        """
        fig, ax = plt.subplots(figsize=(18, 12))

        def plot_shape(shape, color, label, alpha=1.0, linestyle='-'):
            if isinstance(shape, MultiPolygon):
                for poly in shape.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=color, label=label, alpha=alpha, linestyle=linestyle)
                    label = None  # Only label the first in legend
            elif isinstance(shape, Polygon):
                x, y = shape.exterior.xy
                ax.plot(x, y, color=color, label=label, alpha=alpha, linestyle=linestyle)

        # Plot raw/original polygon
        plot_shape(raw_polygon, color='red', label='Original Polygon', linestyle='--')

        # Plot reduced polygon
        plot_shape(reduced_polygon, color='green', label='Reduced Polygon')

        # Plot points
        px, py = zip(*points)
        ax.scatter(px, py, color='blue', s=2, label='Points', alpha=0.7)

        # Plot the centroid: 
        centroid = self.reduced_poly_centroid[route_id]
        ax.scatter(centroid[0], centroid[1], color='red', s=80, marker="*", label='Centroid', alpha=0.7, zorder=5)

        ax.set_title("Polygon Optimization Visualization")
        ax.set_aspect('equal')
        ax.legend()
        plt.show()




    def get_polygons_containing_coordinate(self, coordinate, polygons):
        """
        Returns a list of polygons that contain the given coordinate.

        Args:
            reduced_poly (dict): A dictionary where keys are route IDs and values are Shapely Polygon objects.
            coordinate (tuple): A tuple representing the coordinate (lat, long).

        Returns:
            list: A list of route IDs whose polygons contain the given coordinate.
        """
        def find_closest_center(point, centers_dict):
            closest_id = None
            min_distance = float('inf')

            for center_id, center in centers_dict.items():
                # Calculate Euclidean distance
                distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_id = center_id

            return closest_id, min_distance



        containing_polygons = []
        #               lat             long
        point = Point(coordinate[1], coordinate[0])

        for route_id, polygon in polygons.items():
            if polygon.contains(point):  # Check if the polygon contains the point
                containing_polygons.append(route_id)

        if len(containing_polygons) == 1:
            return containing_polygons[0]
        elif len(containing_polygons) == 0: 
            #get any of the closest centroids
            # return 'b'  #TODO
            return find_closest_center(coordinate, self.reduced_poly_centroid)[0]
        
        else:
            #get the closest centroid of subsection based on polygons
            subselected_poly = {k:v for k,v in self.reduced_poly_centroid.items() if k in containing_polygons}
            # return 'a'  #TODO
            return find_closest_center(coordinate, subselected_poly)[0]

