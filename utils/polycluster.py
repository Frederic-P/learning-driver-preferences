import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPoint, Polygon
#from shapely.ops import unary_union
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import random

import os


class RoutePolygonManager:
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize with an optional DataFrame containing 'lat', 'long', and 'route_id'.
        """
        self.df = data
        self.polygons = {}  # route_id -> Polygon
        self.gdf = None     # GeoDataFrame of polygons

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


    def plot_random_polygons_grid(self, plot_dim=4, figsize=(12, 12)):
        if self.gdf is None:
            raise ValueError("Polygons not yet created or loaded.")

        total_plots = plot_dim * plot_dim
        available_polygons = self.gdf.copy()

        if len(available_polygons) < total_plots:
            raise ValueError(f"Not enough polygons ({len(available_polygons)}) to fill {total_plots} subplots.")

        sampled = available_polygons.sample(n=total_plots)

        fig, axs = plt.subplots(plot_dim, plot_dim, figsize=figsize)
        fig.suptitle(f"{total_plots} Random Route Polygons", fontsize=16)
        axs = axs.flatten()

        for ax, (_, row) in zip(axs, sampled.iterrows()):
            geom = row.geometry
            route_id = row.get('route_id') or row.get('cluster_id', 'Route')

            # Plot polygon and boundary
            gpd.GeoSeries([geom]).boundary.plot(ax=ax, color='blue', linewidth=1)
            gpd.GeoSeries([geom]).plot(ax=ax, alpha=0.3, edgecolor='black')

            # Set title and axis labels
            ax.set_title(str(route_id))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
