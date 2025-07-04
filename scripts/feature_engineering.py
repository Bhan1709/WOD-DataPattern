import duckdb
import io
import geopandas as gpd
from shapely.geometry import Point
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
from matplotlib.table import Table
import seaborn as sns
import os
import traceback
import pandas as pd
import numpy as np
import textwrap

data_dir = __file__.replace("feature_engineering.py", "../data")
log_dir = __file__.replace("feature_engineering.py", "../logs")
stat_dir = __file__.replace("feature_engineering.py", "../stats")

logging.basicConfig(
    filename= log_dir + '/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

db = duckdb.connect(data_dir +"/parsed/wod.duckdb")

def summarize_table(table_name):
    logging.info(f"\nSummary for table: {table_name}")
    df = db.execute(f"SELECT * FROM {table_name}").df()
    buffer = io.StringIO()
    df.info(buf=buffer)
    logging.info("\n" + buffer.getvalue())
    desc = df.describe(include='all')
    logging.info("\n" +desc.to_string())
    print(buffer.getvalue())
    print(desc)
    return df

def add_identifier_column(table='features_imputed', id_col='row_id'):
    try:
        db.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {id_col};")
        db.execute(f"DROP TABLE IF EXISTS {table}_with_rowid;")
        db.execute(f"""
            CREATE OR REPLACE TABLE {table}_with_rowid AS
            SELECT 
                row_number() OVER () AS {id_col},
                *
            FROM {table};
        """)

        # Drop original and rename
        db.execute(f"DROP TABLE {table}")
        db.execute(f"ALTER TABLE {table}_with_rowid RENAME TO {table}")
        logging.info(f"Added identifier column to table {table}")
    except Exception as e:
        logging.error(f"Failed to add identifier column to {table}: {e}")

def tag_and_plot_ocean_basin(table='features_imputed', save_dir= stat_dir + '/distributions'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        # Load ocean shapefile
        basins = gpd.read_file(data_dir + '/../geopandamaps/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')
        basins = basins.to_crs("EPSG:4326")  # Ensure WGS84 for lat/lon compatibility
        world = gpd.read_file(data_dir + '/../geopandamaps/110m_cultural/ne_110m_admin_0_countries.shp')

        # Load lat/lon and row_id
        df = db.execute(f"SELECT row_id, latitude, longitude FROM {table}").fetch_df()
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

        # Spatial join with basin polygons
        joined = gpd.sjoin(gdf, basins[['geometry', 'NAME']], predicate="within", how="left")
        tagged_df = joined[['row_id', 'latitude', 'longitude', 'NAME', 'geometry']].rename(columns={'NAME': 'ocean_basin'})

        # Handle unassigned rows with nearest basin centroid
        untagged = tagged_df[tagged_df['ocean_basin'].isna()].to_crs(epsg=3857)
        if not untagged.empty:
            basin_centroids = basins.to_crs(epsg=3857)
            basin_centroids['centroid'] = basin_centroids.geometry.centroid
            names = basin_centroids['NAME'].values
            centroids = list(basin_centroids['centroid'].values)

            def nearest_basin(point):
                distances = [point.distance(c) for c in centroids]
                return names[int(pd.Series(distances).idxmin())]

            untagged['ocean_basin'] = untagged.geometry.apply(nearest_basin)
            tagged_df.loc[untagged.index, 'ocean_basin'] = untagged['ocean_basin']
            logging.info(f"Assigned {len(untagged)} untagged rows using nearest basin centroid.")

        # Write to duckdb
        db.execute(f"CREATE OR REPLACE TABLE basin_tagged AS SELECT * FROM {table}")
        db.register("temp_tag", tagged_df[['row_id', 'ocean_basin']])
        db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS ocean_basin TEXT")
        db.execute(f"""
                    UPDATE {table}
                    SET ocean_basin = t.ocean_basin
                    FROM temp_tag t
                    WHERE {table}.row_id = t.row_id
                """)
        logging.info(f"Saved ocean_basin sucecsfully to {table}")
        
        # Prepare for plotting
        tagged_points_df = gpd.GeoDataFrame(tagged_df, geometry=gpd.points_from_xy(tagged_df.longitude, tagged_df.latitude), crs="EPSG:4326")
        tagged_points_df = tagged_points_df[tagged_points_df.geometry.notnull() & tagged_points_df.is_valid]
        if tagged_points_df.empty:
            logging.warning("No valid tagged data points to plot.")
            return

        # Palette and color map
        basin_counts = tagged_points_df['ocean_basin'].value_counts().sort_values(ascending=False)
        basins_sorted = basin_counts.index.tolist()
        palette = sns.color_palette("hls", len(basins_sorted))
        basin_color_map = dict(zip(basins_sorted, palette))

        # Plot map without legend
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='whitesmoke', edgecolor='black')
        basins.boundary.plot(ax=ax, color='lightgray', linewidth=0.8)
        for basin in basins_sorted:
            color = basin_color_map[basin]
            subset = tagged_points_df[tagged_points_df["ocean_basin"] == basin]
            if not subset.empty:
                subset.plot(ax=ax, markersize=2, color=color, label=basin, alpha=0.6)
        ax.set_title("Measurement Locations by Ocean Basin", fontsize=16)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(save_dir + f"/lat_lon_distribution_w_basins_{table}.png", dpi=300)
        plt.close()
        logging.info(f"Saved lat-lon world map plot to {save_dir}/lat_lon_distribution_w_basins_{table}.png")

        # Plot legend table separately
        # Create table dataframe
        legend_df = pd.DataFrame({
            "Ocean Basin": basins_sorted,
            "ColorHex": [to_hex(basin_color_map[b]) for b in basins_sorted],
            "Count": [basin_counts[b] for b in basins_sorted]
        })

        # Wrap long basin names for better display (simulate text wrap)
        legend_df["Ocean Basin"] = legend_df["Ocean Basin"].apply(
            lambda x: "\n".join(textwrap.wrap(x, width=38))
        )

        # Log the table
        logging.info("Legend Table:\n" + legend_df.to_string(index=False))

        # Plot table with actual color swatches
        fig, ax = plt.subplots(figsize=(6, len(legend_df) * 0.4 + 1))
        ax.axis('off')
        table = Table(ax, bbox=[0, 0, 1, 1])

        cell_height = 1 / (len(legend_df) + 1)
        col_widths = [0.2, 0.5, 0.3]

        # Add header row
        headers = ["Color", "Ocean Basin", "Count"]
        for col, header in enumerate(headers):
            cell = table.add_cell(0, col, width=col_widths[col], height=cell_height,
                          text=header, loc='center', facecolor='lightgray', edgecolor='black')
            cell.get_text().set_fontsize(10)

        # Add rows
        for row_idx, (_, row) in enumerate(legend_df.iterrows(), start=1):
            # Color swatch (left column)
            table.add_cell(row_idx, 0, width=col_widths[0], height=cell_height,
                        text='', facecolor=row['ColorHex'], edgecolor='black')

            # Basin name
            # font_size = 9 if len(row["Ocean Basin"]) <= 34 else 5
            cell = table.add_cell(row_idx, 1, width=col_widths[1], height=cell_height,
                      text=row["Ocean Basin"], loc='center', edgecolor='black')
            cell.get_text().set_fontsize(9)

            # Count
            cell = table.add_cell(row_idx, 2, width=col_widths[2], height=cell_height,
                        text=str(row["Count"]), loc='center', edgecolor='black')
            cell.get_text().set_fontsize(9)

        ax.add_table(table)

        legend_table_path = os.path.join(save_dir, f"ocean_basins_legend_table.png")
        plt.savefig(legend_table_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved separate legend table with color swatches to {legend_table_path}")

    except Exception as e:
        logging.error(f"Failed to plot lat-lon map for {table}: {e}")
        logging.error("Full traceback:\n" + traceback.format_exc())

# add_identifier_column(table='features_imputed', id_col='row_id')
# summarize_table(table_name='features_imputed')
tag_and_plot_ocean_basin(table='features_imputed', save_dir=stat_dir + '/distributions')
summarize_table(table_name='features_imputed')