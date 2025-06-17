import io
import duckdb
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

data_dir = __file__.replace("preprocessing.py", "../data")
log_dir = __file__.replace("preprocessing.py", "../logs")
stat_dir = __file__.replace("preprocessing.py", "../stats")

logging.basicConfig(
    filename=log_dir + '/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

db = duckdb.connect(data_dir +"/parsed/wod.duckdb")

def summarize_table(table_name):
    logging.info(f"\nSummary for table: {table_name}")
    df = db.execute(f"SELECT * FROM {table_name}").df()
    buffer = io.StringIO()
    df.info(buf=buffer)
    logging.info(buffer.getvalue())
    desc = df.describe(include='all')
    logging.info(desc.to_string())
    print(buffer.getvalue())
    print(desc)
    return df

def null_summary(table_name):
    df = db.execute(f"SELECT * FROM {table_name}").df()
    total = len(df)
    null_counts = df.isnull().sum()
    null_percent = (null_counts / total * 100).round(2)
    summary = pd.DataFrame({'Null Count': null_counts, 'Percent': null_percent})
    summary = summary[summary['Null Count'] > 0].sort_values(by='Null Count', ascending=False)
    logging.info(f"\nNull summary for table {table_name}:")
    logging.info("\n" + summary.to_string())
    print(summary)
    return summary

def rename_feature_columns():
    rename_map_features = {
        "chlorophyl": "chlorophyll",
        "temperatur": "temperature",
        "transmissi": "transmissivity"
        }
    for old_name, new_name in rename_map_features.items():
        db.execute(f'ALTER TABLE features RENAME COLUMN "{old_name}" TO "{new_name}";')

# Plot variable trends against timestamp
def plot_variable_trends():
    result = db.execute("PRAGMA table_info('features')").fetchall()
    feature_columns = [row[1] for row in result if row[1] != 'cast']

    df = db.execute("""
        SELECT f.*, c.timestamp
        FROM features f
        JOIN casts c ON f.cast = c.cast
        WHERE c.timestamp IS NOT NULL
    """).df()

    for var in feature_columns:
        if df[var].notnull().sum() == 0:
            logging.info(f"Skipping {var}: all values are null")
            continue

        plt.figure(figsize=(12, 4))
        sns.lineplot(data=df, x='timestamp', y=var, marker="o", linewidth=0.5)
        plt.title(f"{var.capitalize()} Over Time", fontsize=12)
        plt.xlabel("Timestamp", fontsize=10)
        plt.ylabel(var.capitalize(), fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(stat_dir +f"/preclean_trend_{var}.png", dpi=300)
        plt.close()
        logging.info(f"Saved trend plot for {var} as preclean_trend_{var}.png")

def plot_null_heatmap_over_time(table='features',granularity='month', save_file='null_heatmap'):
    granularity_format = {
        'month': ('%b %Y', 'month'),
        'year': ('%Y', 'year'),
        'day': ('%d %b %Y', 'day'),
        'week': ('%W %Y', 'week')
    }

    if granularity not in granularity_format:
        raise ValueError("Granularity must be one of: 'day', 'month', 'year', 'week'")

    date_format, trunc_unit = granularity_format[granularity]

    result = db.execute(f"PRAGMA table_info('{table}')").fetchall()
    feature_columns = [row[1] for row in result if row[1] not in ('cast', 'timestamp', 'latitude', 'longitude')]

    heatmap_data = {}
    all_time_labels = set()

    for feature in tqdm(feature_columns, desc=f"Aggregating nulls ({table})"):
        query = f'''
            SELECT date_trunc('{trunc_unit}', timestamp) AS time_group,
                   COUNT(*) AS total,
                   SUM(CASE WHEN "{feature}" IS NULL THEN 1 ELSE 0 END) AS null_count
            FROM {table}
            WHERE timestamp IS NOT NULL
            GROUP BY time_group
            ORDER BY time_group
        '''
        df = db.execute(query).df()
        if df.empty:
            continue
        df['label'] = df['time_group'].dt.strftime(date_format)
        df.set_index('label', inplace=True)
        all_time_labels.update(df.index)
        heatmap_data[feature] = (df['null_count'] / df['total'] * 100).round(2)

    sorted_labels = sorted(all_time_labels, key=lambda x: datetime.strptime(x, date_format))
    heatmap_df = pd.DataFrame(heatmap_data).T.reindex(columns=sorted_labels)

    plt.figure(figsize=(14, max(6, len(heatmap_df) * 0.5)))
    ax = sns.heatmap(heatmap_df, cmap='magma_r', cbar_kws={'label': '% Null'}, linewidths=0.4)
    ax.set_title(f'Null Values Over Time ({table} - {granularity.capitalize()})', fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")

    xticklabels = heatmap_df.columns
    if len(xticklabels) > 20:
        step = max(1, len(xticklabels) // 20)
        selected_labels = xticklabels[::step]
        ax.set_xticks([i for i in range(0, len(xticklabels), step)])
        ax.set_xticklabels(selected_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(stat_dir + f"/{save_file}_{table}_{granularity}.png", dpi=300)
    plt.close()
    logging.info(f"Saved null heatmap to {save_file}_{table}_{granularity}.png")

def plot_null_percent_lineplot(granularity='month'):
    granularity_sql = {
        'day': 'day',
        'week': 'week',
        'month': 'month',
        'year': 'year'
    }

    if granularity not in granularity_sql:
        raise ValueError("Granularity must be one of: 'day', 'week', 'month', 'year'")

    trunc_unit = granularity_sql[granularity]
    result = db.execute("PRAGMA table_info('features')").fetchall()
    feature_columns = [row[1] for row in result if row[1] != 'cast']

    all_stats = []

    for feature in tqdm(feature_columns, desc=f"Lineplot aggregation ({granularity})"):
        query = f'''
            SELECT date_trunc('{trunc_unit}', c.timestamp) AS time_group,
                   COUNT(*) AS total,
                   SUM(CASE WHEN f."{feature}" IS NULL THEN 1 ELSE 0 END) AS null_count
            FROM features f
            JOIN casts c ON f.cast = c.cast
            WHERE c.timestamp IS NOT NULL
            GROUP BY time_group
            ORDER BY time_group
        '''
        df = db.execute(query).df()
        if df.empty:
            continue
        df['null_percent'] = (df['null_count'] / df['total']) * 100
        df['variable'] = feature
        all_stats.append(df[['time_group', 'variable', 'null_percent']])

    if not all_stats:
        logging.warning("No data to plot for null percentages.")
        return

    final_df = pd.concat(all_stats)
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=final_df, x='time_group', y='null_percent', hue='variable', linewidth=1)
    plt.title(f"Null Percentage per Variable ({granularity.capitalize()})")
    plt.ylabel("% Null")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(stat_dir + f"/null_lineplot_{granularity}.png", dpi=300)
    plt.close()
    logging.info(f"Saved null percentage line plot as null_lineplot_{granularity}.png")

def drop_invalid_lat_lon(table='features_clean'):
    try:
        db.execute(f'''
            CREATE OR REPLACE TABLE {table} AS
            SELECT *
            FROM {table}
            WHERE CAST(latitude AS DOUBLE) BETWEEN -90 AND 90
              AND CAST(longitude AS DOUBLE) BETWEEN -180 AND 180;
        ''')
        logging.info(f"Dropped rows with invalid lat/lon from table {table}")
    except Exception as e:
        logging.error(f"Failed to drop invalid lat/lon from {table}: {e}")

def plot_feature_distributions(table='features_clean'):
    try:
        columns = db.execute(f"PRAGMA table_info({table})").df()
        feature_cols = [col for col in columns['name'] if col not in ('cast', 'timestamp', 'latitude', 'longitude')]

        for col in feature_cols:
            df = db.execute(f'''
                SELECT CAST({col} AS DOUBLE) AS value
                FROM {table}
                WHERE {col} IS NOT NULL
            ''').df()

            if df.empty:
                continue

            # histogram
            plt.figure(figsize=(10, 4))
            sns.histplot(df['value'], kde=True, bins=100, color='skyblue')
            plt.title(f"Distribution of {col.capitalize()}")
            plt.xlabel(col.capitalize())
            plt.ylabel("Frequency")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(stat_dir +  f"/distribution_{table}_{col}.png", dpi=300)
            plt.close()
            logging.info(f"Saved distribution plot for {col} to distribution_{table}_{col}.png")

            # log scaled histogram
            plt.figure(figsize=(10, 4))
            sns.histplot(df['value'], kde=True, bins=100, color='skyblue')
            plt.title(f"Distribution of {col.capitalize()}")
            plt.xlabel(col.capitalize())
            plt.ylabel("Frequency")
            plt.yscale('log')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(stat_dir +  f"/_log_scaled_distribution_{table}_{col}.png", dpi=300)
            plt.close()

            # Quantify outlier prevalence
            q1 = df['value'].quantile(0.25)
            q3 = df['value'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
            outlier_count = len(outliers)
            total_count = len(df)
            outlier_pct = (outlier_count / total_count) * 100
            logging.info(f"Outliers in {col}: {outlier_count} out of {total_count} ({outlier_pct:.2f}%)")

            # Box plot
            plt.figure(figsize=(6, 6))
            sns.boxplot(y=df['value'], color='lightcoral')
            plt.title(f"Distribution Box Plot of {col.capitalize()}")
            plt.ylabel(col.capitalize())
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(stat_dir + f"/boxplot_{table}_{col}.png", dpi=300)
            plt.close()
            logging.info(f"Saved box plot for {col} to boxplot_{table}_{col}.png")

            # Bar plot
            plt.figure(figsize=(12, 4))
            bin_counts, bin_edges = np.histogram(df['value'], bins=50)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.bar(bin_centers, bin_counts, width=np.diff(bin_edges), align='center', color='mediumseagreen', alpha=0.8)
            plt.title(f"Distribution Bar Plot of {col.capitalize()}")
            plt.xlabel(col.capitalize())
            plt.ylabel("Count")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(stat_dir +  f"/barplot_{table}_{col}.png", dpi=300)
            plt.close()
            logging.info(f"Saved bar plot for {col} to barplot_{table}_{col}.png")

    except Exception as e:
        logging.error(f"Error plotting feature distributions from {table}: {e}")

def plot_lat_lon_map(table='features_imputed'):
    try:
        df = db.execute(f'''
            SELECT 
                CAST(latitude AS DOUBLE) AS latitude,
                CAST(longitude AS DOUBLE) AS longitude
            FROM {table}
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                  AND CAST(latitude AS DOUBLE) BETWEEN -90 AND 90
                  AND CAST(longitude AS DOUBLE) BETWEEN -180 AND 180;
        ''').df()

        world = gpd.read_file(data_dir + '/../geopandamaps/110m_cultural/ne_110m_admin_0_countries.shp')
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)], crs="EPSG:4326")

        fig, ax = plt.subplots(figsize=(14, 7))
        world.plot(ax=ax, color='whitesmoke', edgecolor='black')
        gdf.plot(ax=ax, markersize=2, color='skyblue', alpha=0.5)
        ax.set_title('Global Distribution of Measurement Locations', fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(stat_dir + f"/lat_lon_distribution_{table}.png", dpi=300)
        plt.close()
        logging.info(f"Saved lat-lon world map plot to lat_lon_distribution_{table}.png")

    except Exception as e:
        logging.error(f"Failed to plot lat-lon map for {table}: {e}")
# Hexbin map of latitude and longitude
def plot_hexbin_lat_lon(table='features_clean'):
    try:
        df = db.execute(f'''
            SELECT CAST(latitude AS DOUBLE) AS latitude,
                   CAST(longitude AS DOUBLE) AS longitude
            FROM {table}
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND CAST(latitude AS DOUBLE) BETWEEN -90 AND 90
              AND CAST(longitude AS DOUBLE) BETWEEN -180 AND 180;
        ''').df()

        if df.empty:
            logging.warning(f"No valid coordinates in {table} for hexbin plot.")
            return

        world = gpd.read_file(data_dir + '/../geopandamaps/110m_cultural/ne_110m_admin_0_countries.shp')
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")

        fig, ax = plt.subplots(figsize=(14, 7))
        world.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=0.5)

        hb = ax.hexbin(df['longitude'], df['latitude'], gridsize=100, cmap='viridis', bins='log')
        cb = fig.colorbar(hb, ax=ax, label='log10(N)')

        ax.set_title('Hexbin Map of Measurement Locations')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        plt.tight_layout()
        plt.savefig(stat_dir + f"/lat_lon_distribution_{table}.png", dpi=300)
        plt.close()
        logging.info(f"Saved GeoPandas hexbin lat-lon plot for {table} to lat_lon_distribution_{table}.png")
    except Exception as e:
        logging.error(f"Failed to plot GeoPandas hexbin lat-lon map for {table}: {e}")

def winsorize_table(table='features_clean'):
    try:
        columns = db.execute(f"PRAGMA table_info({table})").df()
        feature_cols = [col for col in columns['name'] if col not in ('cast', 'timestamp', 'latitude', 'longitude')]

        for col in feature_cols:
            df = db.execute(f"SELECT CAST({col} AS DOUBLE) AS val FROM {table} WHERE {col} IS NOT NULL").df()
            if df.empty:
                continue
            lower = df['val'].quantile(0.01)
            upper = df['val'].quantile(0.99)

            before = db.execute(f"SELECT COUNT(*) FROM {table} WHERE CAST({col} AS DOUBLE) < {lower} OR CAST({col} AS DOUBLE) > {upper}").fetchone()[0]

            db.execute(f'''
                UPDATE {table}
                SET {col} =
                    CASE
                        WHEN CAST({col} AS DOUBLE) < {lower} THEN {lower}
                        WHEN CAST({col} AS DOUBLE) > {upper} THEN {upper}
                        ELSE {col}
                    END
                WHERE {col} IS NOT NULL;
            ''')

            logging.info(f"Winsorized {col} to 1-99 percentile range: [{lower:.4f}, {upper:.4f}] — Modified {before} values")

        logging.info(f"Winsorization completed for table: {table}")

    except Exception as e:
        logging.error(f"Failed to winsorize features in table {table}: {e}")

# Spatial-temporal imputation for all variables

def spatial_temporal_imputation(table='features_clean', vars_to_impute=None, output_table='features_imputed'):
    if vars_to_impute is None:
        vars_to_impute = ['oxygen', 'salinity', 'temperature']

    try:
        db.execute("DROP TABLE IF EXISTS temp_impute;")
        db.execute(f"""
        CREATE TABLE temp_impute AS
        SELECT *,
            date_trunc('month', timestamp) AS month_ts,
            date_trunc('day', timestamp) AS day_ts,
            round(CAST(latitude AS DOUBLE), 1) AS lat_bin,
            round(CAST(longitude AS DOUBLE), 1) AS lon_bin
        FROM {table};
        """)

        strategy_counts = {}

        for col in vars_to_impute:
            strategy_counts[col] = {}
            initial_nulls = db.execute(f"SELECT COUNT(*) FROM temp_impute WHERE {col} IS NULL").fetchone()[0]

            strategies = [
                ("day + lat/lon", f"day_ts = t.day_ts AND lat_bin = t.lat_bin AND lon_bin = t.lon_bin"),
                ("month + lat/lon", f"month_ts = t.month_ts AND lat_bin = t.lat_bin AND lon_bin = t.lon_bin"),
                ("lat/lon", f"lat_bin = t.lat_bin AND lon_bin = t.lon_bin"),
                ("month", f"month_ts = t.month_ts"),
                ("global", f"TRUE")
            ]

            for name, condition in strategies:
                before = db.execute(f"SELECT COUNT(*) FROM temp_impute WHERE {col} IS NULL").fetchone()[0]
                db.execute(f"""
                UPDATE temp_impute t
                SET {col} = (
                    SELECT avg({col}) FROM temp_impute
                    WHERE {condition} AND {col} IS NOT NULL
                )
                WHERE {col} IS NULL;
                """)
                after = db.execute(f"SELECT COUNT(*) FROM temp_impute WHERE {col} IS NULL").fetchone()[0]
                strategy_counts[col][name] = before - after

            total_filled = sum(strategy_counts[col].values())
            logging.info(f"Finished hierarchical imputation for {col}: {strategy_counts[col]} (Filled {total_filled} out of {initial_nulls})")

        db.execute(f"DROP TABLE IF EXISTS {output_table};")
        result = db.execute(f"PRAGMA table_info('{table}')").fetchall()
        original_columns = [row[1] for row in result if not any(suffix in row[1] for suffix in ['_ts', '_bin','_1'])]
        db.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT {', '.join([f'"{col}"' for col in original_columns])}
        FROM temp_impute;
        """)
        db.execute("DROP TABLE temp_impute;")

    except Exception as e:
        logging.error(f"Failed spatial-temporal imputation: {e}")

def analyze_timestamp_distribution(table='features_imputed', granularity='month'):
    granularity_unit = {
        'day': 'day',
        'month': 'month',
        'year': 'year',
        'week': 'week'
    }.get(granularity, 'month')

    try:
        summary = db.execute(f'''
            SELECT 
                MIN(timestamp) AS start_date,
                MAX(timestamp) AS end_date,
                COUNT(*) AS total_records,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) AS active_days,
                COUNT(DISTINCT DATE_TRUNC('month', timestamp)) AS active_months,
                COUNT(DISTINCT DATE_TRUNC('year', timestamp)) AS active_years
            FROM {table}
            WHERE timestamp IS NOT NULL;
        ''').df()

        logging.info(f"Timestamp distribution for {table} (granularity={granularity}):\n{summary.to_string(index=False)}")
        print(summary.to_string(index=False))

        time_counts = db.execute(f'''
            SELECT DATE_TRUNC('{granularity_unit}', timestamp) AS period, COUNT(*) AS count
            FROM {table}
            WHERE timestamp IS NOT NULL
            GROUP BY period
            ORDER BY period;
        ''').df()

        plt.figure(figsize=(14, 5))
        plt.bar(time_counts['period'].dt.to_pydatetime(), time_counts['count'], width=20 if granularity_unit == 'year' else 10)
        plt.title(f"{granularity.capitalize()} Record Count for {table}")
        plt.xlabel(granularity.capitalize())
        plt.ylabel("Number of Records")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(stat_dir + f"/timestamp_distribution_{table}_{granularity}.png", dpi=300)
        plt.close()
        logging.info(f"Saved timestamp distribution plot to timestamp_distribution_{table}_{granularity}.png")

    except Exception as e:
        logging.error(f"Error analyzing timestamp distribution for {table}: {e}")

def analyze_depth_distribution(table='features_imputed'):
    try:
        depth_counts = db.execute(f'''
            SELECT ROUND(depth, 0) AS depth_bin, COUNT(*) AS count
            FROM {table}
            WHERE depth IS NOT NULL
            GROUP BY depth_bin
            ORDER BY depth_bin;
        ''').df()

        plt.figure(figsize=(14, 5))
        plt.bar(depth_counts['depth_bin'], depth_counts['count'], width=1.0)
        plt.title(f"Depth Distribution for {table}")
        plt.xlabel("Depth (m, binned)")
        plt.ylabel("Number of Records")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"depth_distribution_{table}.png", dpi=300)
        plt.close()
        logging.info(stat_dir + f"/Saved depth distribution plot to depth_distribution_{table}.png")
    except Exception as e:
        logging.error(f"Error analyzing depth distribution for {table}: {e}")


# print(db.execute("SHOW TABLES;").df())
# summarize_table("casts")
# summarize_table("features")
# null_summary("casts")
# null_summary("features")
# plot_variable_trends()
# rename_feature_columns()
# plot_null_heatmap_over_time('features','week')
# plot_null_percent_lineplot('day')

# Create joined table with required columns and filtered rows
# db.execute("""
# CREATE TABLE IF NOT EXISTS features_clean AS
# SELECT f.*, c.timestamp, c.latitude, c.longitude
# FROM features f
# JOIN casts c ON f.cast = c.cast
# WHERE f.chlorophyll IS NOT NULL;
# """)

# summarize_table("features_clean")
# null_summary("features_clean")

# Drop columns with excessive nulls from features_clean if they exist
# for col in ['transmissivity', 'ph', 'nitrate']:
#    try:
#        db.execute(f"ALTER TABLE features_clean DROP COLUMN IF EXISTS \"{col}\";")
#        logging.info(f"Dropped column '{col}' from features_clean")
#    except Exception as e:
#        logging.warning(f"Failed to drop column '{col}' from features_clean: {e}")

# drop_invalid_lat_lon()
# plot_feature_distributions()
# summarize_table("features_clean")
# plot_hexbin_lat_lon('features_clean')
# winsorize_table()
# null_summary("features_clean")
# plot_null_heatmap_over_time('features_clean','month','null_heatmap_cleaned')

# spatial_temporal_imputation()
# summarize_table("features_imputed")
# null_summary("features_imputed")
# plot_null_heatmap_over_time('features_imputed','month','null_heatmap_imputed')

# analyze_timestamp_distribution(table='features_imputed', granularity='month')
