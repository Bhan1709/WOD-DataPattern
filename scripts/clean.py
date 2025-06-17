import duckdb
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

data_dir = __file__.replace("clean.py", "../data")
log_dir = __file__.replace("clean.py", "../logs")
stat_dir = __file__.replace("clean.py", "../stats")

logging.basicConfig(
    filename=log_dir + '/clean.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

db = duckdb.connect(data_dir +"/parsed/wod.duckdb")

# Create joined table with required columns and filtered rows
db.execute("""
CREATE TABLE IF NOT EXISTS features_clean AS
SELECT f.*, c.timestamp, c.latitude, c.longitude
FROM features f
JOIN casts c ON f.cast = c.cast
WHERE f.chlorophyll IS NOT NULL;
""")

def plot_null_heatmap(granularity='month'):
    granularity_format = {
        'month': ('%b %Y', 'month'),
        'year': ('%Y', 'year'),
        'day': ('%d %b %Y', 'day'),
        'week': ('%W %Y', 'week')
    }

    if granularity not in granularity_format:
        raise ValueError("Granularity must be one of: 'day', 'month', 'year', 'week'")

    date_format, trunc_unit = granularity_format[granularity]

    result = db.execute("PRAGMA table_info('features_clean')").fetchall()
    feature_columns = [row[1] for row in result if row[1] not in ('cast', 'timestamp', 'latitude', 'longitude')]

    heatmap_data = {}
    all_time_labels = set()

    for feature in tqdm(feature_columns, desc="Aggregating nulls (features_clean)"):
        query = f'''
            SELECT date_trunc('{trunc_unit}', timestamp) AS time_group,
                   COUNT(*) AS total,
                   SUM(CASE WHEN "{feature}" IS NULL THEN 1 ELSE 0 END) AS null_count
            FROM features_clean
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
    ax.set_title(f'Null Values Over Time (features_clean - {granularity.capitalize()})', fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")

    xticklabels = heatmap_df.columns
    if len(xticklabels) > 20:
        step = max(1, len(xticklabels) // 20)
        selected_labels = xticklabels[::step]
        ax.set_xticks([i for i in range(0, len(xticklabels), step)])
        ax.set_xticklabels(selected_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(stat_dir + f"/null_heatmap_features_clean_{granularity}.png", dpi=300)
    plt.close()
    logging.info(f"Saved null heatmap to null_heatmap_features_clean_{granularity}.png")

plot_null_heatmap('day')
plot_null_heatmap('week')
plot_null_heatmap('month')
plot_null_heatmap('year')
