import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.utils import resample
import hdbscan
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import umap
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import logging
import warnings
from tqdm import tqdm
from memory_profiler import profile
from time import time
import sys
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
data_dir = __file__.replace("clustering.py", "../data")
log_dir = __file__.replace("clustering.py", "../logs")
stat_dir = __file__.replace("clustering.py", "../stats")

db = duckdb.connect(data_dir +"/parsed/wod.duckdb")
TABLE_NAME = "features_imputed"
FEATURES = ["chlorophyll", "oxygen", "temperature", "salinity", "depth"]
MODEL_DIR = __file__.replace("clustering.py", "../trained_models")
EXPORT_DIR = stat_dir + "/clustering_outputs"
LOGGED_EXPERIMENTS_PATH = os.path.join(EXPORT_DIR, "experiment_logs.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- LOGGING ---
logging.basicConfig(
    filename=os.path.join(log_dir, "clustering.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- LOAD DATA ---
@profile
def load_features():
    try:
        query = f"""
            SELECT row_id, latitude, longitude, {', '.join(FEATURES)}
            FROM {TABLE_NAME}
            WHERE {' AND '.join([f'{f} IS NOT NULL' for f in FEATURES])}
        """
        df = db.execute(query).fetch_df()
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}\n{traceback.format_exc()}")
        raise

def get_next_model_index(method: str, model_dir: str=MODEL_DIR) -> int:
    """
    Scan the model directory and return the next available index for a given method.
    Expects files in the format: {method}_{idx}_model.joblib
    """
    pattern = re.compile(rf"^{re.escape(method)}_(\d+)_model\.joblib$")
    existing = []

    if os.path.isdir(model_dir):
        for fname in os.listdir(model_dir):
            match = pattern.match(fname)
            if match:
                existing.append(int(match.group(1)))

    return max(existing, default=-1) + 1

# --- CLUSTERING EVALUATION ---
@profile
def manual_silhouette_score(X, labels):
    try:
        n = len(X)
        # if n > 10000:
        #    logging.warning("Skipping manual silhouette: too many points.")
        #    return None
        distances = pairwise_distances(X)
        silhouette_vals = []
        for i in tqdm(range(n), desc="Silhouette computation"):
            same_cluster = labels == labels[i]
            a = np.mean(distances[i, same_cluster]) if np.sum(same_cluster) > 1 else 0
            b = np.min([np.mean(distances[i, labels == lbl]) for lbl in set(labels) if lbl != labels[i]])
            silhouette_vals.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
        return np.mean(silhouette_vals)
    except Exception as e:
        logging.error(f"Error in manual silhouette: {e}\n{traceback.format_exc()}")
        return None
    
@profile
def evaluate_clustering(X, labels, max_sample_size=100_000):
    try:
        metrics = {}
        if len(set(labels) - {-1}) >= 1:
            metrics["silhouette"] = silhouette_score(X, labels, sample_size=max_sample_size, random_state=42)
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        else:
            metrics["silhouette"] = None
            metrics["calinski_harabasz"] = None
            metrics["davies_bouldin"] = None
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating clustering: {e}\n{traceback.format_exc()}")
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

# --- HYPERPARAMETER TUNING ---
@profile
def run_clustering_experiments(df, features):
    try:
        X = df[features].dropna()
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.astype(np.float32)

        results = []

        experiments = {
            "hdbscan": ParameterGrid({"min_cluster_size": [250, 500, 750, 1000, 2500]}),
            "optics": ParameterGrid({"min_samples": [10, 25, 50, 100], "xi": [0.1, 0.25, 0.5]}),
            "kmeans": ParameterGrid({"n_clusters": [3,5,10,15,20,25,30]}),
            "minibatch": ParameterGrid({"n_clusters": [3,5,10,15,20,25,30], "batch_size": [5000, 10000, 25000, 50000, 75000, 100_000]}),
            "gmm": ParameterGrid({"n_components": [3,5,10,15,20,25,30]}),
            "birch": ParameterGrid({"n_clusters": [3,5,10,15,20,25,30], "threshold": [1.0, 2.5, 5.0, 7.5, 10.0]}),
        }

        
        for method, param_grid in experiments.items():
            logging.info(f"Running {method.upper()} experiments...")
            for idx, params in enumerate(tqdm(param_grid, desc=f"{method.upper()} Experiments")):
                try:
                    if method == "kmeans":
                        model = KMeans(**params, random_state=42)
                        labels = model.fit_predict(X_scaled)
                        metrics = evaluate_clustering(X_scaled, labels)
                    elif method == "minibatch":
                        model = MiniBatchKMeans(**params, random_state=42)
                        labels = model.fit_predict(X_scaled)
                        metrics = evaluate_clustering(X_scaled, labels)
                    elif method == "gmm":
                        model = GaussianMixture(**params, random_state=42)
                        labels = model.fit_predict(X_scaled)
                        metrics = evaluate_clustering(X_scaled, labels)
                    elif method == "hdbscan":
                        subset = resample(X_scaled, n_samples=100_000, replace=False, random_state=42)
                        model = hdbscan.HDBSCAN(**params)
                        labels = model.fit_predict(subset)
                        metrics = evaluate_clustering(subset, labels)
                    elif method == "birch":
                        model = Birch(**params)
                        labels = model.fit_predict(X_scaled)
                        metrics = evaluate_clustering(X_scaled, labels)
                    elif method == "optics":
                        subset = resample(X_scaled, n_samples=250_000, replace=False, random_state=42)
                        model = OPTICS(**params)
                        labels = model.fit_predict(subset)
                        metrics = evaluate_clustering(subset, labels)
                    else:
                        continue
                    
                    id = get_next_model_index(method)
                    param_id = f"{method}_{id}"
                    joblib.dump(model, os.path.join(MODEL_DIR, f"{param_id}_model.joblib"))
                    np.save(os.path.join(MODEL_DIR, f"{param_id}_labels.npy"), labels)

                    log_msg = f"{method.upper()} | Params: {params} | Silhouette: {metrics['silhouette']}, CH: {metrics['calinski_harabasz']}, DB: {metrics['davies_bouldin']}"
                    logging.info(log_msg)

                    pd.DataFrame([{"method": method, **params, **metrics}]).to_csv(
                        LOGGED_EXPERIMENTS_PATH, mode='a', header=not os.path.exists(LOGGED_EXPERIMENTS_PATH), index=False
                    )

                    results.append({
                        "method": method,
                        "params": params,
                        "silhouette": metrics["silhouette"],
                        "calinski_harabasz": metrics["calinski_harabasz"],
                        "davies_bouldin": metrics["davies_bouldin"]
                    })
                except Exception as e:
                    logging.warning(f"Failed {method} with {params}: {e}\n{traceback.format_exc()}")

        return results, X_scaled, df.loc[X.index]
    except Exception as e:
        logging.error(f"Error running clustering experiments: {e}\n{traceback.format_exc()}")
        raise

# --- SAVE BEST MODELS ---
@profile
def save_best_models(results, X_scaled, df):
    try:
        df_logs = pd.read_csv(LOGGED_EXPERIMENTS_PATH)
        comparison_data = []
        for method in set(r["method"] for r in results):
            method_results = [r for r in results if r["method"] == method and r["silhouette"] is not None]
            if not method_results:
                continue
            method_logs = df_logs[df_logs["method"] == method]
            if method_logs.empty: return None
            best_row = method_logs.sort_values("silhouette", ascending=False).iloc[0]
            idx = method_logs.index.get_loc(best_row.name)
            param_id = f"{method}_{idx}"
            labels = np.load(os.path.join(MODEL_DIR, f"{param_id}_labels.npy"))

            df_with_labels = df.copy()
            df_with_labels[f"{method}_cluster"] = labels

            db.register("temp_df", df_with_labels[["row_id", f"{method}_cluster"]])
            db.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS {method}_cluster INTEGER")
            db.execute(f"""
                UPDATE {TABLE_NAME}
                SET {method}_cluster = temp_df.{method}_cluster
                FROM temp_df
                WHERE {TABLE_NAME}.row_id = temp_df.row_id
            """)
            db.unregister("temp_df")

            df_with_labels[["row_id", "latitude", "longitude"] + FEATURES + [f"{method}_cluster"]].to_csv(
                os.path.join(EXPORT_DIR, f"{method}_clustered_data.csv"), index=False
            )

            profile = df_with_labels.groupby(f"{method}_cluster")[FEATURES].median()
            profile.to_csv(os.path.join(EXPORT_DIR, f"{method}_cluster_profiles.csv"))

            summary = df_with_labels.groupby(f"{method}_cluster")[FEATURES].agg(["mean", "std", "count"])
            summary.to_csv(os.path.join(EXPORT_DIR, f"{method}_cluster_summary_stats.csv"))

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_with_labels, x="longitude", y="latitude", hue=f"{method}_cluster", palette="tab10", s=10)
            plt.title(f"{method.upper()} Cluster Distribution (Lat/Lon)")
            plt.tight_layout()
            plt.savefig(os.path.join(EXPORT_DIR, f"{method}_lat_lon_clusters.png"), dpi=300)
            plt.close()

            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=10)
            plt.title(f"{method.upper()} Clusters - PCA")
            plt.tight_layout()
            plt.savefig(os.path.join(EXPORT_DIR, f"{method}_pca.png"), dpi=300)
            plt.close()

            reducer = umap.UMAP(random_state=42)
            coords = reducer.fit_transform(X_scaled)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=10)
            plt.title(f"{method.upper()} Clusters - UMAP")
            plt.tight_layout()
            plt.savefig(os.path.join(EXPORT_DIR, f"{method}_umap.png"), dpi=300)
            plt.close()

            comparison_data.append(best_row.to_dict())

        pd.DataFrame(comparison_data).to_csv(os.path.join(EXPORT_DIR, "clustering_comparison_summary.csv"), index=False)

    except Exception as e:
        logging.error(f"Error saving model results: {e}\n{traceback.format_exc()}")
        raise

# --- MAIN ---
if __name__ == "__main__":
    try:
        df = load_features()
        results, X_scaled, used_df = run_clustering_experiments(df, FEATURES)
        if not results:
            logging.error("No valid clustering results found. Exiting.")
            sys.exit(1)
        save_best_models(results, X_scaled, used_df)
    except Exception as main_error:
        logging.critical(f"Fatal error in clustering pipeline: {main_error}\n{traceback.format_exc()}")
        raise