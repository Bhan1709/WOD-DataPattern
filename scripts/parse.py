import os
import pandas as pd
import logging
import duckdb
from tqdm import tqdm
from collections import Counter
from datetime import datetime, timedelta

__dir__ = os.path.dirname(os.path.abspath(__file__)) + "/.."

data_dir = __dir__ + "/data"
log_dir = __dir__ + "/logs"
stat_dir = __dir__ + "/stats"

logging.basicConfig(
    filename= log_dir + '/parse.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define schemas
CAST_COLUMNS = ['latitude', 'longitude', 'year', 'month', 'day', 'time', 'bottom depth', 'cast']
FEATURE_COLUMNS = ['chlorophyl', 'depth', 'nitrate', 'oxygen', 'ph', 'salinity', 'temperatur', 'transmissi', 'cast']

db = duckdb.connect(data_dir +"/parsed/wod.duckdb")

# Create tables with predefined schemas
db.execute(f"""
CREATE TABLE IF NOT EXISTS casts (
    {', '.join([f'"{col}" TEXT' if col != 'cast' else '"cast" INT PRIMARY KEY' for col in CAST_COLUMNS])},
    timestamp TIMESTAMP
);
""")

db.execute(f"""
CREATE TABLE IF NOT EXISTS features (
    {', '.join([f'"{col}" DOUBLE' if col != 'cast' else '"cast" INT' for col in FEATURE_COLUMNS])},
    FOREIGN KEY("cast") REFERENCES casts("cast")
);
""")

def parse_cast_blocks(csv_file):
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    cast_blocks = []
    current_block = []
    in_block = False
    for line in lines:
        if line.strip().startswith("#--------------------------------------------------------------------------------"):
            if current_block:
                cast_blocks.append(current_block)
                current_block = []
            in_block = True
        elif in_block:
            current_block.append(line.strip('\n'))
    if current_block:
        cast_blocks.append(current_block)

    return cast_blocks

def extract_metadata_and_variables(block):
    metadata_keys = set()
    variable_names = set()
    parsing_vars = False

    for i, line in enumerate(block):
        parts = [p.strip() for p in line.split(',') if p.strip() != '']

        if not parts:
            continue

        if parts[0].lower() == 'variables':
            parsing_vars = True
            variable_names.update([p.strip().lower() for p in parts[1:] if p.strip().lower() not in ['f', 'o']])
            continue

        if parsing_vars:
            if parts[0].lower().startswith('end of variables'):
                break
        else:
            if len(parts) >= 3:
                key = parts[0].lower().strip()
                if key:
                    metadata_keys.add(key)

    return metadata_keys, variable_names

def scan_metadata_and_variables(folder_path):
    all_metadata_keys = set()
    all_variable_names = set()
    metadata_counter = Counter()
    variable_counter = Counter()

    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    logging.info(f"Scanning folder {folder_path} for metadata and variable names")

    for file_path in tqdm(csv_files, desc="Scanning CSV files"):
        try:
            cast_blocks = parse_cast_blocks(file_path)
            for block in cast_blocks:
                try:
                    metadata_keys, variable_names = extract_metadata_and_variables(block)
                    all_metadata_keys.update(metadata_keys)
                    all_variable_names.update(variable_names)
                    metadata_counter.update(metadata_keys)
                    variable_counter.update(variable_names)
                except Exception as e:
                    logging.warning(f"Block skipped during scanning in file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

    logging.info(f"Finished scanning folder. Found {len(all_metadata_keys)} metadata keys and {len(all_variable_names)} variables.")
    logging.info(f"Metadata Keys: {sorted(all_metadata_keys)}")
    logging.info(f"Variable Names: {sorted(all_variable_names)}")

    logging.info("Metadata Key Frequencies:")
    for key, count in metadata_counter.items():
        logging.info(f"  {key}: {count}")

    logging.info("Variable Name Frequencies:")
    for var, count in variable_counter.items():
        logging.info(f"  {var}: {count}")

    return sorted(all_metadata_keys), sorted(all_variable_names), metadata_counter, variable_counter

#Get raw feature frequencies
# metadata_keys, variable_names = scan_metadata_and_variables(data_dir + "/raw")
# print(metadata_keys)
# print(variable_names)

def extract_cast_data(block):
    metadata = {col: None for col in CAST_COLUMNS}
    variable_data = []
    variable_columns = []
    parsing_vars = False

    for line in block:
        parts = [p.strip() for p in line.split(',')]

        if not parts or parts[0].lower().startswith("end of variables"):
            parsing_vars = False
            continue

        if parts[0].lower() == 'variables':
            parsing_vars = True
            variable_columns = [p.lower() for i, p in enumerate(parts[1:]) if i % 3 == 0]
            continue

        if parsing_vars:
            try:
                values = []
                for i in range(1, len(parts), 3):
                    val = parts[i].strip()
                    values.append(float(val) if val else None)

                if len(values) == len(variable_columns):
                    variable_data.append(dict(zip(variable_columns, values)))
                else:
                    logging.debug(f"Variable row length mismatch: {parts}")
            except Exception as e:
                logging.debug(f"Skipping line due to parse error: {e}, line: {parts}")
        else:
            if len(parts) >= 3:
                key = parts[0].lower().strip()
                value = parts[2].strip()
                if key in metadata:
                    metadata[key] = value

    cast_id = metadata.get('cast')
    if cast_id is None:
        raise ValueError("Missing cast ID")
    metadata['cast'] = int(float(cast_id))

    # Construct timestamp
    try:
        if metadata['year'] and metadata['month'] and metadata['day']:
            y = int(metadata['year'])
            m = int(metadata['month'])
            d = int(metadata['day'])
            if d == 0:
                d = 1
            t = float(metadata['time']) if metadata['time'] else 0.0

            # Convert decimal hours to timedelta
            extra_time = timedelta(hours=t)
            base_date = datetime(y, m, d)
            final_timestamp = base_date + extra_time
            metadata['timestamp'] = final_timestamp
        else:
            raise ValueError("Missing date field")
    except Exception as e:
        logging.warning(f"Failed to create timestamp for cast {metadata['cast']}: {e}")
        metadata['timestamp'] = None

    return metadata, variable_columns, variable_data

def insert_to_duckdb(metadata, variables):
    metadata_df = pd.DataFrame([metadata])
    for col in CAST_COLUMNS:
        if col not in metadata_df.columns:
            metadata_df[col] = None
    if 'timestamp' not in metadata_df.columns:
        metadata_df['timestamp'] = None
    metadata_df = metadata_df[CAST_COLUMNS + ['timestamp']]
    db.execute("INSERT OR REPLACE INTO casts SELECT * FROM metadata_df")

    if variables:
        features_df = pd.DataFrame(variables)
        features_df['cast'] = metadata['cast']
        for col in FEATURE_COLUMNS:
            if col not in features_df.columns:
                features_df[col] = None
        features_df = features_df[FEATURE_COLUMNS]
        db.execute("INSERT INTO features SELECT * FROM features_df")

def process_folder(folder_path):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    logging.info(f"Processing folder {folder_path} for predefined schema ingestion")

    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        try:
            cast_blocks = parse_cast_blocks(file_path)
            for block in tqdm(cast_blocks, desc=f"Extracting from {os.path.basename(file_path)}", leave=False):
                try:
                    metadata, _, variable_data = extract_cast_data(block)
                    insert_to_duckdb(metadata, variable_data)
                except Exception as e:
                    logging.warning(f"Failed to process cast in {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

process_folder(data_dir + "/raw")