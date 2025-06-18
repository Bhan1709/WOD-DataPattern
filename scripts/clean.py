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
