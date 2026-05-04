"""
exp09_cost_analysis.py — EXP-09: Computational Cost Analysis.
Evaluates training resources (CPU/RAM) via external benchmark scripts, and production inference latency and throughput.
"""

import os
import json
import subprocess
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

tf.get_logger().setLevel('ERROR')

from utils.logger_config import setup_logger
from config.constants import TARGET, GLOBAL_SEED, SEGMENTS
from utils.data_utils import load_listings
from utils.model_io import load_segmented_models, patch_input_layer_compat

logger = setup_logger(__name__)
patch_input_layer_compat()

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, 'data', 'madrid', 'listings_production_test.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results', 'experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

MC_SAMPLES   = 200

def get_segment_name(price):
    for seg in SEGMENTS:
        if seg['min'] <= price < seg['max']:
            return seg['name']
    return SEGMENTS[-1]['name']

def load_test_data():
    logger.info(f"Loading test data: {DATA_PATH}")
    return load_listings(DATA_PATH)

def load_models(df_test):
    return load_segmented_models(ARTIFACTS_DIR, df_test)

def run_external_benchmark(script_name):
    script_path = os.path.join(BASE_DIR, 'src', script_name)
    logger.info(f"Running benchmark: {script_path}")
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR, check=False)
    if result.returncode != 0:
        logger.warning(f"{script_name} exited with code {result.returncode}")
    return result.returncode == 0

def exp_09_cost_analysis(df_test, preprocessor, models_det, models_bay, mc_samples=MC_SAMPLES):
    logger.info("EXP-09: Computational Cost Analysis start")

    det_csv = os.path.join(RESULTS_DIR, 'exp09_train_det.csv')
    bay_csv = os.path.join(RESULTS_DIR, 'exp09_train_bay.csv')

    if not os.path.exists(det_csv):
        run_external_benchmark('train_det_benchmark.py')
    if not os.path.exists(bay_csv):
        run_external_benchmark('train_bay_benchmark.py')

    if not os.path.exists(det_csv) or not os.path.exists(bay_csv):
        logger.error(f"Benchmark CSVs missing in {RESULTS_DIR}. Aborting.")
        return {}

    df_det_res = pd.read_csv(det_csv)
    df_bay_res = pd.read_csv(bay_csv)

    total_det_time = df_det_res.groupby('Segment')['Elapsed_s'].max().sum()
    total_bay_time = df_bay_res.groupby('Segment')['Elapsed_s'].max().sum()
    
    cpu_det_mean   = df_det_res['CPU_pct'].mean()
    cpu_bay_mean   = df_bay_res['CPU_pct'].mean()
    cpu_det_peak   = df_det_res['CPU_pct'].max()
    cpu_bay_peak   = df_bay_res['CPU_pct'].max()
    
    ram_det_peak   = df_det_res['RAM_MB'].max()
    ram_bay_peak   = df_bay_res['RAM_MB'].max()

    logger.info("Training benchmarks registered. Starting inference evaluation...")

    x_test  = preprocessor.transform(df_test.copy()).astype('float32')
    df_test = df_test.copy()
    df_test['seg'] = df_test[TARGET].apply(get_segment_name)
    segs    = df_test['seg'].values

    n_sample = min(500, len(x_test))
    idx_s    = np.random.choice(len(x_test), n_sample, replace=False)
    x_s, seg_s = x_test[idx_s], segs[idx_s]

    latencies_det, latencies_bay = [], []
    for _ in range(100):
        i  = np.random.randint(0, n_sample)
        sn = seg_s[i]; xi = x_s[i:i+1]
        if sn in models_det:
            t0 = time.time()
            _ = models_det[sn](xi, training=False)
            latencies_det.append((time.time() - t0) * 1000)
        if sn in models_bay:
            t0 = time.time()
            for _ in range(mc_samples):
                _ = models_bay[sn](xi, training=False).mean().numpy()
            latencies_bay.append((time.time() - t0) * 1000)

    lat_det = np.mean(latencies_det or [0])
    lat_bay = np.mean(latencies_bay or [0])

    unique_segs = [s['name'] for s in SEGMENTS]
    t0 = time.time()
    for sn in unique_segs:
        if sn in models_det:
            mask = (seg_s == sn)
            if np.any(mask):
                _ = models_det[sn](x_s[mask], training=False).numpy()
    throughput_det = n_sample / max(time.time() - t0, 1e-6)

    t0 = time.time()
    for sn in unique_segs:
        if sn in models_bay:
            mask = (seg_s == sn)
            if np.any(mask):
                x_rep = np.tile(x_s[mask], (mc_samples, 1))
                _ = models_bay[sn](x_rep, training=False).mean().numpy()
    throughput_bay = n_sample / max(time.time() - t0, 1e-6)

    params_det  = sum(m.count_params() for m in models_det.values())
    params_bay  = sum(m.count_params() for m in models_bay.values())
    size_det_mb = params_det * 4 / (1024 * 1024)
    size_bay_mb = params_bay * 4 / (1024 * 1024)

    res_dict = {
        'Training_Time_Det_s':  [total_det_time],
        'Training_Time_Bay_s':  [total_bay_time],
        'Training_Time_Ratio':  [total_bay_time / max(total_det_time, 1e-6)],
        'CPU_Det_Mean_pct':     [cpu_det_mean],
        'CPU_Bay_Mean_pct':     [cpu_bay_mean],
        'CPU_Det_Peak_pct':     [cpu_det_peak],
        'CPU_Bay_Peak_pct':     [cpu_bay_peak],
        'RAM_Det_Peak_MB':      [ram_det_peak],
        'RAM_Bay_Peak_MB':      [ram_bay_peak],
        'Latency_Det_ms':       [lat_det],
        'Latency_Bay_ms':       [lat_bay],
        'Latency_Ratio':        [lat_bay / max(lat_det, 1e-6)],
        'Throughput_Det_sps':   [throughput_det],
        'Throughput_Bay_sps':   [throughput_bay],
        'Throughput_Ratio':     [throughput_det / max(throughput_bay, 1e-6)],
        'Params_Det':           [params_det],
        'Params_Bay':           [params_bay],
        'Params_Ratio':         [params_bay / max(params_det, 1)],
        'Size_Det_MB':          [size_det_mb],
        'Size_Bay_MB':          [size_bay_mb],
        'Size_Ratio':           [size_bay_mb / max(size_det_mb, 1e-6)],
    }

    csv_out = os.path.join(RESULTS_DIR, 'exp09_cost_analysis_detailed.csv')
    pd.DataFrame(res_dict).to_csv(csv_out, index=False)
    logger.info(f"Results saved: {csv_out}")

    plot_resource_usage(df_det_res, df_bay_res)

    return {'Throughput_Ratio': throughput_det / max(throughput_bay, 1e-6)}

def plot_resource_usage(df_det, df_bay):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_det = df_det.copy()
    df_bay = df_bay.copy()

    if 'Timestamp' in df_det.columns:
        df_det = df_det.sort_values('Timestamp')
        df_det['Elapsed_s'] = df_det['Timestamp'] - df_det['Timestamp'].min()
    elif 'timestamp' in df_det.columns:
        df_det = df_det.sort_values('timestamp')
        df_det['Elapsed_s'] = df_det['timestamp'] - df_det['timestamp'].min()
        
    if 'Timestamp' in df_bay.columns:
        df_bay = df_bay.sort_values('Timestamp')
        df_bay['Elapsed_s'] = df_bay['Timestamp'] - df_bay['Timestamp'].min()
    elif 'timestamp' in df_bay.columns:
        df_bay = df_bay.sort_values('timestamp')
        df_bay['Elapsed_s'] = df_bay['timestamp'] - df_bay['timestamp'].min()

    C_DET = '#FF6B6B'
    C_BAY = '#6200EA'

    ax = axes[0]
    ax.plot(df_det['Elapsed_s'], df_det['CPU_pct'], label='Deterministic', color=C_DET, alpha=0.85)
    ax.plot(df_bay['Elapsed_s'], df_bay['CPU_pct'], label='Bayesian', color=C_BAY, alpha=0.85)
    ax.set_title('CPU Consumption over Time', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CPU Usage (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df_det['Elapsed_s'], df_det['RAM_MB'], label='Deterministic', color=C_DET, alpha=0.85)
    ax.plot(df_bay['Elapsed_s'], df_bay['RAM_MB'], label='Bayesian', color=C_BAY, alpha=0.85)
    ax.set_title('Memory Consumption over Time', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RAM Usage (MB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp09_resource_usage.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Resource usage plot saved: {path}")

def main():
    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    logger.info("EXP-09 start")

    df_test = load_test_data()
    preprocessor, models_det, models_bay = load_models(df_test)

    exp_09_cost_analysis(df_test, preprocessor, models_det, models_bay)

    logger.info(f"EXP-09 COMPLETED — results in {RESULTS_DIR}")

if __name__ == '__main__':
    main()