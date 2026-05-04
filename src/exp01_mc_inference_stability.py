"""
exp01_mc_inference_stability.py — EXP-01: Monte Carlo Inference Stability.
Analyzes the convergence of the Bayesian model by evaluating MAE, MAPE, 
and simulation error across K = [5, 10, 20, 30, 50, 100, 200] samples.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from utils.logger_config import setup_logger
from config.constants import TARGET, GLOBAL_SEED, SEGMENTS
from utils.models import build_bayesian_model

logger = setup_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'madrid', 'listings_production_test.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

MC_BENCHMARK_SIZE = 200

def get_segment_name(price):
    for seg in SEGMENTS:
        if seg['min'] <= price < seg['max']:
            return seg['name']
    return SEGMENTS[-1]['name']

def load_essentials(df_dummy):
    with open(os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)

    input_dim = preprocessor.transform(df_dummy).shape[1]
    models_bay = {}
    for seg in SEGMENTS:
        name = seg['name']
        weights_path = os.path.join(ARTIFACTS_DIR, 'bayesian', f'segment_{name}_weights.h5')
        if os.path.exists(weights_path):
            model = build_bayesian_model(input_dim, 1000)
            model.load_weights(weights_path)
            models_bay[name] = model
    return preprocessor, models_bay

def exp_01_mc_inference_stability(df_test, preprocessor, models_bay):
    logger.info("Starting EXP-01: Monte Carlo Inference Stability")

    K_values = [5, 10, 20, 30, 50, 100, 200]
    n = min(MC_BENCHMARK_SIZE, len(df_test))

    df_sample = df_test.sample(n=n, random_state=GLOBAL_SEED).copy()
    df_sample['segment'] = df_sample[TARGET].apply(get_segment_name)
    x_sample = preprocessor.transform(df_sample).astype('float32')
    y_true = df_sample[TARGET].values
    segments = df_sample['segment'].values

    results = []
    all_means_per_k = {}

    for k in K_values:
        logger.info(f"Evaluating inference stability for K = {k}")
        y_preds_all, y_means_k, valid_y = [], [], []

        for idx in range(n):
            seg_name = segments[idx]
            if seg_name not in models_bay: continue
            model = models_bay[seg_name]
            x_single = x_sample[idx:idx+1]
            
            preds = [float(model(x_single, training=False).mean().numpy().flatten()[0]) for _ in range(k)]
            y_preds_all.append(preds)
            y_means_k.append(np.mean(preds))
            valid_y.append(y_true[idx])

        y_means_k = np.array(y_means_k)
        valid_y = np.array(valid_y)
        y_preds_all = np.array(y_preds_all)
        all_means_per_k[k] = y_means_k

        mae = float(np.mean(np.abs(y_means_k - valid_y)))
        rmse = float(np.sqrt(np.mean((y_means_k - valid_y) ** 2)))
        mape = float(np.mean(np.abs((y_means_k - valid_y) / (valid_y + 1e-8))) * 100)
        
        s_values = np.std(y_preds_all, axis=1, ddof=1) if k > 1 else np.zeros_like(y_means_k)
        s_avg = float(np.mean(s_values))
        mcse_avg = float(np.mean(s_values / np.sqrt(k)))
        
        y_bar = float(np.mean(y_means_k))
        cv_mc = (mcse_avg / y_bar) * 100 if y_bar > 0 else 0.0

        results.append({
            'K': k, 
            'MAE': mae, 
            'RMSE': rmse, 
            'MAPE': mape,
            'MCSE': mcse_avg, 
            'CV MC': cv_mc,
            'Avg std': s_avg
        })

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'exp01_mc_inference_stability.csv')
    df_results.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ks = df_results['K']

    axes[0].plot(ks, df_results['MAE'], marker='o', color='#FF6B6B', linewidth=2)
    axes[0].set_title('MAE Convergence', fontweight='bold')
    axes[0].set_ylabel('MAE (€)')
    axes[0].set_xlabel('K (MC Samples)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, df_results['MCSE'], marker='s', color='#4ECDC4', linewidth=2, label='Empirical')
    theo_y = df_results['MCSE'].iloc[0] * (np.sqrt(ks.iloc[0]) / np.sqrt(ks))
    axes[1].plot(ks, theo_y, '--', color='gray', alpha=0.6, label='Theoretical $1/\sqrt{K}$')
    axes[1].set_title('Standard Error (MCSE)', fontweight='bold')
    axes[1].set_ylabel('Simulation Error (€)')
    axes[1].set_xlabel('K (MC Samples)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ks, df_results['CV MC'], marker='^', color='#9B59B6', linewidth=2)
    axes[2].set_title('Relative Error (CV MC)', fontweight='bold')
    axes[2].set_ylabel('Error %')
    axes[2].set_xlabel('K (MC Samples)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp01_mc_inference_stability.png'), dpi=300)
    plt.close()

    ncols = 4
    nrows = -(-len(K_values) // ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes2 = axes2.flatten()

    for i, k in enumerate(K_values):
        ax = axes2[i]
        ax.hist(all_means_per_k[k], bins=25, density=True, color='#6200EA', alpha=0.7, edgecolor='white')
        ax.set_title(f'K = {k}', fontweight='bold')
        ax.set_xlabel('Mean Price (€)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.2)

    for j in range(len(K_values), len(axes2)):
        axes2[j].set_visible(False)

    fig2.suptitle('Predictive Density Stabilization across K', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp01_mc_inference_stability_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {csv_path}")

if __name__ == '__main__':
    from utils.data_utils import load_listings
    df = load_listings(DATA_PATH)
    preprocessor, models_bay = load_essentials(df.head(1))
    exp_01_mc_inference_stability(df, preprocessor, models_bay)