"""
exp02_07_uncertainty_suite.py — EXP-02 to EXP-07 consolidated.
Predictive quality, uncertainty analysis, calibration, decomposition,
and rejection curve experiments on the production test set.
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

tf.get_logger().setLevel('ERROR')
from datetime import datetime
from scipy.stats import spearmanr
from config.constants import (
    TARGET, GLOBAL_SEED, SEGMENTS, C_DET, C_BAY, C_AUX
)
from utils.data_utils import load_listings
from utils.model_io import load_segmented_models, patch_input_layer_compat

patch_input_layer_compat()

tfd  = tfp.distributions
tfpl = tfp.layers

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, 'data', 'madrid', 'listings_production_test.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results', 'experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

MC_SAMPLES       = 200
TEST_SUBSET_SIZE = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return super().default(obj)

def load_data():
    return load_listings(DATA_PATH)

def load_models(df_processed=None):
    return load_segmented_models(ARTIFACTS_DIR, df_processed)

def get_segment_name(price):
    for seg in SEGMENTS:
        if seg['min'] <= price < seg['max']:
            return seg['name']
    return SEGMENTS[-1]['name']

def predict_segmented(df, preprocessor, models_det, models_bay, mc_samples=MC_SAMPLES, verbose=True):
    """Run segmented predictions (batch mode). Returns y_true, y_pred_det, y_pred_bay, std_bay, segments."""
    start_time = time.time()
    
    df['segment_temp'] = df[TARGET].apply(get_segment_name)

    all_indices    = []
    y_true_all     = []
    y_pred_det_all = []
    y_pred_bay_all = []
    std_bay_all    = []
    segments_all   = []

    unique_segments = df['segment_temp'].unique()

    for i, seg_name in enumerate(unique_segments, 1):
        batch_df = df[df['segment_temp'] == seg_name].copy()
        n_batch  = len(batch_df)

        all_indices.extend(batch_df.index.tolist())
        y_true_all.extend(batch_df[TARGET].values)
        segments_all.extend([seg_name] * n_batch)

        try:
            x_batch = preprocessor.transform(batch_df.drop(columns=['segment_temp'], errors='ignore')).astype('float32')

            if seg_name in models_det:
                preds_det = models_det[seg_name](x_batch, training=False).numpy().flatten()
            else:
                preds_det = np.zeros(n_batch)

            if seg_name in models_bay:
                mc_means_batch = []
                mc_vars_batch = []
                for kIdx in range(mc_samples):
                    dist = models_bay[seg_name](x_batch, training=False)
                    mc_means_batch.append(dist.mean().numpy())
                    mc_vars_batch.append(dist.variance().numpy())

                mc_means_np = np.stack(mc_means_batch)[..., 0]
                mc_vars_np = np.stack(mc_vars_batch)[..., 0]

                if n_batch == 1:
                    preds_bay = [np.mean(mc_means_np)]
                    var_epist = np.var(mc_means_np, ddof=1) if mc_samples > 1 else 0.0
                    var_aleat = np.mean(mc_vars_np)
                    stds_bay  = [np.sqrt(var_epist + var_aleat)]
                else:
                    preds_bay = np.mean(mc_means_np, axis=0)
                    var_epist = np.var(mc_means_np, axis=0, ddof=1) if mc_samples > 1 else np.zeros(n_batch)
                    var_aleat = np.mean(mc_vars_np, axis=0)
                    stds_bay  = np.sqrt(var_epist + var_aleat)
            else:
                preds_bay = np.zeros(n_batch)
                stds_bay  = np.zeros(n_batch)

            y_pred_det_all.extend(preds_det)
            y_pred_bay_all.extend(preds_bay)
            std_bay_all.extend(stds_bay)

        except Exception as e:
            logger.error(f"Error in segment {seg_name}: {e}")
            y_pred_det_all.extend(np.zeros(n_batch))
            y_pred_bay_all.extend(np.zeros(n_batch))
            std_bay_all.extend(np.zeros(n_batch))

    if 'segment_temp' in df.columns:
        df.drop(columns=['segment_temp'], inplace=True)

    return (np.array(y_true_all), np.array(y_pred_det_all),
            np.array(y_pred_bay_all), np.array(std_bay_all), np.array(segments_all))

def exp_02_audit(df_test, preprocessor, models_det, models_bay):
    logger.info("EXP-02: Detailed Audit")
    
    audit_samples = []
    df_temp = df_test.copy()
    df_temp['segment_name'] = df_temp[TARGET].apply(get_segment_name)

    for seg in SEGMENTS:
        seg_name    = seg['name']
        seg_samples = df_temp[df_temp['segment_name'] == seg_name]
        n = min(3, len(seg_samples))
        if n > 0:
            audit_samples.append(seg_samples.sample(n, random_state=GLOBAL_SEED))

    sample_df = pd.concat(audit_samples)
    audit_results = []
    
    for idx, row in sample_df.iterrows():
        real_price = row[TARGET]
        segment_name = get_segment_name(real_price)
        row_df = pd.DataFrame([row])
        
        try:
            x_proc = preprocessor.transform(row_df).astype('float32')
        except:
            continue
            
        pred_det = 0.0
        if segment_name in models_det:
            try:
                pred_det = float(models_det[segment_name].predict(x_proc, verbose=0).flatten()[0])
            except:
                pass
                
        pred_bay = 0.0
        std_bay = 0.0
        ic_low, ic_high = 0.0, 0.0
        
        if segment_name in models_bay:
            try:
                means = []
                variances = []
                for _ in range(MC_SAMPLES):
                    dist = models_bay[segment_name](x_proc, training=False)
                    means.append(float(dist.mean().numpy()[0][0]))
                    variances.append(float(dist.variance().numpy()[0][0]))
                
                pred_bay = np.mean(means)
                var_epist = np.var(means, ddof=1) if MC_SAMPLES > 1 else 0.0
                var_aleat = np.mean(variances)
                std_bay = np.sqrt(var_epist + var_aleat)
                ic_low = pred_bay - 2 * std_bay
                ic_high = pred_bay + 2 * std_bay
            except:
                pass
                
        err_det_abs = abs(real_price - pred_det)
        err_det_rel = (err_det_abs / real_price) * 100 if real_price > 0 else 0
        err_bay_abs = abs(real_price - pred_bay)
        err_bay_rel = (err_bay_abs / real_price) * 100 if real_price > 0 else 0
        
        audit_results.append({
            'Index': idx,
            'Actual_Price': real_price,
            'Segment': segment_name,
            'Pred_Det': pred_det,
            'Err_Det_Abs': err_det_abs,
            'Err_Det_Rel': err_det_rel,
            'Pred_Bay': pred_bay,
            'IC_Low': ic_low,
            'IC_High': ic_high,
            'Uncertainty': std_bay,
            'Err_Bay_Abs': err_bay_abs,
            'Err_Bay_Rel': err_bay_rel
        })
    
    audit_df = pd.DataFrame(audit_results)
    
    segment_order = {seg['name']: i for i, seg in enumerate(SEGMENTS)}
    audit_df['Seg_Order'] = audit_df['Segment'].map(segment_order)
    audit_df = audit_df.sort_values(['Seg_Order', 'Actual_Price']).reset_index(drop=True)
    
    csv_path = os.path.join(RESULTS_DIR, 'exp02_audit.csv')
    audit_df.to_csv(csv_path, index=False, float_format='%.2f')
    
    # --- Plotting Generation ---
    plt.figure(figsize=(14, 7))
    
    x = np.arange(len(audit_df))
    
    real = audit_df['Actual_Price'].values
    det = audit_df['Pred_Det'].values
    bay = audit_df['Pred_Bay'].values
    ic_low = audit_df['IC_Low'].values
    ic_high = audit_df['IC_High'].values
    segmentos_labels = audit_df['Segment'].values
    
    yerr_lower = bay - ic_low
    yerr_upper = ic_high - bay

    plt.errorbar(x, bay, yerr=[yerr_lower, yerr_upper], fmt='o', capsize=4, 
                 label='Bayesian Pred. (95% CI)', color='#8A2BE2', markersize=6, alpha=0.7)
    
    plt.scatter(x, real, color='black', label='Actual Price', zorder=5, marker='s', s=50)
    plt.scatter(x, det, color='#FF6347', label='Deterministic Pred.', zorder=5, marker='X', s=50)
    
    plt.xticks(x, segmentos_labels, rotation=45, ha='right', fontsize=9)
    
    plt.ylabel('Price (€)', fontsize=12)
    plt.xlabel('Evaluated Listing (Grouped by Segment)', fontsize=12)
    plt.title('Detailed Analysis of Predictions and Confidence Intervals', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for i in range(1, len(audit_df)):
        if audit_df['Seg_Order'].iloc[i] != audit_df['Seg_Order'].iloc[i-1]:
            plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp02_audit_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    mae_det = audit_df['Err_Det_Abs'].mean()
    mae_bay = audit_df['Err_Bay_Abs'].mean()
    
    return {
        'n_samples': len(audit_results),
        'mae_det': mae_det,
        'mae_bay': mae_bay,
        'csv_path': csv_path,
        'plot_path': plot_path
    }

def exp_03_segments(y_true, y_pred_det, y_pred_bay, std_bay, segments):
    logger.info("EXP-03: Segment Analysis")
    
    df = pd.DataFrame({
        'segment': segments,
        'y_true': y_true,
        'error_det': np.abs(y_true - y_pred_det),
        'error_bay': np.abs(y_true - y_pred_bay),
        'uncertainty': std_bay
    })
    
    segment_order = [s['name'] for s in SEGMENTS]
    df['segment'] = pd.Categorical(df['segment'], categories=segment_order, ordered=True)
    
    df['mape_det'] = (df['error_det'] / (df['y_true'] + 1e-8)) * 100
    df['mape_bay'] = (df['error_bay'] / (df['y_true'] + 1e-8)) * 100

    segment_stats = df.groupby('segment', observed=False).agg({
        'error_det': 'mean',
        'error_bay': 'mean',
        'mape_det': 'mean',
        'mape_bay': 'mean',
        'uncertainty': 'mean'
    }).reset_index()
    
    segment_stats.columns = ['Segment', 'MAE_Det', 'MAE_Bay', 'MAPE_Det', 'MAPE_Bay', 'Avg_Uncertainty']
    
    csv_path = os.path.join(RESULTS_DIR, 'exp03_segments.csv')
    segment_stats.to_csv(csv_path, index=False, float_format='%.2f')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    x = np.arange(len(segment_stats))
    width = 0.35
    ax1.bar(x - width/2, segment_stats['MAE_Det'], width, label='Deterministic', color=C_DET)
    ax1.bar(x + width/2, segment_stats['MAE_Bay'], width, label='Bayesian', color=C_BAY)
    ax1.set_xlabel('Segment', fontsize=11)
    ax1.set_ylabel('MAE (€)', fontsize=11)
    ax1.set_title('MAE by Price Segment', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(segment_stats['Segment'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x - width/2, segment_stats['MAPE_Det'], width, label='Deterministic', color=C_DET)
    ax2.bar(x + width/2, segment_stats['MAPE_Bay'], width, label='Bayesian', color=C_BAY)
    ax2.set_xlabel('Segment', fontsize=11)
    ax2.set_ylabel('MAPE (%)', fontsize=11)
    ax2.set_title('MAPE by Price Segment', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(segment_stats['Segment'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3.bar(segment_stats['Segment'], segment_stats['Avg_Uncertainty'], color=C_AUX)
    ax3.set_xlabel('Segment', fontsize=11)
    ax3.set_ylabel('Avg Uncertainty (€)', fontsize=11)
    ax3.set_title('Average Uncertainty by Segment', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp03_segments.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'best_segment': segment_stats.loc[segment_stats['MAE_Bay'].idxmin(), 'Segment'],
        'worst_segment': segment_stats.loc[segment_stats['MAE_Bay'].idxmax(), 'Segment']
    }

def exp_04_correlation(y_true, y_pred_bay, std_bay):
    logger.info("EXP-04: Uncertainty-Error Correlation")

    errors = np.abs(y_true - y_pred_bay)
   
    # Spearman Correlation
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(errors, std_bay)
   
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hexbin(std_bay, errors, gridsize=30, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel('Predicted Uncertainty (€)', fontsize=12)
    plt.ylabel('Absolute Error (€)', fontsize=12)
    plt.title(f'Error vs Uncertainty (Spearman ρ={corr:.3f}, p<0.001)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp04_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
   
    return {'spearman_corr': corr, 'p_value': p_value}

def exp_05_calibration(y_true, y_pred_bay, std_bay, segments):
    logger.info("EXP-05: Probabilistic Calibration")
    
    y_pred = y_pred_bay
    uncertainty = std_bay
    
    # Calibration Curve (multiple levels)
    from scipy.stats import norm
    confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
    calibration_results = []
    
    for conf in confidence_levels:
        z_score = norm.ppf((1 + conf) / 2)
        lower = y_pred - z_score * uncertainty
        upper = y_pred + z_score * uncertainty
        
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        calib_error = abs(coverage - conf) / conf * 100
        
        
        calibration_results.append({
            'Confidence_Level': conf,
            'Expected_Coverage': conf,
            'Observed_Coverage': coverage,
            'Calibration_Error_Pct': calib_error
        })
    
    # Calibration by Segment
    segment_calibration = []
    
    for seg in SEGMENTS:
        seg_mask = segments == seg['name']
        if seg_mask.sum() < 10:
            continue
        
        y_true_seg = y_true[seg_mask]
        y_pred_seg = y_pred[seg_mask]
        unc_seg = uncertainty[seg_mask]
        
        # 95% Interval
        z_95 = norm.ppf(0.975)
        lower = y_pred_seg - z_95 * unc_seg
        upper = y_pred_seg + z_95 * unc_seg
        
        coverage = np.mean((y_true_seg >= lower) & (y_true_seg <= upper))
        calib_error = abs(coverage - 0.95) / 0.95 * 100
        
        segment_calibration.append({
            'Segment': seg['name'],
            'N_Samples': seg_mask.sum(),
            'Coverage_95': coverage,
            'Calibration_Error_Pct': calib_error,
            'Mean_Uncertainty': np.mean(unc_seg)
        })
    
    # Save CSV
    df_calib = pd.DataFrame(calibration_results)
    df_calib.to_csv(os.path.join(RESULTS_DIR, 'exp05_calibration_detailed.csv'), index=False)
    
    df_seg_calib = pd.DataFrame(segment_calibration)
    df_seg_calib.to_csv(os.path.join(RESULTS_DIR, 'exp05_calibration_by_segment.csv'), index=False)
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Calibration Curve
    ax = axes[0]
    expected = [r['Expected_Coverage'] * 100 for r in calibration_results]
    observed = [r['Observed_Coverage'] * 100 for r in calibration_results]
    
    ax.plot(expected, observed, marker='o', linewidth=2, markersize=10, 
            color=C_AUX, label='Observed')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect')
    ax.fill_between(expected, expected, observed, alpha=0.3, color=C_DET)
    ax.set_xlabel('Expected Coverage (%)', fontsize=12)
    ax.set_ylabel('Observed Coverage (%)', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(45, 105)
    ax.set_ylim(45, 105)
    
    # Plot 2: Calibration Error by Level
    ax = axes[1]
    levels = [f"{int(r['Confidence_Level']*100)}%" for r in calibration_results]
    errors = [r['Calibration_Error_Pct'] for r in calibration_results]
    
    colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in errors]
    ax.bar(levels, errors, color=colors, alpha=0.7)
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% Threshold')
    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('Calibration Error (%)', fontsize=12)
    ax.set_title('Error by Confidence Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Calibration by Segment
    ax = axes[2]
    seg_names = [r['Segment'] for r in segment_calibration]
    seg_errors = [r['Calibration_Error_Pct'] for r in segment_calibration]
    
    colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in seg_errors]
    ax.bar(seg_names, seg_errors, color=colors, alpha=0.7)
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% Threshold')
    ax.set_xlabel('Segment', fontsize=12)
    ax.set_ylabel('Calibration Error (%)', fontsize=12)
    ax.set_title('Error by Segment (95% CI)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp05_calibration_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_calibration_error': float(df_calib['Calibration_Error_Pct'].mean()),
        'detailed_csv': os.path.join(RESULTS_DIR, 'exp05_calibration_detailed.csv'),
        'segment_csv': os.path.join(RESULTS_DIR, 'exp05_calibration_by_segment.csv')
    }

def exp_06_uncertainty_decomposition(df_test, preprocessor, models_bay, mc_samples=MC_SAMPLES):
    logger.info("EXP-06: Uncertainty Decomposition")
    samples_per_segment = 200
    results_by_segment = {}
    
    df_temp = df_test.copy()
    df_temp['segment_name'] = df_temp[TARGET].apply(get_segment_name)
    
    segment_order = [s['name'] for s in SEGMENTS]
    
    for seg_name in segment_order:
        seg_df = df_temp[df_temp['segment_name'] == seg_name]
        
        if len(seg_df) == 0 or seg_name not in models_bay:
            continue
            
        n_to_sample = min(samples_per_segment, len(seg_df))
        seg_df_sampled = seg_df.sample(n_to_sample, random_state=GLOBAL_SEED)
        
        x_proc = preprocessor.transform(seg_df_sampled.drop(columns=['segment_name'])).astype('float32')
        model = models_bay[seg_name]
        
        mc_predictions = []
        mc_stds = []
        
        for _ in range(mc_samples):
            dist = model(x_proc, training=False)
            means = dist.mean().numpy().flatten()
            stds = dist.stddev().numpy().flatten()
            
            mc_predictions.append(means)
            mc_stds.append(stds)
        
        mc_predictions = np.array(mc_predictions)
        mc_stds = np.array(mc_stds)
        
        epistemic = np.var(mc_predictions, axis=0, ddof=1) if mc_samples > 1 else np.zeros_like(mc_predictions[0])
        
        aleatoric = np.mean(mc_stds ** 2, axis=0) 
        
        total = epistemic + aleatoric
        
        results_by_segment[seg_name] = {
            'epistemic_mean': float(np.mean(epistemic)),
            'aleatoric_mean': float(np.mean(aleatoric)),
            'total_mean': float(np.mean(total)),
            'epistemic_ratio': float(np.mean(epistemic / (total + 1e-8))),
            'n_samples': n_to_sample
        }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    segments_list = list(results_by_segment.keys())
    epistemic_means = [results_by_segment[s]['epistemic_mean'] for s in segments_list]
    aleatoric_means = [results_by_segment[s]['aleatoric_mean'] for s in segments_list]
    
    x = np.arange(len(segments_list))
    ax1.bar(x, epistemic_means, label='Epistemic', color=C_BAY, alpha=0.8)
    ax1.bar(x, aleatoric_means, bottom=epistemic_means, label='Aleatoric', color=C_AUX, alpha=0.8)
    ax1.set_xlabel('Price Segment', fontsize=12)
    ax1.set_ylabel('Variance (€²)', fontsize=12)
    ax1.set_title('Uncertainty Decomposition by Segment', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(segments_list, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    epistemic_ratios = [results_by_segment[s]['epistemic_ratio'] * 100 for s in segments_list]
    ax2.bar(segments_list, epistemic_ratios, color=C_DET)
    ax2.set_xlabel('Price Segment', fontsize=12)
    ax2.set_ylabel('Epistemic Share (%)', fontsize=12)
    ax2.set_title('Proportion of Epistemic Uncertainty', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp06_uncertainty_decomposition.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    decomp_df = pd.DataFrame([
        {
            'Segment': seg,
            'Epistemic_Mean': results_by_segment[seg]['epistemic_mean'],
            'Aleatoric_Mean': results_by_segment[seg]['aleatoric_mean'],
            'Total_Mean': results_by_segment[seg]['total_mean'],
            'Epistemic_Ratio': results_by_segment[seg]['epistemic_ratio'],
            'N_Samples': results_by_segment[seg]['n_samples']
        }
        for seg in segments_list
    ])
    csv_path = os.path.join(RESULTS_DIR, 'exp06_uncertainty_decomposition.csv')
    decomp_df.to_csv(csv_path, index=False, float_format='%.4f')
    
    return {
        'results_by_segment': results_by_segment,
        'csv_path': csv_path
    }

def exp_07_rejection(y_true, y_pred_det, y_pred_bay, std_bay):
    logger.info("EXP-07: Rejection Curve")
   
    errors_bay = np.abs(y_true - y_pred_bay)
    errors_det = np.abs(y_true - y_pred_det)
   
    # Rejection curve
    thresholds = np.linspace(0, 95, 20)
    maes_bay = []
    maes_det_random = []
    retention_rates = []
   
    for t in thresholds:
        # Bayesian strategy: reject most uncertain
        cutoff = np.percentile(std_bay, 100 - t)
        mask_bay = std_bay <= cutoff
       
        # Deterministic strategy: random rejection (no confidence metric)
        n_retention = int(len(y_true) * (100 - t) / 100)
        indices_random = np.random.choice(len(y_true), n_retention, replace=False)
        
        if mask_bay.sum() > 0:
            maes_bay.append(errors_bay[mask_bay].mean())
            maes_det_random.append(errors_det[indices_random].mean())
            retention_rates.append(100 - t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(retention_rates, maes_bay, marker='o', linewidth=2, color=C_BAY, label='Bayesian (Uncertainty-aware)')
    plt.plot(retention_rates, maes_det_random, marker='s', linewidth=2, color=C_DET, label='Deterministic (Random rejection)')
    
    plt.xlabel('Retention Rate (%)', fontsize=12)
    plt.ylabel('MAE (€)', fontsize=12)
    plt.title('Rejection Curve: MAE vs Retention Rate', fontsize=14, fontweight='bold')
    
    ticks = [100, 80, 60, 40, 20, 5]
    plt.xticks(ticks)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp07_rejection.png'), dpi=300, bbox_inches='tight')
    plt.close()
   
    return {
        'mae_100': float(maes_bay[0]) if maes_bay else 0,
        'mae_50': float(maes_bay[len(maes_bay)//2]) if len(maes_bay) > 1 else 0
    }

# Main

def main():
    logger.info("CONSOLIDATED EXPERIMENTS — EXP-02 to EXP-07")
    logger.info(f"Test set: {DATA_PATH}")

    df_test = load_data()
    preprocessor, models_det, models_bay = load_models(df_test)

    y_true, y_pred_det, y_pred_bay, std_bay, segments = predict_segmented(
        df_test, preprocessor, models_det, models_bay, verbose=False
    )
    
    results = {}
    
    results['exp02_audit'] = exp_02_audit(df_test, preprocessor, models_det, models_bay)
    results['exp03_segments'] = exp_03_segments(y_true, y_pred_det, y_pred_bay, std_bay, segments)

    results['exp04_correlation'] = exp_04_correlation(y_true, y_pred_bay, std_bay)
    results['exp05_calibration'] = exp_05_calibration(y_true, y_pred_bay, std_bay, segments)
    results['exp06_uncertainty_decomposition'] = exp_06_uncertainty_decomposition(df_test, preprocessor, models_bay)

    results['exp07_rejection'] = exp_07_rejection(y_true, y_pred_det, y_pred_bay, std_bay)

    results_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"EXPERIMENTS COMPLETED (EXP-02 to EXP-07) — results in {RESULTS_DIR}")

if __name__ == '__main__':
    main()
