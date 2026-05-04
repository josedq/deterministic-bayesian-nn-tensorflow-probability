"""
exp08_multi_city.py — EXP-08: Multi-City Generalization Study.
Trains deterministic and Bayesian segmented models on multiple cities using
percentile-based price segmentation. MAPE is the primary cross-city metric
since prices are in different currencies.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logger_config import setup_logger
from config.constants import (
    TARGET, GLOBAL_SEED,
    NUMERICAL_STANDARD_FEATURES as NUMERICAL_STANDARD,
    NUMERICAL_REVIEW_FEATURES as NUMERICAL_REVIEW,
    CATEGORICAL_FEATURES as CATEGORICAL,
    COLS_TO_DROP, SEGMENTS,
)
from utils.data_utils import count_items, days_since

logger = setup_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

tf.get_logger().setLevel('ERROR')
tfd  = tfp.distributions
tfpl = tfp.layers

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEGMENT_NAMES = [s['name'] for s in SEGMENTS]

# Columns specific to exp08: drop after deriving features
_COLS_TO_DROP_EXP08 = COLS_TO_DROP + ['amenities', 'host_since', 'first_review', 'last_review']

TEST_SPLIT     = 0.2
MAX_EPOCHS     = 100
BATCH_SIZE     = 32
PATIENCE       = 10
VAL_SPLIT      = 0.2
MC_SAMPLES     = 200

PERCENTILE_CUTS = [30, 50, 70, 82.5, 90]

CITIES = {
    'Tokyo':   os.path.join(BASE_DIR, 'data', 'tokyo',   'listings.csv'),
    'Madrid':      os.path.join(BASE_DIR, 'data', 'madrid',      'listings.csv'),
    'Barcelona':   os.path.join(BASE_DIR, 'data', 'barcelona',   'listings.csv'),
    'Mexico City': os.path.join(BASE_DIR, 'data', 'mexico-city', 'listings.csv'),
}

CITIES_CURRENCY = {
    'Tokyo': 'JPY', 'Madrid': 'EUR', 'Barcelona': 'EUR', 'Mexico City': 'MXN'
}

def load_city_raw(raw_path):
    """Loads raw Airbnb CSV, applies feature engineering, returns cleaned DataFrame."""
    logger.info(f"Loading {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)

    # Parse price
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].replace('[\\$,]', '', regex=True).astype(float)
    df = df[df[TARGET] > 0]

    # Feature engineering
    df['amenities_count']          = df['amenities'].apply(count_items)
    df['host_days_active']         = df['host_since'].apply(days_since)
    df['days_since_first_review']  = df['first_review'].apply(days_since)
    df['days_since_last_review']   = df['last_review'].apply(days_since)

    for col in ['host_is_superhost', 'host_identity_verified', 'instant_bookable']:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

    df = df.drop(columns=_COLS_TO_DROP_EXP08, errors='ignore')

    # Keep only needed feature columns + target
    keep = [TARGET] + [c for c in NUMERICAL_STANDARD + NUMERICAL_REVIEW + CATEGORICAL if c in df.columns]
    df = df[keep]

    if len(df) == 0:
        raise ValueError(
            f"No valid prices found in {raw_path}. "
            f"The 'price' column may be empty (common in some Inside Airbnb exports). "
            f"Try a different city dataset."
        )

    logger.info(f"Loaded {len(df)} rows | price range: {df[TARGET].min():.0f} - {df[TARGET].max():.0f}")
    return df

def compute_segments(prices):
    """
    Returns 5 segment dicts using percentile cuts [30, 50, 70, 82.5].
    The 90th percentile is used as the upper price cap (included in last segment).
    """
    cuts = [np.percentile(prices, p) for p in PERCENTILE_CUTS]
    p30, p50, p70, p82, p90 = cuts
    segs = [
        {'name': 'Low',          'min': 0,    'max': p30},
        {'name': 'Low_Medium',   'min': p30,  'max': p50},
        {'name': 'Medium',       'min': p50,  'max': p70},
        {'name': 'Medium_High',  'min': p70,  'max': p82},
        {'name': 'High',         'min': p82,  'max': p90},
    ]
    return segs, p90

def create_preprocessor(available_cats):
    """Returns an unfitted ColumnTransformer using available categorical columns."""
    cats = [c for c in CATEGORICAL if c in available_cats]
    nums_std = [c for c in NUMERICAL_STANDARD if c in available_cats]
    nums_rev = [c for c in NUMERICAL_REVIEW if c in available_cats]

    review_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0)),
        ('scl', StandardScaler()),
    ])
    std_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    transformers = []
    if nums_std: transformers.append(('num', std_pipe, nums_std))
    if nums_rev: transformers.append(('rev', review_pipe, nums_rev))
    if cats:     transformers.append(('cat', cat_pipe, cats))

    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)

def build_det_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model

def build_bay_model(input_dim, n_train):
    kl_w = 1.0 / n_train
    div_fn = lambda q, p, _: tfd.kl_divergence(q, p) * kl_w

    def normal_fn(t):
        return tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        tfpl.DenseFlipout(128, activation='relu', kernel_divergence_fn=div_fn),
        tfpl.DenseFlipout(64,  activation='relu', kernel_divergence_fn=div_fn),
        tfpl.DenseFlipout(32,  activation='relu', kernel_divergence_fn=div_fn),
        tfpl.DenseFlipout(2,   activation=None,   kernel_divergence_fn=div_fn),
        tfpl.DistributionLambda(normal_fn),
    ])

    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=nll)
    return model

def _callbacks():
    return [keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]

def train_det(x_tr, y_tr, x_te, y_te):
    model = build_det_model(x_tr.shape[1])
    t0 = time.time()
    hist = model.fit(
        x_tr, y_tr,
        validation_split=VAL_SPLIT,
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        verbose=0, callbacks=_callbacks(),
    )
    elapsed = time.time() - t0

    # Training MAE
    tr_mae = float(model.evaluate(x_tr, y_tr, verbose=0)[1])
    y_tr_pred = model.predict(x_tr, verbose=0).flatten()
    tr_mape = float(np.mean(np.abs((y_tr - y_tr_pred) / (y_tr + 1e-8))) * 100)

    # Test MAE
    te_mae = float(model.evaluate(x_te, y_te, verbose=0)[1])
    y_te_pred = model.predict(x_te, verbose=0).flatten()
    te_rmse = float(np.sqrt(np.mean((y_te - y_te_pred) ** 2)))
    te_mape = float(np.mean(np.abs((y_te - y_te_pred) / (y_te + 1e-8))) * 100)

    return {
        'mae_train': tr_mae, 'mape_train': tr_mape,
        'mae_test':  te_mae, 'mape_test':  te_mape,
        'rmse_test': te_rmse,
        'epochs_trained': len(hist.history['loss']),
        'training_time_s': round(elapsed, 2),
    }, hist.history.get('val_loss', []), hist.history.get('loss', [])

def train_bay(x_tr, y_tr, x_te, y_te):
    model = build_bay_model(x_tr.shape[1], len(x_tr))
    t0 = time.time()
    hist = model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        ],
    )
    elapsed = time.time() - t0

    # MC evaluation on test set
    mc_preds = np.array([model(x_te, training=False).mean().numpy().flatten() for _ in range(MC_SAMPLES)])
    y_te_pred = mc_preds.mean(axis=0)

    te_mae  = float(np.mean(np.abs(y_te - y_te_pred)))
    te_rmse = float(np.sqrt(np.mean((y_te - y_te_pred) ** 2)))
    te_mape = float(np.mean(np.abs((y_te - y_te_pred) / (y_te + 1e-8))) * 100)

    # Training set MC eval
    mc_tr = np.array([model(x_tr, training=False).mean().numpy().flatten() for _ in range(MC_SAMPLES)])
    y_tr_pred = mc_tr.mean(axis=0)
    tr_mae  = float(np.mean(np.abs(y_tr - y_tr_pred)))
    tr_mape = float(np.mean(np.abs((y_tr - y_tr_pred) / (y_tr + 1e-8))) * 100)

    return {
        'mae_train': tr_mae, 'mape_train': tr_mape,
        'mae_test':  te_mae, 'mape_test':  te_mape,
        'rmse_test': te_rmse,
        'epochs_trained': len(hist.history['loss']),
        'training_time_s': round(elapsed, 2),
    }, hist.history.get('val_loss', []), hist.history.get('loss', [])

def run_city(city_name, raw_path):
    logger.info(f"CITY: {city_name}")

    # Load raw data
    df = load_city_raw(raw_path)
    y_all = df[TARGET].values

    # Compute price cap and segments from ALL prices
    segments, price_cap = compute_segments(y_all)
    df = df[df[TARGET] <= price_cap].copy()
    logger.info(f"Price cap (p90): {price_cap:.2f} | Retained: {len(df)} rows")
    for s in segments:
        logger.info(f"{s['name']:15s}: {s['min']:.1f} - {s['max']:.1f}")

    # Split
    feat_cols = [c for c in NUMERICAL_STANDARD + NUMERICAL_REVIEW + CATEGORICAL if c in df.columns]
    X = df[feat_cols]
    y = df[TARGET].values.astype('float32')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SPLIT, random_state=GLOBAL_SEED)

    # Fit preprocessor on training split
    preprocessor = create_preprocessor(X_tr.columns.tolist())
    X_tr_proc = preprocessor.fit_transform(X_tr).astype('float32')
    X_te_proc = preprocessor.transform(X_te).astype('float32')
    input_dim = X_tr_proc.shape[1]
    logger.info(f"Input dimension: {input_dim} | Train: {len(y_tr)} | Test: {len(y_te)}")

    # Loop over segments
    records = []
    # Histories: dict of model_type -> {seg_name -> (val_loss, train_loss)}
    histories = {'deterministic': {}, 'bayesian': {}}

    for seg in segments:
        name = seg['name']
        mask_tr = (y_tr > seg['min']) & (y_tr <= seg['max'])
        mask_te = (y_te > seg['min']) & (y_te <= seg['max'])

        x_tr_s, y_tr_s = X_tr_proc[mask_tr], y_tr[mask_tr]
        x_te_s, y_te_s = X_te_proc[mask_te], y_te[mask_te]

        n_tr, n_te = len(y_tr_s), len(y_te_s)
        logger.info(f"Segment {name}: n_train={n_tr}, n_test={n_te}")

        if n_tr < 50 or n_te < 10:
            logger.info(f"Skipping (insufficient samples)")
            continue

        base_row = {
            'city': city_name, 'segment': name,
            'price_min': round(seg['min'], 2), 'price_max': round(seg['max'], 2),
            'n_train': n_tr, 'n_test': n_te, 'input_dim': input_dim,
        }

        # Deterministic
        logger.info(f"Training deterministic...")
        det_metrics, det_val_loss, det_train_loss = train_det(x_tr_s, y_tr_s, x_te_s, y_te_s)
        records.append({**base_row, 'model_type': 'deterministic', **det_metrics})
        histories['deterministic'][name] = (det_val_loss, det_train_loss)

        # Bayesian
        logger.info(f"Training Bayesian...")
        bay_metrics, bay_val_loss, bay_train_loss = train_bay(x_tr_s, y_tr_s, x_te_s, y_te_s)
        records.append({**base_row, 'model_type': 'bayesian', **bay_metrics})
        histories['bayesian'][name] = (bay_val_loss, bay_train_loss)

    return records, histories

# Plotting
def plot_training_curves(city_name, histories):
    """One figure per city: 2 rows (det / bay) × up to 5 columns (segments)."""
    seg_names = list(histories['deterministic'].keys())
    n_segs = len(seg_names)
    if n_segs == 0:
        return

    fig, axes = plt.subplots(2, n_segs, figsize=(4 * n_segs, 7))
    if n_segs == 1:
        axes = axes.reshape(2, 1)

    colors = ['#FF6B6B', '#4ECDC4', '#6200EA', '#FFB347', '#2ECC71']

    for col_i, seg_name in enumerate(seg_names):
        color = colors[col_i % len(colors)]

        for row_i, model_type in enumerate(['deterministic', 'bayesian']):
            ax = axes[row_i][col_i]
            val_loss, train_loss = histories[model_type].get(seg_name, ([], []))

            if train_loss:
                ax.plot(train_loss, label='train', color=color, alpha=0.5, linestyle='--')
            if val_loss:
                ax.plot(val_loss,   label='val',   color=color, linewidth=2)

            ax.set_title(f'{seg_name}', fontsize=11, fontweight='bold')
            if col_i == 0:
                label = 'Deterministic' if model_type == 'deterministic' else 'Bayesian'
                ax.set_ylabel(f'{label}\nLoss', fontsize=10)
                
            if model_type == 'bayesian' and len(val_loss) > 5:
                y_max = max(max(val_loss[5:]), max(train_loss[5:]) if len(train_loss) > 5 else 0)
                if y_max > 0:
                    ax.set_ylim(0, y_max * 1.2)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{city_name} — Training Curves by Segment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    city_slug = city_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'exp08_{city_slug}_training_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved: {path}")

def plot_summary(df_all):
    """
    Summary plot: MAPE per city for det vs bay (test set).
    """
    cities = df_all['City'].unique().tolist() if 'City' in df_all.columns else df_all['city'].unique().tolist()
    city_col = 'City' if 'City' in df_all.columns else 'city'
    model_col = 'Model_Type' if 'Model_Type' in df_all.columns else 'model_type'
    n_test_col = 'N_Test' if 'N_Test' in df_all.columns else 'n_test'
    mape_col = 'MAPE_Test' if 'MAPE_Test' in df_all.columns else 'mape_test'

    # Weighted MAPE per city per model type
    rows = []
    for city in cities:
        for mt in ['deterministic', 'bayesian']:
            sub = df_all[(df_all[city_col] == city) & (df_all[model_col] == mt)]
            if sub.empty:
                continue
            total_n = sub[n_test_col].sum()
            w_mape = (sub[mape_col] * sub[n_test_col]).sum() / total_n
            rows.append({'city': city, 'model': mt, 'w_mape': w_mape, 'label': city})

    summary = pd.DataFrame(rows)
    C_DET, C_BAY = '#FF6B6B', '#6200EA'
    x = np.arange(len(cities))
    w = 0.35
    city_labels = [rows[i*2]['label'] for i in range(len(cities))]

    fig, ax = plt.subplots(figsize=(8, 6))

    for j, (mt, col) in enumerate([('deterministic', C_DET), ('bayesian', C_BAY)]):
        vals = [summary.loc[(summary['city']==c) & (summary['model']==mt), 'w_mape'].values
                for c in cities]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + j*w - w/2, vals, w, label=mt.capitalize(), color=col, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(city_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel('Weighted MAPE (%)', fontsize=12)
    ax.set_title('Weighted MAPE by City', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp08_multi_city_generalization_summary.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot saved: {path}")

def plot_segment_comparison(df_all):
    """Per-segment MAPE across all cities (det vs bay)."""
    seg_order = ['Low', 'Low_Medium', 'Medium', 'Medium_High', 'High']
    cities = df_all['city'].unique().tolist()
    n_seg  = len(seg_order)
    x      = np.arange(n_seg)
    city_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, mt in enumerate(['deterministic', 'bayesian']):
        ax = axes[ax_i]
        for c_i, city in enumerate(cities):
            currency = CITIES_CURRENCY.get(city, '?')
            vals = []
            for seg in seg_order:
                row = df_all[(df_all['city']==city) & (df_all['model_type']==mt) & (df_all['segment']==seg)]
                vals.append(float(row['mape_test'].values[0]) if not row.empty else 0)
            offset = (c_i - len(cities)/2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=f'{city} ({currency})',
                   color=city_colors[c_i], alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(seg_order, rotation=20, ha='right')
        ax.set_ylabel('MAPE test (%)', fontsize=11)
        ax.set_title(f'{mt.capitalize()} — MAPE per Segment', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp08_multi_city_segment_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Segment comparison plot saved: {path}")

# Main
def main():
    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    logger.info("EXP-08: Multi-City Generalization Study")
    logger.info(f"Cities: {list(CITIES.keys())}")
    logger.info(f"Percentile cuts: {PERCENTILE_CUTS}")

    all_records = []

    for city_name, raw_path in CITIES.items():
        if not os.path.exists(raw_path):
            logger.warning(f"Raw CSV not found for {city_name}: {raw_path} — skipping.")
            continue

        records, histories = run_city(city_name, raw_path)
        all_records.extend(records)

        # Training curves per city
        plot_training_curves(city_name, histories)

    if not all_records:
        logger.error("No results collected. Aborting.")
        return

    # Assemble and save CSV
    df_all = pd.DataFrame(all_records)
    col_order = [
        'city', 'model_type', 'segment', 'price_min', 'price_max',
        'n_train', 'n_test', 'input_dim',
        'mae_train', 'mae_test', 'mape_train', 'mape_test',
        'rmse_test', 'epochs_trained', 'training_time_s',
    ]
    df_all = df_all[[c for c in col_order if c in df_all.columns]]
    
    # Rename for consistency
    renames = {
        'city': 'City', 'model_type': 'Model_Type', 'segment': 'Segment',
        'price_min': 'Price_Min', 'price_max': 'Price_Max',
        'n_train': 'N_Train', 'n_test': 'N_Test', 'input_dim': 'Input_Dim',
        'mae_train': 'MAE_Train', 'mae_test': 'MAE_Test',
        'mape_train': 'MAPE_Train', 'mape_test': 'MAPE_Test',
        'rmse_test': 'RMSE_Test', 'epochs_trained': 'Epochs_Trained',
        'training_time_s': 'Training_Time_s'
    }
    df_csv = df_all.rename(columns=renames)
    
    csv_path = os.path.join(RESULTS_DIR, 'exp08_multi_city_generalization.csv')
    df_csv.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Results CSV saved: {csv_path}")

    # Summary visualizations
    plot_summary(df_all)
    plot_segment_comparison(df_all)

    logger.info("EXP-08 COMPLETED")
    logger.info(f"Generated files in {RESULTS_DIR}:")
    for city_name in CITIES:
        slug = city_name.lower().replace(' ', '_')
        logger.info(f"exp08_{slug}_training_curves.png")
    logger.info("exp08_multi_city_generalization.csv")
    logger.info("exp08_multi_city_generalization_summary.png")
    logger.info("exp08_multi_city_segment_comparison.png")

if __name__ == '__main__':
    main()
