"""
production_training.py — Train segmented deterministic and Bayesian models.
Saves all artifacts required by downstream experiments.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

tf.get_logger().setLevel('ERROR')

from utils.logger_config import setup_logger
logger = setup_logger(__name__)

from sklearn.model_selection import train_test_split
from datetime import datetime
import time

from config.constants import (
    TARGET, TARGET_CAP, GLOBAL_SEED, TEST_SPLIT_SIZE,
    ARCHITECTURE, SEGMENTS,
    NUMERICAL_STANDARD_FEATURES, NUMERICAL_REVIEW_FEATURES, CATEGORICAL_FEATURES,
)
from utils.data_utils import apply_feature_engineering
from utils.models import build_deterministic_model, build_bayesian_model
from utils.preprocessing import build_preprocessor

tfd = tfp.distributions

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH            = os.path.join(BASE_DIR, 'data', 'madrid', 'listings.csv')
ARTIFACTS_DIR        = os.path.join(BASE_DIR, 'artifacts')
RESULTS_TRAINING_DIR = os.path.join(BASE_DIR, 'results', 'training')

os.makedirs(os.path.join(ARTIFACTS_DIR, 'deterministic'), exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, 'bayesian'),      exist_ok=True)
os.makedirs(RESULTS_TRAINING_DIR, exist_ok=True)

MAX_EPOCHS          = 1000
BATCH_SIZE          = 32
VAL_SPLIT_SIZE      = 0.2
PATIENCE            = 20
N_POSTERIOR_SAMPLES = 200

tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

def load_and_preprocess_data():
    logger.info(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = apply_feature_engineering(df)
    x = df[NUMERICAL_STANDARD_FEATURES + NUMERICAL_REVIEW_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    return x, y

def create_preprocessor():
    return build_preprocessor()

def build_det_model(input_dim, segment_name):
    model = build_deterministic_model(input_dim)
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model

def build_bay_model(input_dim, n_train, segment_name):
    return build_bayesian_model(input_dim, n_train)

def train_deterministic(x_train, y_train, x_test, y_test, segment_name, input_dim):
    logger.info(f"Training Det [{segment_name}]")
    model = build_det_model(input_dim, segment_name)

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT_SIZE,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
    )
    training_time = time.time() - start_time

    test_mae = model.evaluate(x_test, y_test, verbose=0)[1]
    y_pred   = model.predict(x_test, verbose=0).flatten()
    rmse     = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape     = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    model_path = os.path.join(ARTIFACTS_DIR, 'deterministic', f'segment_{segment_name}.keras')
    model.save(model_path)
    logger.info(f"Saved deterministic model: {model_path}")

    return {
        'segment': segment_name,
        'model_type': 'deterministic',
        'test_mae': float(test_mae),
        'test_rmse': float(rmse),
        'test_mape': float(mape),
        'training_time_sec': float(training_time),
        'epochs_trained': len(history.history['loss']),
        'n_params': int(model.count_params())
    }

def train_bayesian(x_train, y_train, x_test, y_test, segment_name, input_dim):
    logger.info(f"Training Bay [{segment_name}]")
    model = build_bay_model(input_dim, len(x_train), segment_name)

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT_SIZE,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
    )
    training_time = time.time() - start_time

    predictions = []
    for _ in range(N_POSTERIOR_SAMPLES):
        y_pred_sample = model(x_test).mean().numpy().flatten()
        predictions.append(y_pred_sample)

    predictions  = np.array(predictions)
    y_pred_mean  = predictions.mean(axis=0)
    y_pred_std   = predictions.std(axis=0)

    test_mae       = np.mean(np.abs(y_test - y_pred_mean))
    rmse           = np.sqrt(np.mean((y_test - y_pred_mean) ** 2))
    mape           = np.mean(np.abs((y_test - y_pred_mean) / y_test)) * 100
    avg_uncertainty = np.mean(y_pred_std)

    weights_path = os.path.join(ARTIFACTS_DIR, 'bayesian', f'segment_{segment_name}_weights.h5')
    config_path  = os.path.join(ARTIFACTS_DIR, 'bayesian', f'segment_{segment_name}_config.json')
    model.save_weights(weights_path)
    with open(config_path, 'w') as f:
        json.dump({'input_dim': input_dim, 'n_train': len(x_train)}, f)
    logger.info(f"Saved Bayesian weights: {weights_path}")

    return {
        'segment': segment_name,
        'model_type': 'bayesian',
        'test_mae': float(test_mae),
        'test_rmse': float(rmse),
        'test_mape': float(mape),
        'avg_uncertainty': float(avg_uncertainty),
        'training_time_sec': float(training_time),
        'epochs_trained': len(history.history['loss']),
        'n_params': int(model.count_params())
    }

def main():
    logger.info("PRODUCTION TRAINING START")
    logger.info(f"Architecture: {ARCHITECTURE['hidden_layers']}, dropout={ARCHITECTURE['dropout_rate']}")
    logger.info(f"Segments: {len(SEGMENTS)} | Artifacts: {ARTIFACTS_DIR}")

    x, y = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SPLIT_SIZE, random_state=GLOBAL_SEED
    )

    logger.info("Fitting preprocessor...")
    preprocessor    = create_preprocessor()
    x_train_proc    = preprocessor.fit_transform(x_train).astype('float32')
    x_test_proc     = preprocessor.transform(x_test).astype('float32')
    y_train         = y_train.astype('float32')
    y_test          = y_test.astype('float32')
    input_dim       = x_train_proc.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    preprocessor_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved preprocessor: {preprocessor_path}")

    # Save raw train/test splits for reproducible evaluation
    df_raw = pd.read_csv(DATA_PATH)
    df_raw = df_raw.dropna(subset=[TARGET])
    df_raw[TARGET] = df_raw[TARGET].replace('[\\$,]', '', regex=True).astype(float)
    df_raw = df_raw[(df_raw[TARGET] > 0) & (df_raw[TARGET] <= TARGET_CAP)]
    df_raw_train, df_raw_test = train_test_split(
        df_raw, test_size=TEST_SPLIT_SIZE, random_state=GLOBAL_SEED
    )
    data_dir  = os.path.dirname(DATA_PATH)
    train_csv = os.path.join(data_dir, 'listings_production_training.csv')
    test_csv  = os.path.join(data_dir, 'listings_production_test.csv')
    df_raw_train.to_csv(train_csv, index=False)
    df_raw_test.to_csv(test_csv,  index=False)
    logger.info(f"Saved train split ({len(df_raw_train)} rows): {train_csv}")
    logger.info(f"Saved test split  ({len(df_raw_test)} rows): {test_csv}")

    all_results = []
    for segment in SEGMENTS:
        seg_name = segment['name']
        logger.info(f"SEGMENT: {seg_name} ({segment['min']}-{segment['max']})")

        train_mask  = (y_train > segment['min']) & (y_train <= segment['max'])
        test_mask   = (y_test  > segment['min']) & (y_test  <= segment['max'])
        x_train_seg = x_train_proc[train_mask]
        y_train_seg = y_train[train_mask]
        x_test_seg  = x_test_proc[test_mask]
        y_test_seg  = y_test[test_mask]

        if len(y_train_seg) < 50 or len(y_test_seg) < 10:
            logger.warning(f"Skipping {seg_name}: insufficient samples")
            continue

        all_results.append(train_deterministic(
            x_train_seg, y_train_seg, x_test_seg, y_test_seg, seg_name, input_dim
        ))
        all_results.append(train_bayesian(
            x_train_seg, y_train_seg, x_test_seg, y_test_seg, seg_name, input_dim
        ))

    metrics_csv  = os.path.join(RESULTS_TRAINING_DIR, 'metrics_summary.csv')
    metrics_json = os.path.join(RESULTS_TRAINING_DIR, 'metrics_summary.json')
    pd.DataFrame(all_results).to_csv(metrics_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({
            'architecture': ARCHITECTURE,
            'segments': SEGMENTS,
            'results': all_results,
            'training_date': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Saved metrics: {metrics_csv}")
    logger.info("PRODUCTION TRAINING COMPLETED")

if __name__ == '__main__':
    main()