"""train_det_benchmark.py — Deterministic training benchmark for EXP-09."""

import os
import threading
import pandas as pd
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split

from utils.logger_config import setup_logger
from config.constants import (
    GLOBAL_SEED, SEGMENTS, TARGET, TARGET_CAP,
    NUMERICAL_STANDARD_FEATURES, NUMERICAL_REVIEW_FEATURES, CATEGORICAL_FEATURES,
    TEST_SPLIT_SIZE,
)
from utils.data_utils import apply_feature_engineering
from utils.models import build_deterministic_model
from utils.preprocessing import build_preprocessor
from utils.monitoring import resource_monitor

logger = setup_logger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'madrid', 'listings.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_EPOCHS     = 1000
BATCH_SIZE     = 32
VAL_SPLIT_SIZE = 0.2
PATIENCE       = 20

tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

def main():
    logger.info("Deterministic training benchmark start")

    df = pd.read_csv(DATA_PATH)
    df = apply_feature_engineering(df)
    x  = df[NUMERICAL_STANDARD_FEATURES + NUMERICAL_REVIEW_FEATURES + CATEGORICAL_FEATURES]
    y  = df[TARGET].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SPLIT_SIZE, random_state=GLOBAL_SEED
    )

    preprocessor = build_preprocessor()
    x_train_proc = preprocessor.fit_transform(x_train).astype('float32')
    x_test_proc  = preprocessor.transform(x_test).astype('float32')
    y_train      = y_train.astype('float32')
    y_test       = y_test.astype('float32')
    input_dim    = x_train_proc.shape[1]

    all_records = []

    for segment in SEGMENTS:
        seg_name   = segment['name']
        train_mask = (y_train > segment['min']) & (y_train <= segment['max'])
        x_seg      = x_train_proc[train_mask]
        y_seg      = y_train[train_mask]

        if len(y_seg) < 50:
            logger.warning(f"Skipping {seg_name}: insufficient samples ({len(y_seg)})")
            continue

        logger.info(f"Segment: {seg_name} | n={len(y_seg)}")

        records    = []
        stop_event = threading.Event()
        monitor_t  = threading.Thread(target=resource_monitor, args=(records, stop_event))

        model = build_deterministic_model(input_dim)
        monitor_t.start()
        model.fit(
            x_seg, y_seg,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT_SIZE,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        )
        stop_event.set()
        monitor_t.join()

        for r in records:
            r['Segment'] = seg_name
        all_records.extend(records)

    output_path = os.path.join(RESULTS_DIR, 'exp09_train_det.csv')
    pd.DataFrame(all_records).to_csv(output_path, index=False)
    logger.info(f"Benchmark results saved: {output_path}")

if __name__ == '__main__':
    main()
