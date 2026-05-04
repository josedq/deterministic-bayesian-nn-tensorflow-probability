"""Loading trained segmented models from the artifacts directory."""

import json
import os
import pickle
import tempfile
import zipfile

import tf_keras as keras
from utils.logger_config import setup_logger
from utils.models import build_bayesian_model, build_deterministic_model
from config.constants import SEGMENTS

logger = setup_logger(__name__)

def patch_input_layer_compat() -> None:
    """Patch tf_keras InputLayer to accept legacy 'batch_shape' key (avoids ValueError on load)."""
    original = keras.layers.InputLayer.from_config
    if getattr(original, '_patched_for_batch_shape', False):
        return

    @classmethod
    def _patched(cls, config):
        config = dict(config)
        if 'batch_shape' in config:
            batch_shape = config.pop('batch_shape')
            config['input_shape'] = batch_shape[1:]
        return cls(**{k: v for k, v in config.items()
                      if k in ('input_shape', 'dtype', 'sparse', 'ragged', 'name')})

    _patched._patched_for_batch_shape = True
    keras.layers.InputLayer.from_config = _patched

def load_det_from_keras_zip(path: str, input_dim: int) -> keras.Model:
    """Load deterministic model weights directly from the .h5 inside a .keras ZIP, bypassing config deserialization."""
    model = build_deterministic_model(input_dim)
    with zipfile.ZipFile(path, 'r') as z:
        h5_files = [f for f in z.namelist() if f.endswith('.h5')]
        if not h5_files:
            raise FileNotFoundError(f"No .h5 weights found inside {path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = z.extract(h5_files[0], tmpdir)
            model.load_weights(extracted)
    return model

def load_segmented_models(artifacts_dir: str, df_processed):
    """Load preprocessor + all segment models from artifacts_dir. Returns (preprocessor, models_det, models_bay)."""
    with open(os.path.join(artifacts_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)

    input_dim = None
    if df_processed is not None:
        try:
            input_dim = preprocessor.transform(df_processed.head(1)).shape[1]
        except Exception as exc:
            logger.warning(f"Could not infer input_dim: {exc}")

    models_det: dict = {}
    models_bay: dict = {}

    for seg in SEGMENTS:
        name = seg['name']

        det_path = os.path.join(artifacts_dir, 'deterministic', f'segment_{name}.keras')
        if os.path.exists(det_path) and input_dim is not None:
            try:
                models_det[name] = load_det_from_keras_zip(det_path, input_dim)
            except Exception as exc:
                logger.warning(f"ZIP extraction failed for {name}: {exc}. Trying load_model.")
                try:
                    models_det[name] = keras.models.load_model(det_path)
                except Exception as exc2:
                    logger.error(f"Could not load {name}: {exc2}")

        config_path  = os.path.join(artifacts_dir, 'bayesian', f'segment_{name}_config.json')
        weights_path = os.path.join(artifacts_dir, 'bayesian', f'segment_{name}_weights.h5')

        if os.path.exists(weights_path):
            n_train = 1000
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)

                if input_dim is None:
                    input_dim = cfg.get('input_dim', None)
                n_train = cfg.get('n_train', n_train)
            else:
                logger.warning(f"No config.json for {name} — using input_dim={input_dim}")

            if input_dim is not None:
                model = build_bayesian_model(input_dim, n_train)
                model.compile(loss='mse')
                model.load_weights(weights_path)
                models_bay[name] = model

    logger.info(f"Loaded {len(models_det)} deterministic, {len(models_bay)} Bayesian models")
    return preprocessor, models_det, models_bay
