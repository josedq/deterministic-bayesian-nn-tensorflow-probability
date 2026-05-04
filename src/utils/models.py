"""Model builder functions used across experiment and benchmark scripts."""

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from config.constants import ARCHITECTURE

tfd  = tfp.distributions
tfpl = tfp.layers

def build_deterministic_model(input_dim: int) -> keras.Model:
    """Dense [128, 64, 32] + Dropout(0.2), huber loss."""
    layers = [keras.layers.Input(shape=(input_dim,))]
    for units in ARCHITECTURE['hidden_layers']:
        layers.append(keras.layers.Dense(units, activation=ARCHITECTURE['activation']))
        layers.append(keras.layers.Dropout(ARCHITECTURE['dropout_rate']))
    layers.append(keras.layers.Dense(1))

    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='huber')
    return model

def build_bayesian_model(input_dim: int, n_train: int) -> keras.Model:
    """DenseFlipout [128, 64, 32, 2] with Normal output; KL weight = 1/n_train."""
    kl_weight = 1.0 / n_train

    def divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) * kl_weight

    def normal_fn(t):
        return tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]),
        )

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        tfpl.DenseFlipout(128, activation='relu', kernel_divergence_fn=divergence_fn),
        tfpl.DenseFlipout(64,  activation='relu', kernel_divergence_fn=divergence_fn),
        tfpl.DenseFlipout(32,  activation='relu', kernel_divergence_fn=divergence_fn),
        tfpl.DenseFlipout(2,   activation=None,   kernel_divergence_fn=divergence_fn),
        tfpl.DistributionLambda(normal_fn),
    ])

    model.compile(optimizer='adam', loss=lambda y, p: -p.log_prob(y))
    return model
