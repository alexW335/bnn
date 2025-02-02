import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# custom negative log-likelihood loss
def custom_neg_log_likelihood(y_true, y_pred):
    """
    y_pred has shape (batch, 7) with:
      - [0]: p_logit, transformed via sigmoid -> p in (0,1)
      - [1]: lambda_raw, transformed via softplus -> lambda > 0
      - [2:4]: mu_raw for weight and volume, transformed via softplus
      - [4:6]: scale_raw for covariance, transformed via softplus to get positive scales
      - [6]: corr_raw, transformed via tanh -> correlation in (-1,1)
      
    The predicted mean for the totals is:
        mean_total = (sigmoid(p_logit) * softplus(lambda_raw)) * softplus(mu_raw)
    The covariance is built as:
        cov = [[sigma1^2, sigma1*sigma2*rho],
               [sigma1*sigma2*rho, sigma2^2]]
    """
    # unpack predictions
    p_logit    = y_pred[..., 0:1]   # (batch, 1)
    lambda_raw = y_pred[..., 1:2]   # (batch, 1)
    mu_raw     = y_pred[..., 2:4]   # (batch, 2)
    scale_raw  = y_pred[..., 4:6]   # (batch, 2)
    corr_raw   = y_pred[..., 6:7]   # (batch, 1)
    
    # transform to valid parameters
    p        = tf.sigmoid(p_logit)
    lambda_  = tf.nn.softplus(lambda_raw)
    count    = p * lambda_  # expected number of shipments
    
    mu       = tf.nn.softplus(mu_raw)  # per-shipment mean (ensuring positivity)
    mean_total = count * mu            # predicted totals (for weight and volume)
    
    sigma    = tf.nn.softplus(scale_raw)  # shipment-level std dev's
    rho      = tf.tanh(corr_raw)          # correlation in (-1,1)
    
    # build covariance matrix for each sample; shapes: sigma[...,0:1] is (batch,1)
    sigma1 = sigma[..., 0:1]
    sigma2 = sigma[..., 1:2]
    cov11  = sigma1 ** 2
    cov22  = sigma2 ** 2
    cov12  = sigma1 * sigma2 * rho
    cov21  = cov12
    
    # combine into a covariance matrix of shape (batch, 2, 2)
    cov_matrix = tf.stack([
        tf.concat([cov11, cov12], axis=-1),
        tf.concat([cov21, cov22], axis=-1)
    ], axis=-2)
    
    # define multivariate normal likelihood
    mvn = tfd.MultivariateNormalFullCovariance(loc=mean_total, covariance_matrix=cov_matrix)
    
    # negative log likelihood (nll)
    nll = -mvn.log_prob(y_true)
    return tf.reduce_mean(nll)


class BayesianShipmentModel:
    """
    A Bayesian neural network that predicts parameters for a custom likelihood:
      - shipment occurrence (via a Bernoulli head)
      - shipment count (via a Poisson head)
      - shipment value (via a Normal head whose parameters include a covariance
        matrix parameterized via two scales and a correlation)
        
    The final output per sample is a 7-dimensional vector:
      [p_logit, lambda_raw, mu_raw (2 values), scale_raw (2 values), corr_raw]
    """
    def __init__(self, input_dim: int,
                 hidden_units_shared: list = [32, 16],
                 hidden_units_head: list = [16],
                 kl_weight: float = 1e-3,
                 learning_rate: float = 1e-3):
        self.input_dim = input_dim
        self.hidden_units_shared = hidden_units_shared
        self.hidden_units_head = hidden_units_head
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self._build_model()
    
    def _kl_divergence_fn(self, q, p, _):
        return tfd.kl_divergence(q, p) * self.kl_weight
    
    def _build_model(self):
        prior_fn     = tfp.layers.default_multivariate_normal_fn
        posterior_fn = tfp.layers.default_mean_field_normal_fn()
        
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # Shared base: extract features common to all heads.
        x_shared = inputs
        for units in self.hidden_units_shared:
            x_shared = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(x_shared)
        
        # --- Bernoulli head (shipment occurrence) ---
        bernoulli = x_shared
        for units in self.hidden_units_head:
            bernoulli = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(bernoulli)
        # One output node (logit) for p.
        p_logit = tfp.layers.DenseFlipout(
            1,
            activation=None,
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(bernoulli)
        
        # --- Poisson head (shipment count) ---
        poisson = x_shared
        for units in self.hidden_units_head:
            poisson = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(poisson)
        # One output node for the Poisson rate.
        lambda_raw = tfp.layers.DenseFlipout(
            1,
            activation=None,
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(poisson)
        
        # --- Normal head (shipment value: mean, scale, and correlation) ---
        normal = x_shared
        for units in self.hidden_units_head:
            normal = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(normal)
        # This head outputs 5 values:
        #   - 2 for mu_raw (per-shipment mean for weight and volume),
        #   - 2 for scale_raw (to be transformed into standard deviations),
        #   - 1 for corr_raw (to be transformed into a correlation via tanh).
        normal_params = tfp.layers.DenseFlipout(
            5,
            activation=None,
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(normal)
        # Split the 5 outputs.
        mu_raw    = normal_params[..., 0:2]   # indices 0,1
        scale_raw = normal_params[..., 2:4]   # indices 2,3
        corr_raw  = normal_params[..., 4:5]   # index 4
        
        # --- Concatenate all heads ---
        # Order: p_logit (1), lambda_raw (1), mu_raw (2), scale_raw (2), corr_raw (1)
        outputs = tf.keras.layers.Concatenate(axis=-1)([p_logit, lambda_raw, mu_raw, scale_raw, corr_raw])
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=custom_neg_log_likelihood
        )
    
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        X : array-like
            Input features (e.g., one-hot encoded product flags) with shape (n_samples, input_dim).
        y : array-like
            Observed totals with shape (n_samples, 2), corresponding to [total_weight, total_volume].
        kwargs : dict
            Additional keyword arguments passed to tf.keras.Model.fit.
        """
        self.model.fit(X, y, **kwargs)
    
    def predict(self, X):
        """
        Generate Monte Carlo samples from the predictive distribution.
        
        Because the DenseFlipout layers sample new weights when `training=True` is passed,
        calling the model repeatedly yields different outputs. These samples represent
        draws from the predictive distribution.
        
        Parameters
        ----------
        X : array-like
            Input features with shape (n_samples, input_dim).
        
        Returns
        -------
        samples : np.ndarray
            An array of shape (batch_size, 7) with the predictive parameters.
        """
        return self.model(X, training=True).numpy().astype(np.float64)

