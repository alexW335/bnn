import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# custom negative log-likelihood loss
def custom_neg_log_likelihood(y_true, y_pred, n_max=10, eps=1e-3):
    """
    Custom negative log likelihood that fully incorporates:
      1. a Bernoulli event with probability p = sigmoid(p_logit)
      2. a Poisson-distributed shipment count (with rate lambda = softplus(lambda_raw))
      3. a Normal shipment contribution (with mean mu = softplus(mu_raw)
         and covariance Sigma built from softplus(scale_raw) and tanh(corr_raw)).
    
    y_pred has shape (batch, 7) with:
      - index 0: p_logit
      - index 1: lambda_raw
      - indices 2-3: mu_raw (for weight and volume)
      - indices 4-5: scale_raw (for covariance; will be transformed via softplus)
      - index 6: corr_raw (for correlation; will be transformed via tanh)
    
    The generative process is:
      - With probability (1-p) the shipment does not occur (n=0).
      - With probability p, n is drawn from a Poisson with rate lambda.
      - Given n shipments, the observed total is the sum of n shipment contributions.
        For n >= 1, sum_{i=1}^n X_i ~ MVN(n * mu, n * Sigma).
      - To avoid degenerate covariance when n=0, we add a small fixed noise covariance
        (eps * I) in all cases.
    
    We marginalize over n = 0, 1, ... , n_max.
    """
    # Unpack predicted parameters.
    # p_logit and lambda_raw are scalars per sample.
    p_logit    = y_pred[..., 0:1]    # shape (batch, 1)
    lambda_raw = y_pred[..., 1:2]    # shape (batch, 1)
    mu_raw     = y_pred[..., 2:4]    # shape (batch, 2)
    scale_raw  = y_pred[..., 4:6]    # shape (batch, 2)
    corr_raw   = y_pred[..., 6:7]    # shape (batch, 1)

    # Transform to valid parameters.
    p       = tf.sigmoid(p_logit)              # (batch, 1) in (0,1)
    lam     = tf.nn.softplus(lambda_raw)         # (batch, 1) > 0
    mu      = tf.nn.softplus(mu_raw)             # (batch, 2) > 0
    sigma   = tf.nn.softplus(scale_raw)          # (batch, 2) > 0
    rho     = tf.tanh(corr_raw)                  # (batch, 1) in (-1,1)

    # Build covariance matrix Sigma for one shipment.
    sigma1 = sigma[..., 0:1]                     # (batch, 1)
    sigma2 = sigma[..., 1:2]                     # (batch, 1)
    cov11  = sigma1 ** 2                         # (batch, 1)
    cov22  = sigma2 ** 2                         # (batch, 1)
    cov12  = sigma1 * sigma2 * rho                # (batch, 1)
    # Each covariance matrix is 2x2.
    Sigma = tf.stack([
                tf.concat([cov11, cov12], axis=-1),
                tf.concat([cov12, cov22], axis=-1)
            ], axis=-2)                       # shape (batch, 2, 2)
    
    batch_size = tf.shape(y_true)[0]
    
    # Define fixed observation noise covariance.
    noise_cov = eps * tf.eye(2, dtype=tf.float32)   # shape (2,2)
    # Expand noise_cov to (batch, 1, 2, 2).
    noise_cov = tf.broadcast_to(noise_cov, [batch_size, 1, 2, 2])
    
    # Prepare a vector of shipment counts: n = 0, 1, ..., n_max.
    n_values = tf.cast(tf.range(0, n_max+1), dtype=tf.float32)  # shape (n_max+1,)
    # Expand to (1, n_max+1) so that it can broadcast with (batch,1).
    n_values_2d = tf.reshape(n_values, [1, n_max+1])
    
    # Compute mixture log weights.
    # For n == 0:
    #   logw0 = log((1-p) + p * exp(-lam))
    # For n >= 1:
    #   logwn = log(p) - lam + n*log(lam) - log(n!)
    logw = tf.where(
        tf.equal(n_values_2d, 0.),
        tf.math.log((1 - p) + p * tf.exp(-lam)),
        tf.math.log(p) - lam + n_values_2d * tf.math.log(lam + 1e-8) - tf.math.lgamma(n_values_2d + 1)
    )  # shape (batch, n_max+1)
    
    # Now, for each candidate n, compute the distribution of y.
    # For each sample and each n:
    #   if n == 0: mean = [0,0]
    #   if n >= 1: mean = n * mu.
    # We create a tensor of means with shape (batch, n_max+1, 2).
    # First, expand n_values to (batch, n_max+1, 1):
    n_expanded = tf.reshape(n_values, [1, n_max+1, 1])
    n_expanded = tf.cast(n_expanded, dtype=tf.float32)
    # For n==0, we want zero; for n>=1, n*mu.

    # Expand mu to shape (batch, 1, 2) so multiplication broadcasts correctly.
    # works because when n==0, product is 0.
    # Alternatively, one could write: mean_n = tf.where(n_expanded==0, tf.zeros_like(mu), n_expanded * mu)
    mu_expanded = tf.expand_dims(mu, axis=1)   # shape: (batch, 1, 2)
    
    # Compute the mean for each candidate n.
    mean_n = n_expanded * mu_expanded          # shape: (batch, n_max+1, 2)
    
    # For covariance: for each candidate n, covariance = n * Sigma + noise_cov.
    # First, expand Sigma to shape (batch, 1, 2, 2):
    Sigma_expanded = tf.expand_dims(Sigma, axis=1)  # (batch, 1, 2,2)
    # Expand n_values to (1, n_max+1, 1, 1):
    n_values_4d = tf.reshape(n_values, [1, n_max+1, 1, 1])
    n_values_4d = tf.cast(n_values_4d, dtype=tf.float32)
    cov_n = n_values_4d * Sigma_expanded + noise_cov  # shape (batch, n_max+1, 2, 2)
    
    # Create a Multivariate Normal distribution for each candidate n.
    # We need to compute the log_prob of y_true under each.
    # Expand y_true from (batch, 2) to (batch, 1, 2) to broadcast over n.
    y_true_expanded = tf.expand_dims(y_true, axis=1)  # (batch, 1, 2)
    
    # tfd.MultivariateNormalFullCovariance accepts batch parameters.
    # Set up a distribution with loc of shape (batch, n_max+1, 2) and
    # covariance_matrix of shape (batch, n_max+1, 2, 2).
    mvn = tfd.MultivariateNormalFullCovariance(loc=mean_n, covariance_matrix=cov_n)
    logp_y = mvn.log_prob(y_true_expanded)  # shape (batch, n_max+1)
    
    # The total log likelihood for each sample is the log-sum-exp over n of (log_weight + logp_y)
    log_mix = tf.reduce_logsumexp(logw + logp_y, axis=1)  # shape (batch,)
    
    nll = -tf.reduce_mean(log_mix)
    return nll


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

