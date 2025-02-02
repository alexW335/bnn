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
    p_logit   = y_pred[..., 0:1]   # (batch, 1)
    lambda_raw= y_pred[..., 1:2]   # (batch, 1)
    mu_raw    = y_pred[..., 2:4]   # (batch, 2)
    scale_raw = y_pred[..., 4:6]   # (batch, 2)
    corr_raw  = y_pred[..., 6:7]   # (batch, 1)
    
    # transform to valid parameters
    p       = tf.sigmoid(p_logit)
    lambda_ = tf.nn.softplus(lambda_raw)
    count   = p * lambda_  # expected number of shipments
    
    mu      = tf.nn.softplus(mu_raw)  # per-shipment mean (ensuring positivity)
    mean_total = count * mu          # predicted total (for weight and volume)
    
    sigma   = tf.nn.softplus(scale_raw)  # shipment-level std dev's
    rho     = tf.tanh(corr_raw)          # correlation in (-1,1)
    
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
        
    The final layer outputs 7 numbers per sample.
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
        prior_fn    = tfp.layers.default_multivariate_normal_fn
        posterior_fn= tfp.layers.default_mean_field_normal_fn()
        
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        
        # shared base: extract features
        for units in self.hidden_units_shared:
            x = tfp.layers.DenseFlipout(
                units, activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(x)
        
        # head: further process for likelihood parameters
        for units in self.hidden_units_head:
            x = tfp.layers.DenseFlipout(
                units, activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(x)
        
        # Final output: 7 parameters
        outputs = tfp.layers.DenseFlipout(
            7, activation=None,
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(x)
        
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
            Input features (e.g. one-hot encoded product flags) with shape (n_samples, input_dim).
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
            An array of shape (batch_size, 2) with predictive samples for total weight and volume.
        """
        return self.model(X, training=True).numpy().astype(np.float64)
