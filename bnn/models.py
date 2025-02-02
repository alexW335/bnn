import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

class BayesianShipmentModel:
    """
    A Bayesian neural network model for day/product shipment totals.
    
    Each input is a one-hot encoded vector (or any other features) that identifies a product (or other groups).
    The model is designed to reflect an underlying generative process:
      1. A Bernoulli (parameterized by p) governs whether any shipments occur.
      2. Conditional on shipments occurring, the number of shipments follows a Poisson distribution with mean λ.
      3. Each shipment's weight and volume is drawn from a multivariate normal with mean μ = [μ_weight, μ_volume].
    
    The expected (observed) total weight and volume are given by:
    
         E[totals] = (\sigma(p) * softplus(λ)) * softplus(μ)
    
    (Here, \sigma is the sigmoid function and softplus is used to ensure positivity.)
    
    The network is built using TFP's DenseFlipout layers so that we learn distributions over weights.
    Monte Carlo prediction (by sampling the network's weights) then provides a predictive distribution.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units_shared: list = [32, 16],
        hidden_units_head: list = [16],
        kl_weight: float = 1e-3,
        learning_rate: float = 1e-3
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the one-hot encoded (or otherwise encoded) input.
        hidden_units_shared : list, optional
            List of hidden units for the shared feature extractor.
        hidden_units_head : list, optional
            List of hidden units for each of the three heads.
        kl_weight : float, optional
            Scaling factor for the KL divergence regularization.
        learning_rate : float, optional
            Learning rate for the optimizer.
        """
        self.input_dim = input_dim
        self.hidden_units_shared = hidden_units_shared
        self.hidden_units_head = hidden_units_head
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self._build_model()
    
    def _kl_divergence_fn(self, q, p, _):
        """Scale the KL divergence (which TFP will add to the loss) by a factor."""
        return tfd.kl_divergence(q, p) * self.kl_weight
    
    def _build_model(self):
        prior_fn = tfp.layers.default_multivariate_normal_fn
        posterior_fn = tfp.layers.default_mean_field_normal_fn()
        
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # Shared base: extract features common to all heads.
        x = inputs
        for units in self.hidden_units_shared:
            x = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(x)
        
        # --- Bernoulli head (shipment occurrence) ---
        bernoulli = x
        for units in self.hidden_units_head:
            bernoulli = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(bernoulli)
        # One output node (logit) transformed by a sigmoid.
        p_logit = tfp.layers.DenseFlipout(
            1,
            activation=None,
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(bernoulli)
        p = tf.keras.layers.Activation('sigmoid', name='p')(p_logit)
        
        # --- Poisson head (conditional shipment count) ---
        poisson = x
        for units in self.hidden_units_head:
            poisson = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(poisson)
        # One output node transformed with softplus to ensure positivity.
        lambda_out = tfp.layers.DenseFlipout(
            1,
            activation='softplus',
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(poisson)
        
        # --- Normal head (average shipment weight & volume) ---
        normal = x
        for units in self.hidden_units_head:
            normal = tfp.layers.DenseFlipout(
                units,
                activation='relu',
                kernel_prior_fn=prior_fn,
                kernel_posterior_fn=posterior_fn,
                kernel_divergence_fn=self._kl_divergence_fn
            )(normal)
        # Two output nodes: one for weight and one for volume.
        # We use softplus to ensure these averages are positive.
        mean_out = tfp.layers.DenseFlipout(
            2,
            activation='softplus',
            kernel_prior_fn=prior_fn,
            kernel_posterior_fn=posterior_fn,
            kernel_divergence_fn=self._kl_divergence_fn
        )(normal)
        
        # --- Combine heads to form the predicted totals ---
        # Expected number of shipments = sigmoid(p) * softplus(λ)
        expected_shipments = tf.keras.layers.Multiply(name='expected_shipments')([p, lambda_out])
        # Predicted total (for weight and volume) = (expected shipments) * (average shipment weight/volume)
        predicted_totals = tf.keras.layers.Multiply(name='predicted_totals')([expected_shipments, mean_out])
        # predicted_totals will have shape (batch_size, 2): [total_weight, total_volume]
        
        self.model = tf.keras.Model(inputs=inputs, outputs=predicted_totals)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
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
    
    def predict(self, X, num_samples: int = 100):
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
