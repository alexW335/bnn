from .schemas import SimulatedDataSchema
import numpy as np
import pandas as pd

class DataGenerator():
    def __init__(
            self, 
            p_bern: np.ndarray, 
            lambda_s: np.ndarray, 
            mu: np.ndarray, 
            cov: np.ndarray, 
            rng: np.random.Generator=np.random.default_rng()
        ):

        if p_bern.shape[0] != mu.shape[0] or p_bern.shape[0] != cov.shape[0] or p_bern.shape[0] != lambda_s.shape[0]:
            raise ValueError("The first dimension of p_bern, lambda_s, mu, and cov must match")
       
        if mu.shape[1] != 2:
            raise ValueError("The second dimension of mu must be 2")
        
        if cov.shape[1] != 2 or cov.shape[2] != 2:
            raise ValueError("Each element of cov along the first dimension must be a 2x2 matrix")
         
        self.p_bern = p_bern
        self.lambda_s = lambda_s
        self.mu = mu
        self.cov = cov
        self.rng = rng
        
    def generate(self, samples_per_product: int = 100, n_attempts: int = 1000) -> SimulatedDataSchema:
        res = []
        # iterate through the number of products
        for p in np.arange(self.p_bern.shape[-1]):
            # start by pulling samples from the bernoulli distribution
            bern = self.rng.uniform(low=0, high=1, size=samples_per_product)
            bern = np.where(bern <= self.p_bern[p], 0, 1)

            # now pull a bunch of non-zero samples from the poisson distribution
            poiss = bern.copy()
            for idx in np.where(bern == 1)[0]:
                candidate = 0
                draw_count = 0
                while candidate == 0:
                    candidate = self.rng.poisson(lam=self.lambda_s[p])
                    draw_count += 1

                    if draw_count >= n_attempts:
                        raise RuntimeError(f'Non non-zero draws after {n_attempts} attempts')
                    
                poiss[idx] = candidate

            mvn = self.rng.multivariate_normal(mean=self.mu[p], cov=self.cov[p], size=samples_per_product)

            # stack the poisson samples to match the shape of the mvn and do an element-wise multiplication
            samples = np.multiply(mvn, np.hstack((poiss[:,np.newaxis], poiss[:,np.newaxis])))

            # put together a product column
            prod = p*np.ones(shape=(samples_per_product,1))

            t_ = np.hstack((prod, samples))

            res.append(pd.DataFrame({'product':t_[:,0], 'weight':t_[:,1], 'volume':t_[:,2]}))

        df = SimulatedDataSchema(pd.concat(res))
        
        return df