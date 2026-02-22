# risk_models.py

import numpy as np

def monte_carlo_simulation(S0, mu, sigma, days=252, simulations=1000):

    dt = 1/252
    price_paths = np.zeros((days, simulations))
    price_paths[0] = S0

    for t in range(1, days):
        Z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * Z
        )

    return price_paths
