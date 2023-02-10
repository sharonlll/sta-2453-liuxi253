import pandas as pd
import numpy as np

from model.black_scholes import black_scholes_put
from sklearn.preprocessing import PowerTransformer

# Domain boundaries
S_bound = (0.0, 200.0)
K_bound = (50.0, 150.0)
T_bound = (0.0, 5.0)
r_bound = (0.001, 0.05)
sigma_bound = (0.05, 1.5)


def generate_data(n):
    # TODO: modify me!
    return np.random.rand(n, 5)


def generate_black_scholes_put_data(n):
    x = generate_data(n)

    S_delta = S_bound[1] - S_bound[0]
    K_delta = K_bound[1] - K_bound[0]
    T_delta = T_bound[1] - T_bound[0]
    r_delta = r_bound[1] - r_bound[0]
    sigma_delta = sigma_bound[1] - sigma_bound[0]

    deltas = np.array([S_delta, K_delta, T_delta, r_delta, sigma_delta])
    l_bounds = np.array([S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0]])

    x = x * deltas + l_bounds
    y = black_scholes_put(S=x[:, 0], K=x[:, 1], T=x[:, 2], r=x[:, 3], sigma=x[:, 4]).reshape(-1, 1)

    return np.append(x, y, axis=1)


def main():
    """Generate 1000 Black Scholes put data points and save to 'bs-put-1k.csv'"""
    xy = generate_black_scholes_put_data(1_000)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    power = PowerTransformer()
    power.fit(xy_df)
    trans = power.transform(xy_df)
    trans = pd.DataFrame(trans)
    trans_rename = trans.set_axis(["S", "K", 'T', 'r', 'sigma', 'value'], axis=1, inplace=False)
    trans_rename.to_csv("bs-put-1k.csv")


if __name__ == "__main__":
    main()
