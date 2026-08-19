import numpy as np
import pandas as pd

def engineer_features(X):
    X = X.copy()

    def ratio(a, b, name):
        if a in X.columns and b in X.columns:
            X[name] = X[a] / X[b].replace(0, np.nan)

    ratio("koi_depth", "koi_duration", "depth_duration_ratio")
    ratio("koi_prad", "koi_srad", "planet_star_size_ratio")
    ratio("koi_period", "koi_duration", "period_duration_ratio")

    if "koi_depth" in X.columns:
        X["log_transit_depth"] = np.log1p(X["koi_depth"].clip(lower=0))

    if "koi_period" in X.columns:
        X["log_period"] = np.log1p(X["koi_period"].clip(lower=0))

    return X.replace([np.inf, -np.inf], np.nan)
