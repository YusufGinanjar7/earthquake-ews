import numpy as np
from scipy import stats

def extract_features(x: np.ndarray) -> dict:
    features = {}

    features["mean"] = x.mean()
    features["std"] = x.std()
    features["max"] = x.max()
    features["min"] = x.min()
    features["abs_mean"] = np.abs(x).mean()
    features["abs_max"] = np.abs(x).max()
    features["kurtosis"] = stats.kurtosis(x)
    features["skew"] = stats.skew(x)

    fft = np.abs(np.fft.fft(x))
    features["fft_mean"] = fft.mean()
    features["fft_std"] = fft.std()
    features["fft_max"] = fft.max()

    return features
