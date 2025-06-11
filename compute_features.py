import numpy as np
import torch

def compute_all_features(x: np.ndarray,
                        resampled_length: int, 
                        descriptors: list):
    features = {}
    for desc in descriptors:
        features[desc] = np.array(torch.rand(resampled_length))
    return features

