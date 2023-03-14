import numpy as np
from skimage.metrics import structural_similarity as ssim


def ssim_score(y_pred, y_target):
    return np.mean([
        ssim(y_pred[i], y_target[i], channel_axis=0)
        for i in range(y_pred.shape[0])
    ])
