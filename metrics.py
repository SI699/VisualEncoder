import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def ssim_score(y_pred, y_target):
    return np.mean([
        ssim(y_pred[i], y_target[i], channel_axis=0, data_range=1.0)
        for i in range(y_pred.shape[0])
    ])


def psnr_score(y_pred, y_target):
    return np.mean([
        psnr(y_pred[i], y_target[i], data_range=1.0)
        for i in range(y_pred.shape[0])
    ])
