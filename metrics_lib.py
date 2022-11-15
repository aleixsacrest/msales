import numpy as np
import pandas as pd


Q_VALUES = np.concatenate((np.arange(0, 85, 5), np.arange(82, 102, 2)))


def get_rmse_score(pred, y):
    return np.sqrt(((pred - y) ** 2).mean()).round(3)


def get_correlation_score(pred, y):
    corr = pd.DataFrame({'pred': pred, 'obs': y}).corr().iloc[0, 1].round(3)
    return corr


def get_norm_rmse_score(pred, y):
    std = y.std()
    rmse = np.sqrt(((pred - y) ** 2).mean())
    nrmse = rmse/std
    return nrmse.round(3)


def get_quantile_values(pred, y, qValues=Q_VALUES):
    qPred = list(np.percentile(pred, qValues).round(3))
    qY = list(np.percentile(y, qValues).round(3))
    return qPred, qY, list(qValues)


def get_metrics_dict(pred, y):
    rmse, rmseScaled = get_rmse_score(pred, y), get_norm_rmse_score(pred, y)
    corr = get_correlation_score(pred, y)
    qPred, qY, qValues = get_quantile_values(pred, y)
    return {'rmse': rmse, 'rmseScaled': rmseScaled, 'corr': corr,
            'qPred': qPred, 'qObs': qY, 'qValues': qValues}
