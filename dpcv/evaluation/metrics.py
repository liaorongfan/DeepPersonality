import numpy as np


def concordance_correlation_coefficient(y_true, y_pred):
    """ Concordance correlation coefficient.

    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    # from sklearn.metrics import concordance_correlation_coefficient # note may not supported now
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def compute_pcc(outputs, labels):
    from scipy.stats import pearsonr
    keys = ['O', 'C', 'E', 'A', 'N']
    pcc_dic = {}
    pcc_sum = 0
    for i, key in enumerate(keys):
        res = pearsonr(outputs[:, i], labels[:, i])
        # res[1] records p-value
        pcc_dic[key] = np.round(res[0], 4)
        pcc_sum += res[0]
    mean = np.round((pcc_sum / 5), 4)
    return pcc_dic, mean


def compute_ccc(outputs, labels):
    keys = ['O', 'C', 'E', 'A', 'N']
    ccc_dic = {}
    ccc_sum = 0
    for i, key in enumerate(keys):
        res = concordance_correlation_coefficient(labels[:, i], outputs[:, i])
        ccc_dic[key] = np.round(res, 4)
        ccc_sum += res
    mean = np.round((ccc_sum / 5), 4)
    return ccc_dic, mean


if __name__ == "__main__":
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    b = concordance_correlation_coefficient(y_true, y_pred)
    print(b)
