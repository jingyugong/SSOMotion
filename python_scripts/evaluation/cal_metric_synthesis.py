import numpy as np
from scipy import linalg
import tqdm
import pickle

def calculate_activation_statistics(activations):
    """
    SRC:https://github.com/GuyTevet/motion-diffusion-model/blob/ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778/data_loaders/humanml/utils/metrics.py
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calc_fid_for_prediction(pred_result_paths):
    pred_features = []
    gt_features = []
    for pred_result_path in tqdm.tqdm(pred_result_paths):
        gt_result_path = pred_result_path.replace('pred', 'gt')
        file_fmt = pred_result_path.split('.')[-1]
        if file_fmt == 'pkl':
            with open(pred_result_path, 'rb') as f:
                pred_result = pickle.load(f)
            with open(gt_result_path, 'rb') as f:
                gt_result = pickle.load(f)
            pred_motion = pred_result['motion']
            gt_motion = gt_result['motion']

            pred_motion_feature = pred_motion['smplx_params'].reshape(1,30*69)
            gt_motion_feature = gt_motion['smplx_params'].reshape(1,30*69)
        elif file_fmt == 'npy':
            pred_motion_feature = np.load(pred_result_path).reshape(1,-1)
            gt_motion_feature = np.load(gt_result_path).reshape(1,-1)

        pred_features.append(pred_motion_feature)
        gt_features.append(gt_motion_feature)
    pred_features = np.concatenate(pred_features, axis=0)
    gt_features = np.concatenate(gt_features, axis=0)

    gt_mean, gt_cov = calculate_activation_statistics(gt_features)
    gt_var = gt_cov.diagonal()
    pred_features = (pred_features - gt_mean) / np.sqrt(gt_var)
    gt_features = (gt_features - gt_mean) / np.sqrt(gt_var)

    pred_mean, pred_cov = calculate_activation_statistics(pred_features)
    gt_mean, gt_cov = calculate_activation_statistics(gt_features)
    fid = calculate_frechet_distance(pred_mean, pred_cov, gt_mean, gt_cov)
    return fid

if __name__ == "__main__":
    #glob_pattern = f'./save/results_for_eval/cmdm_action2motion_qkv/humanise/sample*/rep*/pred/results.pkl'
    #pred_result_paths = sorted(glob.glob(glob_pattern))
    pred_result_paths = [f'./save/results_for_eval/cmdm_action2motion_qkv/humanise_in_tmrfeats/pred/sample{i}.npy' for i in range(0, 21919)] #total 21919
    n_frames = 30
    fid = calc_fid_for_prediction(pred_result_paths)
    print(fid)
