from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN


def parallel_dbscan(points, eps, min_samples, n_jobs=-1):
    """Parallel DBSCAN clustering"""

    def _single_dbscan(pc):
        return DBSCAN(eps=eps, min_samples=min_samples).fit(pc).labels_

    if len(points.shape) == 2:  # single sample
        return _single_dbscan(points)
    else:  # batch processing
        return Parallel(n_jobs=n_jobs)(
            delayed(_single_dbscan)(pc) for pc in points)