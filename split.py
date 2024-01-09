import numpy as np
from typing import List





# def dirichlet_distribution_noniid_slice(labels: np.ndarray, 
#                                         num_clients: int, 
#                                         alpha: float = 0.5, 
#                                         min_size=10)->List[List[int]]:
#     if len(labels.shape) != 1:
#         raise ValueError('Only support single-label tasks!')
#     num = len(labels)
#     classes = len(np.unique(labels))
#     assert num > num_clients * min_size, f'The number of sample should be greater than {num_clients * min_size}.'
#     idx_slice = [[] for _ in range(num_clients)]
#     size = 0
#     while size < min_size:
#         for k in range(classes):
#             # for label k
#             idx_k = np.where(labels== k)[0]
#             np.random.shuffle(idx_k)
#             prop = np.random.dirichlet(np.repeat(alpha, num_clients))
#             prop = np.array([
#                 p * (len(idx_j) < num / num_clients)
#                 for p, idx_j in zip(prop, idx_slice)
#             ])
#             prop = prop / sum(prop)
#             prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
#             idx_slice = [
#                 idx_j + idx.tolist()
#                 for idx_j, idx in zip(idx_slice, np.array_split(idx_k, prop))
#             ]
#             size = min([len(idx_j) for idx_j in idx_slice])
#     for i in range(num_clients):
#         np.random.shuffle(idx_slice[i])
#     return idx_slice

import numpy as np
from typing import List

def dirichlet_distribution_noniid_slice(labels: np.ndarray, 
                                        num_clients: int, 
                                        alpha: float = 0.5, 
                                        min_size=10) -> List[List[int]]:
    if len(labels.shape) != 1:
        raise ValueError('Only support single-label tasks!')
    num = len(labels)
    classes = len(np.unique(labels))
    assert num > num_clients * min_size, f'The number of sample should be greater than {num_clients * min_size}.'

    # Pre-compute class indices
    class_indices = [np.where(labels == k)[0] for k in range(classes)]

    idx_slice = [[] for _ in range(num_clients)]
    size = 0
    while size < min_size:
        for k in range(classes):
            idx_k = class_indices[k]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, num_clients))
            prop = np.array([p * (len(idx_slice[j]) < num / num_clients) for j, p in enumerate(prop)])
            prop = prop / prop.sum()
            cum_prop = np.cumsum(prop[:-1]) * len(idx_k)
            indices = np.split(idx_k, cum_prop.astype(int))
            idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, indices)]

        size = min(len(idx_j) for idx_j in idx_slice)

    for idx in idx_slice:
        np.random.shuffle(idx)

    return idx_slice


if __name__ == "__main__":
    labels = np.random.choice(10, 1000)
    idx_slice = dirichlet_distribution_noniid_slice(labels, 5)
    print(len(idx_slice))