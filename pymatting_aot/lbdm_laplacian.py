import numpy as np
from numba import njit


@njit("f8[:, :](f8[:, :], f8)")
def calculate_kernel_matrix(X, v):
    n, m = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.exp(-np.sqrt(v) * np.sum(np.square(X[i] - X[j])))
    return K


def _lbdm_laplacian(image, epsilon, r):
    h, w = image.shape[:2]
    n = h * w

    area = (2 * r + 1) ** 2

    indices = np.arange(n).reshape(h, w)

    values = np.zeros((n, area ** 2))
    i_inds = np.zeros((n, area ** 2), dtype=np.int32)
    j_inds = np.zeros((n, area ** 2), dtype=np.int32)

    # gray = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3.0
    # v = np.std(gray)

    for y in range(r, h - r):
        for x in range(r, w - r):
            i = x + y * w

            X = np.ones((area, 3 + 1))

            k = 0
            for y2 in range(y - r, y + r + 1):
                for x2 in range(x - r, x + r + 1):
                    for c in range(3):
                        X[k, c] = image[y2, x2, c]
                    k += 1

            window_indices = indices[y - r : y + r + 1, x - r : x + r + 1].flatten()

            # does not produce better results than no kernel
            # K = calculate_kernel_matrix(X, v)

            K = np.dot(X, X.T)

            f = np.linalg.solve(K + epsilon * np.eye(area), K)

            tmp2 = np.eye(f.shape[0]) - f
            tmp3 = tmp2.dot(tmp2.T)

            for k in range(area):
                i_inds[i, k::area] = window_indices
                j_inds[i, k * area : k * area + area] = window_indices
            values[i] = tmp3.ravel()

    return values.ravel(), i_inds.ravel(), j_inds.ravel()


exports = {
    "_lbdm_laplacian": (
        _lbdm_laplacian,
        "Tuple((f8[:], i4[:], i4[:]))(f8[:, :, :], f8, i4)",
    ),
}
