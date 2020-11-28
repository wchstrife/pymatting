import numpy as np


def _cf_laplacian(image, epsilon, r, values, indices, indptr):
    h, w, d = image.shape
    assert d == 3
    size = 2 * r + 1
    window_area = size * size

    for yi in range(h):
        for xi in range(w):
            i = xi + yi * w
            k = i * (4 * r + 1) ** 2
            for yj in range(yi - 2 * r, yi + 2 * r + 1):
                for xj in range(xi - 2 * r, xi + 2 * r + 1):
                    j = xj + yj * w

                    if 0 <= xj < w and 0 <= yj < h:
                        indices[k] = j

                    k += 1

            indptr[i + 1] = k

    # Centered and normalized window colors
    c = np.zeros((2 * r + 1, 2 * r + 1, 3))

    # For each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            # For each color channel
            for dc in range(3):
                # Calculate sum of color channel in window
                s = 0.0
                for dy in range(size):
                    for dx in range(size):
                        s += image[y + dy - r, x + dx - r, dc]

                # Calculate centered window color
                for dy in range(2 * r + 1):
                    for dx in range(2 * r + 1):
                        c[dy, dx, dc] = (
                            image[y + dy - r, x + dx - r, dc] - s / window_area
                        )

            # Calculate covariance matrix over color channels with epsilon regularization
            a00 = epsilon
            a01 = 0.0
            a02 = 0.0
            a11 = epsilon
            a12 = 0.0
            a22 = epsilon

            for dy in range(size):
                for dx in range(size):
                    a00 += c[dy, dx, 0] * c[dy, dx, 0]
                    a01 += c[dy, dx, 0] * c[dy, dx, 1]
                    a02 += c[dy, dx, 0] * c[dy, dx, 2]
                    a11 += c[dy, dx, 1] * c[dy, dx, 1]
                    a12 += c[dy, dx, 1] * c[dy, dx, 2]
                    a22 += c[dy, dx, 2] * c[dy, dx, 2]

            a00 /= window_area
            a01 /= window_area
            a02 /= window_area
            a11 /= window_area
            a12 /= window_area
            a22 /= window_area

            det = (
                a00 * a12 * a12
                + a01 * a01 * a22
                + a02 * a02 * a11
                - a00 * a11 * a22
                - 2 * a01 * a02 * a12
            )

            inv_det = 1.0 / det

            # Calculate inverse covariance matrix
            m00 = (a12 * a12 - a11 * a22) * inv_det
            m01 = (a01 * a22 - a02 * a12) * inv_det
            m02 = (a02 * a11 - a01 * a12) * inv_det
            m11 = (a02 * a02 - a00 * a22) * inv_det
            m12 = (a00 * a12 - a01 * a02) * inv_det
            m22 = (a01 * a01 - a00 * a11) * inv_det

            # For each pair ((xi, yi), (xj, yj)) in a (2 r + 1)x(2 r + 1) window
            for dyi in range(2 * r + 1):
                for dxi in range(2 * r + 1):
                    s = c[dyi, dxi, 0]
                    t = c[dyi, dxi, 1]
                    u = c[dyi, dxi, 2]

                    c0 = m00 * s + m01 * t + m02 * u
                    c1 = m01 * s + m11 * t + m12 * u
                    c2 = m02 * s + m12 * t + m22 * u

                    for dyj in range(2 * r + 1):
                        for dxj in range(2 * r + 1):
                            xi = x + dxi - r
                            yi = y + dyi - r
                            xj = x + dxj - r
                            yj = y + dyj - r

                            i = xi + yi * w
                            j = xj + yj * w

                            # Calculate contribution of pixel pair to L_ij
                            temp = (
                                c0 * c[dyj, dxj, 0]
                                + c1 * c[dyj, dxj, 1]
                                + c2 * c[dyj, dxj, 2]
                            )

                            value = (1.0 if (i == j) else 0.0) - (
                                1 + temp
                            ) / window_area

                            dx = xj - xi + 2 * r
                            dy = yj - yi + 2 * r

                            values[i, dy, dx] += value


exports = {
    "_cf_laplacian": (
        _cf_laplacian,
        "void(f8[:, :, :], f8, i8, f8[:, :, :], i8[:], i8[:])",
    ),
}
