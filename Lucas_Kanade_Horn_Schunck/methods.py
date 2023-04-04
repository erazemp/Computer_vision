import math

import numpy as np
import cv2
import ex1_utils as utils
import time


def spatial_derivatives_calculate(im1, im2, sigma):
    ix1, iy1 = utils.gaussderiv(im1, sigma)
    ix2, iy2 = utils.gaussderiv(im2, sigma)

    it = utils.gausssmooth(im1 - im2, sigma)

    return ix1, iy1, ix2, iy2, it

def average_spatial_derivative_calculate(ix1, ix2, iy1, iy2):
    ix = 0.5 * (np.array(ix1) + np.array(ix2))
    iy = 0.5 * (np.array(iy1) + np.array(iy2))

    return ix, iy

def calculate_u_v_lucas_kanade(D, ix, iy, it, kernel):
    D[D == 0] = 2.2204e-16
    u = ((cv2.filter2D((iy ** 2), -1, kernel)) * (cv2.filter2D((ix * it), -1, kernel)) - \
           (cv2.filter2D((ix * iy), -1, kernel)) * cv2.filter2D((iy * it), -1, kernel))
    u = u / D
    v = ((cv2.filter2D((ix ** 2), -1, kernel)) * cv2.filter2D((iy * it), -1, kernel) - \
           (cv2.filter2D((ix * iy), -1, kernel)) * (cv2.filter2D((ix * it), -1, kernel)))
    v = v / D

    return -u, -v


def calculate_determinant(ix, iy, kernel):
    D = (cv2.filter2D(ix ** 2, -1, kernel)) * (cv2.filter2D(iy ** 2, -1, kernel)) - \
        (cv2.filter2D(((ix * iy) ** 2), -1, kernel))
    return D

def lucas_kanade(im1, im2, N):
    # im1 − first image mat rix ( grayscale)
    # im2 − sec ond image mat rix (grayscale)
    # n − size of the neighborhood (N x N)
    # TODO : the algorithm
    kernel = np.ones((N, N))

    ix_1, iy_1, ix_2, iy_2, it = spatial_derivatives_calculate(im1, im2, 1)
    # spatial derivatives in both directions
    ix, iy = average_spatial_derivative_calculate(ix_1, ix_2, iy_1, iy_2)

    # determinant of the covariance matrix
    D = calculate_determinant(ix, iy, kernel)

    # calculating the vectors
    ##if D == 0:
    ##   D = 2.2204e-16

    u, v = calculate_u_v_lucas_kanade(D, ix, iy, it, kernel)

    return u, v


def lucas_kanade_improved(im1, im2, N):
    s = time.time()
    kernel = np.ones((N, N))
    ix_1, iy_1, ix_2, iy_2, it = spatial_derivatives_calculate(im1, im2, 1)
    # spatial derivatives in both directions
    ix, iy = average_spatial_derivative_calculate(ix_1, ix_2, iy_1, iy_2)
    A11 = cv2.filter2D(ix ** 2, -1, kernel)
    A12 = cv2.filter2D(ix * iy, -1, kernel)
    A21 = cv2.filter2D(iy * ix, -1, kernel)
    A22 = cv2.filter2D(iy ** 2, -1, kernel)
    D = calculate_determinant(ix, iy, kernel)
    u, v = calculate_u_v_lucas_kanade(D, ix, iy, it, kernel)
    ig_i = []
    #print("Lucas-Kanade improved runtime: {:.5f} seconds".format(time.time() - s))
    for i in range(len(A11)):
        for j in range(len(A11[i])):
            A_tA = np.array([[A11[i][j], A12[i][j]], [A21[i][j], A22[i][j]]])
            val, vec = np.linalg.eig(A_tA)
            # print("Lucas-Kanade improved runtime: {:.5f} seconds".format(time.time() - s))
            if val[0] < 0.0001 or val[1] < 0.0001 or val[0] / val[1] > 50 or val[1] / val[0] > 50:
                ig_i.append((i, j))
    #print("Lucas-Kanade improved runtime: {:.5f} seconds".format(time.time() - s))

    for el in ig_i:
        u[el[0]][el[1]] = 0
        v[el[0]][el[1]] = 0
    #print("Lucas-Kanade improved runtime: {:.5f} seconds".format(time.time() - s))
    return u, v


def horn_schunck(im1, im2, n_iters, lmbd):
    ix_1, iy_1, ix_2, iy_2, it = spatial_derivatives_calculate(im1, im2, 1)

    # spatial derivatives in both directions
    ix, iy = average_spatial_derivative_calculate(ix_1, ix_2, iy_1, iy_2)

    Ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    u = np.zeros((im1.shape[0], im1.shape[1]))
    v = np.zeros((im1.shape[0], im1.shape[1]))

    while n_iters != 0:
        ua = cv2.filter2D(u, -1, Ld)
        va = cv2.filter2D(v, -1, Ld)
        P = ix * ua + iy * va + it
        D = lmbd + ix ** 2 + iy ** 2
        u = ua - (ix * (P / D))
        v = va - (iy * (P / D))
        n_iters = n_iters - 1

    return u, v

def horn_schunck_improved(im1, im2, n_iters, lmbd, n):
    kernel = np.ones((n, n))
    Ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    ix_1, iy_1, ix_2, iy_2, it = spatial_derivatives_calculate(im1, im2, 1)
    ix, iy = average_spatial_derivative_calculate(ix_1, ix_2, iy_1, iy_2)
    D = calculate_determinant(ix, iy, kernel)
    u, v = calculate_u_v_lucas_kanade(D, ix, iy, it, kernel)

    convergence_u = None
    for i in range(n_iters):
        ua = cv2.filter2D(u, -1, Ld)
        va = cv2.filter2D(v, -1, Ld)
        P = ix * ua + iy * va + it
        D = lmbd + ix ** 2 + iy ** 2
        u = ua - (ix * (P / D))
        v = va - (iy * (P / D))
        if convergence_u is not None:
            break
        convergence_u = u

    return u, v