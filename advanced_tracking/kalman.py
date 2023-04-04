import math

import matplotlib.pyplot as plt

from ex4_utils import kalman_step
import numpy as np
import sympy as sp


def initialize_kalman(v):
    x = np.cos(v) + v * np.pi
    y = np.sin(v) * x

    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    return x, y, sx, sy


def ncv(qv=1.0, r=100.0):
    T, q = sp.symbols('T q')
    F = sp.Matrix([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    A = sp.exp(F * T)
    A = np.array(A.subs(T, 1)).astype(np.float32)

    T = sp.symbols('T')
    L = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    Q = sp.integrate((A * L) * q * (A * L).T, (T, 0, T))
    Q = np.array(Q.subs({T: 1, q: qv})).astype(np.float32)

    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]]).astype(np.float32)

    R = r * np.array([[1, 0],
                      [0, 1]]).astype(np.float32)

    return A, C, Q, R


def rw(qv=1.0, r=100.0):
    T, q = sp.symbols('T q')
    F = sp.Matrix([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    A = sp.exp(F * T)
    A = np.array(A.subs(T, 1)).astype(np.float32)
    T = sp.symbols('T')
    L = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    Q = sp.integrate((A * L) * q * (A * L).T, (T, 0, T))
    Q = np.array(Q.subs({T: 1, q: qv})).astype(np.float32)

    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]]).astype(np.float32)

    R = r * np.array([[1, 0],
                      [0, 1]]).astype(np.float32)

    return A, C, Q, R


def nca(qv=100.0, r=1.0):
    T, q = sp.symbols('T q')
    F = sp.Matrix([[0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])
    A = sp.exp(F * T)
    A = np.array(A.subs(T, 1)).astype(np.float32)

    T = sp.symbols('T')
    L = sp.Matrix([[0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 1]])
    Q = sp.integrate((A * L) * q * (A * L).T, (T, 0, T))
    Q = np.array(Q.subs({T: 1, q: qv})).astype(np.float32)
    print(Q)
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]]).astype(np.float32)

    R = r * np.array([[1, 0],
                      [0, 1]]).astype(np.float32)

    return A, C, Q, R


def test(p, A, C, Qi, Ri, qi, ri):
    x = p[0]
    y = p[1]
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]
    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Qi, Ri, np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                              np.reshape(state, (-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]

    return x, y, sx, sy, qi, ri


def calculate_selected(qr, l, imgs):
    q = qr[0]
    r = qr[1]
    # A, C, Q, R = rw(q, r)
    A, C, Q, R = nca(q, r)
    x, y, sx, sy, qi, ri = test(l, A, C, Q, R, q, r)

    plt.plot(x, y, 'bo-', label='measurements')
    plt.plot(sx, sy, 'ro-', label='filtered')
    plt.title("NCA " + "q = " + str(q) + " r = " + str(r))
    plt.legend()
    plt.show()
    # plt.savefig('data\\' + imgs)
    plt.close()


if __name__ == '__main__':
    # circular
    N = 40
    v = np.linspace(5 * np.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    l_tup = (x, y)
    qr_tup = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]
    img_names = ['rw01', 'rw02', 'rw03', 'rw04', 'rw05']
    # img_names = ['ncv01', 'ncv02', 'ncv03', 'ncv04', 'ncv05']
    # img_names = ['nca01', 'nca02', 'nca03', 'nca04', 'nca05']
    for idx, el in enumerate(qr_tup):
        calculate_selected(el, l_tup, img_names[idx])
