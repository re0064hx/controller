#! /usr/bin/python
"""
Inverted Pendulum MPC control
author: Atsushi Sakai
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy
from matplotrecorder import matplotrecorder


l_bar = 2.0  # length of bar
M = 1.0  # [kg]
m = 0.3  # [kg]
g = 9.8  # [m/s^2]

Q = np.diag([0.0, 1.0, 1.0, 0.0])
R = np.diag([0.01])
nx = 4
nu = 1   
T = 30  
delta_t = 0.1

animation = False


def main():
    x0 = np.array([
        [0.0],
        [0.0],
        [0.3],
        [0.0]
    ])

    x = np.copy(x0)

    for i in range(50):

        ox, dx, otheta, dtheta, ou = mpc_control(x)
        u = ou[0]
        x = simulation(x, u)
        print(i)
        print(x)
        print(u)

        plt.clf()
        px = float(x[0])
        theta = float(x[2])
        show_cart(px, theta)
        plt.xlim([-5.0, 2.0])
        plt.pause(0.001)

        if animation:
            matplotrecorder.save_frame()

    if animation:
        matplotrecorder.save_movie("animation.gif", 0.1)


def simulation(x, u):

    A, B = get_model_matrix()

    x = np.dot(A, x) + np.dot(B, u)

    return x


def mpc_control(x0):

    x = cvxpy.Variable(nx, T + 1)
    u = cvxpy.Variable(nu, T)

    A, B = get_model_matrix()

    cost = 0.0
    constr = []
    for t in range(T):
        cost += cvxpy.quad_form(x[:, t + 1], Q)
        cost += cvxpy.quad_form(u[:, t], R)
        constr += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]
    constr += [x[:, 0] == x0]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

    start = time.time()
    prob.solve(verbose=False)
    elapsed_time = time.time() - start
    print("calc time:{0} [sec]".format(elapsed_time))

    if prob.status == cvxpy.OPTIMAL:
        ox = get_nparray_from_matrix(x.value[0, :])
        dx = get_nparray_from_matrix(x.value[1, :])
        theta = get_nparray_from_matrix(x.value[2, :])
        dtheta = get_nparray_from_matrix(x.value[3, :])

        ou = get_nparray_from_matrix(u.value[0, :])

    return ox, dx, theta, dtheta, ou


def get_nparray_from_matrix(x):
    u"""
    get build-in list from matrix
    """
    return np.array(x).flatten()


def get_model_matrix():

    # Model Parameter
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B


def flatten(a):
    return np.array(a).flatten()


def show_cart(xt, theta):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")


if __name__ == '__main__':
    main()