import numpy as np
import itertools
import tensorflow as tf
from numpy import polynomial
from twoD.util_otf import fem_otf
import matplotlib.pyplot as plt
import math


def rnd_func(x, coefficients, n):
    X = n * (x + 1)
    return np.dot(coefficients, np.cos(X))

def rnd_func_dx(x, coefficients, n):
    X = n * (x + 1)
    return np.dot(coefficients, - np.sin(X) * n)


def knotvector(a, b, n, p):
    coords = np.linspace(a, b, n - p + 1)

    multiplicity = np.zeros(n - p + 1)
    knots = np.zeros(p + n + 1)

    multiplicity[0] = p + 1
    multiplicity[n - p] = p + 1
    multiplicity[1:(n - p)] = np.ones(n - p - 1)

    knots[0:p] = np.zeros(p)
    knots[p:(n + 1)] = coords
    knots[(n + 1):(n + p + 1)] = np.ones(p)

    return coords, knots, multiplicity


def function_elmat(p, n):
    elmat = np.zeros((n - p, 2), dtype=np.int_)

    for i in range(0, n - p):
        elmat[i, 0] = p + i
        elmat[i, 1] = p + i + 1

    # print("elmat", elmat)
    return elmat


def Bspline(xx, knots, p, i):
    y = np.zeros(xx.shape)  # y has same length as xx array

    if p == 0:
        y[(xx >= knots[i]) & (xx < knots[i + 1])] = 1

    if knots[i + p] != knots[i]:
        c1 = (xx - knots[i]) / (knots[i + p] - knots[i]) * Bspline(xx, knots, p - 1, i)
        y += c1

    if knots[i + p + 1] != knots[i + 1]:
        c2 = (knots[i + p + 1] - xx) / (knots[i + p + 1] - knots[i + 1]) * Bspline(xx, knots, p - 1, i + 1)
        y += c2

    return y


def Bspline_der(xx, knots, p, i):
    y_der = np.zeros(xx.shape)

    if p == 0:
        y_der[(xx >= knots[i]) & (xx < knots[i + 1])] = 1

    if knots[i + p] != knots[i]:
        c1 = p / (knots[i + p] - knots[i]) * Bspline(xx, knots, p - 1, i)
        y_der += c1

    if knots[i + p + 1] != knots[i + 1]:
        c2 = p / (knots[i + p + 1] - knots[i + 1]) * Bspline(xx, knots, p - 1, i + 1)
        y_der -= c2

    return y_der

def cartesian_product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))


def p0_support(n, i):
    return [i]


def p1_support(n, i):
    if i == 0:
        return [i]
        # return []
    elif i == (n - 1):
        # return [i - 1]
        return [i - 1]
    else:
        return [i - 1, i]


def C0_p2_support(n, i):
    if i == 0:
        return [0]
    elif i == (2 * n - 2):
        return [int(i / 2) - 1]
    elif (i % 2) != 0:
        return [int(math.floor(i / 2))]
    else:
        return [int(i / 2) - 1, int(i / 2)]


def C1_p2_support(n, i):
    if i == 0:
        return [0]
    elif i == 1:
        return [0, 1]
    elif i == n:
        return [i - 2]
    elif i == (n - 1):
        return [i - 2, i - 1]
    else:
        return [i - 2, i - 1, i]


def support(shape_func, n, p, C0, disc):

    i, j = shape_func

    if C0 == True:
        x_support, y_support = C0_p2_support(n, i), C0_p2_support(n, j)
    elif disc == True:
        x_support, y_support = [math.floor(i / 2)], [math.floor(j / 2)]
    elif p == 0:
        x_support, y_support = p0_support(n, i), p0_support(n, j)
    elif p == 1:
        x_support, y_support = p1_support(n, i), p1_support(n, j)
    else:
        x_support, y_support = C1_p2_support(n, i), C1_p2_support(n, j)

    return cartesian_product_itertools([x_support, y_support])


def intersect(trial_function, test_function, n, p_trial, p_test, C0_trial, C0_test):
    trial_function_support = support(trial_function, n, p_trial, C0_trial, False)
    test_function_support = support(test_function, n, p_test, C0_test, False)
    support_indicator = [(trial_func == test_function_support).all(axis=1).any() for trial_func in
                         trial_function_support]

    return trial_function_support[support_indicator]

def shape_func(Xcross, knots = None, p = None, spline_nums = None, coefficients = None, n_coeff = None, rnd = False):
    if rnd == False:
        func_val = Bspline(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline(Xcross[:, 1], knots[1], p, spline_nums[1])
    else:
        x = np.reshape(Xcross[:, 0], (1, -1))
        y = np.reshape(Xcross[:, 1], (1, -1))
        func_val = rnd_func(x, coefficients[0, :], n_coeff) * rnd_func(y, coefficients[1, :], n_coeff)

    return func_val


def shape_func_grad(Xcross, knots = None, p = None, spline_nums = None, coefficients = None, n_coeff = None, rnd = False):
    if rnd == False:
        grad = np.array([
            Bspline_der(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline(Xcross[:, 1], knots[1], p, spline_nums[1]),
            Bspline(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline_der(Xcross[:, 1], knots[1], p, spline_nums[1])
                         ])
    else:
        x = np.reshape(Xcross[:, 0], (1, -1))
        y = np.reshape(Xcross[:, 1], (1, -1))
        grad = np.array([
            rnd_func_dx(x, coefficients[0, :], n_coeff) * rnd_func(y, coefficients[1, :], n_coeff),
            rnd_func(x, coefficients[0, :], n_coeff) * rnd_func_dx(y, coefficients[1, :], n_coeff)
        ])

    return grad



def gen_quadrature(x_L, x_R, y_L, y_R, quad_points):
    X, W = polynomial.legendre.leggauss(quad_points)

    x = (x_R - x_L) / 2 * X + (x_L + x_R) / 2
    y = (y_R - y_L) / 2 * X + (y_L + y_R) / 2

    Wcross = cartesian_product_itertools([W, W])
    Wcross = np.prod(Wcross, axis=1)
    Xcross = cartesian_product_itertools([x, y])

    return Xcross, Wcross


def bound_cond(A, f, u_n):
    test_functions = cartesian_product_itertools([range(u_n), range(u_n)])

    bdry_rows = [not {0, u_n - 1}.isdisjoint(set(test_func)) for test_func in test_functions]
    bdry_cols = test_functions[bdry_rows][:, 0] * u_n + test_functions[bdry_rows][:, 1]

    A[bdry_rows] = np.zeros(A[bdry_rows].shape)
    A[bdry_rows, bdry_cols] = np.ones(A[bdry_rows, bdry_cols].shape)
    f[bdry_rows] = np.zeros((len(test_functions[bdry_rows]),))

    return A, f

def bound_cond_mixed(A31, A32, A33, f3, n, v_trial):
    test_functions = cartesian_product_itertools([range(n + v_trial - 1), range(n + v_trial - 1)])

    bdry_rows = [not {0, n + v_trial - 2}.isdisjoint(set(test_func)) for test_func in test_functions]
    bdry_cols = test_functions[bdry_rows][:, 0] * (n + v_trial - 1) + test_functions[bdry_rows][:, 1]

    A31[bdry_rows] = np.zeros(A31[bdry_rows].shape)
    A32[bdry_rows] = np.zeros(A32[bdry_rows].shape)
    A33[bdry_rows] = np.zeros(A33[bdry_rows].shape)
    A33[bdry_rows, bdry_cols] = np.ones(A33[bdry_rows, bdry_cols].shape)

    f3[bdry_rows] = np.zeros(len(test_functions[bdry_rows]))

    return A31, A32, A33, f3


def matrix(p_trial, p_test, knots, n, weak_form, trial_C0 = False, test_C0 = False):
    elmat = function_elmat(1, n)

    trial_n = trial_C0 * (n - p_trial) + n + p_trial - 1
    test_n = test_C0 * (n - p_test) + n + p_test - 1

    K = np.zeros((test_n ** 2, trial_n ** 2))

    trial_functions = cartesian_product_itertools([range(trial_n), range(trial_n)])
    test_functions = cartesian_product_itertools([range(test_n), range(test_n)])

    i = 0
    for test_func in test_functions:
        j = 0
        for trial_func in trial_functions:
            K[i, j] = matr_elem(weak_form, trial_func, test_func, p_trial, p_test, knots, n, elmat, trial_C0, test_C0)
            j += 1
        i += 1

    return K


def matr_elem(weak_form, trial_func, test_func, p_trial, p_test, knots, n, elmat, trial_C0, test_C0):
    integral = 0
    elements = intersect(trial_func, test_func, n, p_trial, p_test, trial_C0, test_C0)

    knots_x, knots_y = knots
    for element in elements:
        x_left, x_right = elmat[element[0]]
        y_left, y_right = elmat[element[1]]
        x_L, x_R = knots_x[x_left], knots_x[x_right]
        y_L, y_R = knots_y[y_left], knots_y[y_right]

        integral += weak_form(x_L, x_R, y_L, y_R, trial_func, test_func)

    return integral


def vector(p_test, knots, n, weak_form, test_C0 = False):

    elmat = function_elmat(1, n)
    test_n = test_C0 * (n - p_test) + n + p_test - 1

    K = np.zeros(test_n ** 2)

    test_functions = cartesian_product_itertools([range(test_n), range(test_n)])

    i = 0
    for test_func in test_functions:
        K[i] += vector_elem(test_func, p_test, knots, n, elmat, weak_form, test_C0)
        i += 1

    return K


def vector_elem(test_func, p_test, knots, n, elmat, weak_form, test_C0):
    integral = 0
    elements = support(test_func, n, p_test, test_C0, False)

    knots_x, knots_y = knots

    for element in elements:
        x_left, x_right = elmat[element[0]]
        y_left, y_right = elmat[element[1]]
        x_L, x_R = knots_x[x_left], knots_x[x_right]
        y_L, y_R = knots_y[y_left], knots_y[y_right]
        integral += weak_form(x_L, x_R, y_L, y_R, test_func)

    return integral



def plot_sol(u, knots_trial, p_trial, x_a, x_b, y_a, y_b, n, peclet, N, u_sol, otf = False, C0 = False, net = False, Galerkin = False):

    x = np.arange(x_a, x_b, (x_b - x_a) / N)
    y = np.arange(y_a, y_b, (y_b - y_a) / N)

    X = cartesian_product_itertools([x, y])

    u_n = C0 * (n - p_trial) + n + p_trial - 1
    trial_functions = cartesian_product_itertools([range(u_n), range(u_n)])
    u_curve = np.zeros((N ** 2,))

    for i in range(u_n ** 2):
        u_curve += u[i] * shape_func(X, knots=knots_trial, p=p_trial, spline_nums=trial_functions[i])

    u_sol_plot = u_sol(X[:, 0], X[:, 1], peclet)

    N_quad = 100
    X_quad, W_quad = np.polynomial.legendre.leggauss(N_quad)
    x_quad = (x_b - x_a) / 2 * X_quad + (x_a + x_b) / 2
    y_quad = (y_b - y_a) / 2 * X_quad + (y_a + y_b) / 2

    X_quad = cartesian_product_itertools([x_quad, y_quad])
    W_quad = cartesian_product_itertools([W_quad, W_quad])
    W_quad = np.prod(W_quad, axis=1)

    u_curve_quad = np.zeros(N_quad ** 2)
    u_curve_quad_grad = np.zeros((2, N_quad ** 2))

    for i in range(u_n ** 2):
        u_curve_quad += u[i] * shape_func(X_quad, knots=knots_trial, p=p_trial, spline_nums=trial_functions[i])
        trial_dx, trial_dy = shape_func_grad(X_quad, knots=knots_trial, p=p_trial, spline_nums=trial_functions[i])
        u_curve_quad_grad[0, :] += u[i] * trial_dx
        u_curve_quad_grad[1, :] += u[i] * trial_dy

    u_sol_quad = u_sol(X_quad[:, 0], X_quad[:, 1], peclet)
    u_sol_quad_grad = np.array([(1 - (peclet * np.exp(peclet * X_quad[:, 0])) / (-1 + np.exp(peclet))) * (X_quad[:, 1] + (np.exp(peclet * X_quad[:, 1]) - 1) / (1 - np.exp(peclet))),
                  (1 - (peclet * np.exp(peclet * X_quad[:, 1])) / (-1 + np.exp(peclet))) * (X_quad[:, 0] + (np.exp(peclet * X_quad[:, 0]) - 1) / (1 - np.exp(peclet)))])

    H1_error_integral1 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * ((u_sol_quad - u_curve_quad) ** 2))
    H1_error_integral2 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * ((u_sol_quad_grad - u_curve_quad_grad) ** 2))

    H1_error = np.sqrt(H1_error_integral1 + H1_error_integral2)
    print('H1 Error: ', H1_error)

    extent = np.min(x), np.max(x), np.max(y), np.min(y)
    minmin = np.min([np.min(u_curve), np.min(u_sol_plot)])
    maxmax = np.max([np.max(u_curve), np.max(u_sol_plot)])

    fig, ax = plt.subplots(nrows=1, ncols=2)

    im1 = ax[0].imshow(np.reshape(u_curve, (N, N)), extent=extent, vmin=minmin, vmax=maxmax, cmap=plt.cm.RdBu)
    im2 = ax[1].imshow(np.reshape(u_sol_plot, (N, N)), extent=extent, vmin=minmin, vmax=maxmax, cmap=plt.cm.RdBu)
    # plt.suptitle('$\Vert u - u_n\Vert_{H^1(\Omega)} = %.3f$' % H1_error, fontsize = 20)
    ax[0].set_title('Galerkin ' * Galerkin + 'FEM' + ' w/ FEM OTF' * otf + ' w/ DeepONet' * net)
    ax[1].set_title('Exact Solution')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    # plt.suptitle("Discretisation using $%d \, x \,%d$ shape functions" % (n, n))
    plt.show()


def gen_sol(u, knots_trial, p_trial, C0, x_a, x_b, y_a, y_b, n, peclet, N, u_sol, otf = False, d = False, net = False):

    u_n = C0 * (n - p_trial) + n + p_trial - 1

    trial_functions = cartesian_product_itertools([range(u_n), range(u_n)])

    N_quad = 100
    X_quad, W_quad = np.polynomial.legendre.leggauss(N_quad)
    x_quad = (x_b - x_a) / 2 * X_quad + (x_a + x_b) / 2
    y_quad = (y_b - y_a) / 2 * X_quad + (y_a + y_b) / 2

    X_quad = cartesian_product_itertools([x_quad, y_quad])
    W_quad = cartesian_product_itertools([W_quad, W_quad])
    W_quad = np.prod(W_quad, axis=1)

    u_curve_quad = np.zeros(N_quad ** 2)
    u_curve_quad_grad = np.zeros((2, N_quad ** 2))

    for i in range(u_n ** 2):
        u_curve_quad += u[i] * shape_func(X_quad, knots=knots_trial, p=p_trial, spline_nums=trial_functions[i])
        trial_dx, trial_dy = shape_func_grad(X_quad, knots=knots_trial, p=p_trial, spline_nums=trial_functions[i])
        u_curve_quad_grad[0, :] += u[i] * trial_dx
        u_curve_quad_grad[1, :] += u[i] * trial_dy

    u_sol_quad = u_sol(X_quad[:, 0], X_quad[:, 1], peclet)

    u_sol_quad_grad = np.array([(1 - (peclet * np.exp(peclet * X_quad[:, 0])) / (-1 + np.exp(peclet))) * (X_quad[:, 1] + (np.exp(peclet * X_quad[:, 1]) - 1) / (1 - np.exp(peclet))),
                                (1 - (peclet * np.exp(peclet * X_quad[:, 1])) / (-1 + np.exp(peclet))) * (X_quad[:, 0] + (np.exp(peclet * X_quad[:, 0]) - 1) / (1 - np.exp(peclet)))])

    H1_error_integral1 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * ((u_sol_quad - u_curve_quad) ** 2))
    H1_error_integral2 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * ((u_sol_quad_grad - u_curve_quad_grad) ** 2))

    H1_error = np.sqrt(H1_error_integral1 + H1_error_integral2)

    H1_norm_integral1 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * (u_sol_quad ** 2))
    H1_norm_integral2 = (x_b - x_a) * (y_b - y_a) / 4 * np.sum(W_quad.T * (u_sol_quad_grad ** 2))

    H1_norm_sol = np.sqrt(H1_norm_integral1 + H1_norm_integral2)

    return [u_n, p_trial, C0, otf, d, net, H1_error, H1_norm_sol]


def match_index(i, n):
    if i == 0:
        x_min = 0
        x_max = 1
    elif i == (n - 1):
        x_min = -2
        x_max = -1
    else:
        x_min = i - 1
        x_max = i + 1
    return x_min, x_max


def match_index_C0_quad(i, n):
    if i == 0:
        x_min = 0
        x_max = 1
    elif i == (2 * n - 2):
        x_min = int(i / 2) - 1
        x_max = int(i / 2)

    elif (i % 2) != 0:
        x_min = int(math.floor(i / 2))
        x_max = int(math.floor(i / 2)) + 1
    else:
        x_min = int(i / 2) - 1
        x_max = int(i / 2) + 1

    return x_min, x_max

def match_index_C1_quad(i, n):
    if i == 0:
        x_min = 0
        x_max = 1
    elif i == 1:
        x_min = 0
        x_max = 2
    elif i == n:
        x_min = n - 2
        x_max = n - 1
    elif i == (n - 1):
        x_min = i - 2
        x_max = i
    else:
        x_min = i - 2
        x_max = i + 1
    return x_min, x_max

def calc_net(input_func, X, x_a, x_b, y_a, y_b, net, var = False):
    X_new = X.copy()
    X_new[:, 0] = (X_new[:, 0] - x_a) / (x_b - x_a)
    X_new[:, 1] = (X_new[:, 1] - y_a) / (y_b - y_a)
    X_new = tf.constant(X_new, dtype = float)

    with tf.GradientTape() as tape:
        tape.watch(X_new)
        net_out = net(input_func, X_new)

    net_grad = tape.gradient(net_out, X_new)

    net_out = np.reshape(net_out, (X_new.shape[0],))

    scale = np.ones((net_grad.shape[0], 2 + var))

    scale[:, 0] = scale[:, 0] * 1 / (x_b - x_a)
    scale[:, 1] = scale[:, 1] * 1 / (y_b - y_a)

    net_grad = net_grad * scale
    net_grad = tf.transpose(net_grad)

    return net_out, net_grad[:2, :]


def gen_input_func(x_min, x_max, y_min, y_max, knots, p, spline_nums, num_points):
    X = cartesian_product_itertools([np.arange(x_min, x_max, (x_max - x_min) / 20), np.arange(y_min, y_max, (y_max - y_min) / 20)])
    input_func = shape_func(X, knots=knots, p=p, spline_nums=spline_nums)
    input_func = np.reshape(input_func, (1, input_func.shape[0]))
    input_func = np.repeat(input_func, num_points ** 2, axis=0)
    return input_func