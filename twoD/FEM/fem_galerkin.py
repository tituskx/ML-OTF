from twoD.util import *
import numpy as np
import time


class lhs():
    def __init__(self, **kwargs):
        self.quad_points = kwargs.get('quad_points')
        self.knots_trial = kwargs.get('knots_trial')
        self.p_trial = kwargs.get('p_trial')
        self.knots_test = kwargs.get('knots_test')
        self.p_test = kwargs.get('p_test')
        self.d = kwargs.get('d')
        self.v = kwargs.get('v')

    def __call__(self, x_L, x_R, y_L, y_R, trial_func, test_func):
        X, W = gen_quadrature(x_L, x_R, y_L, y_R, self.quad_points)

        trial_grad = shape_func_grad(X, knots=self.knots_trial, p=self.p_trial, spline_nums=trial_func)
        test_grad = shape_func_grad(X, knots=self.knots_test, p=self.p_test, spline_nums=test_func)

        grad_prod = np.sum(trial_grad * test_grad, axis = 0)

        beta_grad_prod = np.sum(self.v * trial_grad, axis = 0)
        test_val = shape_func(X, knots = self.knots_test, p = self.p_test, spline_nums = test_func)

        step = (x_R - x_L) * (y_R - y_L) / 4
        integral = np.sum(W * (self.d * grad_prod + beta_grad_prod * test_val))

        return step * integral


class rhs():
    def __init__(self, **kwargs):
        self.quad_points = kwargs.get('quad_points')
        self.knots_trial = kwargs.get('knots_trial')
        self.p_trial = kwargs.get('p_trial')
        self.knots_test = kwargs.get('knots_test')
        self.p_test = kwargs.get('p_test')
        self.d = kwargs.get('d')
        self.v = kwargs.get('v')
        self.source = kwargs.get('source')

    def __call__(self, x_L, x_R, y_L, y_R, test_func):

        X, W = gen_quadrature(x_L, x_R, y_L, y_R, self.quad_points)

        test_val = shape_func(X, knots=self.knots_test, p=self.p_test, spline_nums=test_func)
        source_val = self.source(X, self.v, self.d)

        step = (x_R - x_L) * (y_R - y_L) / 4
        integral = np.sum(W * test_val * source_val)

        return step * integral

def du_dx(x, y):
    return (1 - (peclet * np.exp(peclet * x)) / (-1 + np.exp(peclet))) * (y + (np.exp(peclet * y) - 1) / (1 - np.exp(peclet)))

def du_dx2(x, y):
    return - (peclet ** 2 * np.exp(peclet * x)) / (-1 + np.exp(peclet)) * (y + (np.exp(peclet * y) - 1) / (1 - np.exp(peclet)))

def forcing_term(X, v, d):
    return - d * du_dx2(X[:, 0], X[:, 1]) - d * du_dx2(X[:, 1], X[:, 0]) + v[0] * du_dx(X[:, 0], X[:, 1]) + v[1] * du_dx(X[:, 1], X[:, 0])

def u_sol(x, y, peclet):
    return (x + (np.exp(peclet * x) - 1) / (1 - np.exp(peclet))) * (y + (np.exp(peclet * y) - 1) / (1 - np.exp(peclet)))


a = 0
b = 1
n = 20

peclet = 100
d = 1 / peclet

beta = np.array([[1], [1]])

coords_x, knots_x, _ = knotvector(a, b, n, 1)
coords_y, knots_y, _ = knotvector(a, b, n, 1)

knots = (knots_x, knots_y)

knots_C0_quad_x = np.repeat(knots_x, 2)[1: -1]
knots_C0_quad_y = np.repeat(knots_y, 2)[1: -1]
knots_C0_quad = (knots_C0_quad_x, knots_C0_quad_y)

knots_C1_quad_x = np.insert(knots_x, [0, -1], [0, 1])
knots_C1_quad_y = np.insert(knots_y, [0, -1], [0, 1])
knots_C1_quad = (knots_C1_quad_x, knots_C1_quad_y)

u_trial = 1
v_trial = 1

u_knots = knots
v_knots = knots

C0 = False

u_n = n + u_trial - 1
# u_n = 2 * n - 1

quad_points = 10

lhs = lhs(v = beta, d = d, p_trial = u_trial, knots_trial = u_knots, p_test = v_trial, knots_test = v_knots, source = forcing_term, quad_points = quad_points)
rhs = rhs(v = beta, d = d, p_test = v_trial, knots_test = v_knots, source = forcing_term, quad_points = quad_points)

tic = time.perf_counter()

A = matrix(u_trial, v_trial, knots, n, lhs, trial_C0 = C0, test_C0 = C0)
f = vector(v_trial, knots, n, rhs, test_C0 = C0)
A, f = bound_cond(A, f, u_n)

u = np.linalg.solve(A, f)

print('Building and solving system took %.2f seconds' % (time.perf_counter() - tic))

N = 100

plot_sol(u, u_knots, u_trial, a, b, a, b, n, peclet, N, u_sol, C0 = C0, Galerkin = True)