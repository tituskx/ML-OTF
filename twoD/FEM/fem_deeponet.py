from twoD.util import *
import numpy as np
import tensorflow as tf

class lhs():
    def __init__(self, **kwargs):
        self.quad_points = kwargs.get('quad_points')
        self.knots_trial = kwargs.get('knots_trial')
        self.knots_test = kwargs.get('knots_test')
        self.p_trial = kwargs.get('p_trial')
        self.p_test = kwargs.get('p_test')
        self.d = kwargs.get('d')
        self.v = kwargs.get('v')
        self.net = kwargs.get('net')


    def __call__(self, x_L, x_R, y_L, y_R, trial_func, test_func):
        X, W = gen_quadrature(x_L, x_R, y_L, y_R, self.quad_points)

        x_min, x_max = match_index_C1_quad(test_func[0], n)
        y_min, y_max = match_index_C1_quad(test_func[1], n)

        # x_min, x_max = match_index(test_func[0], n)
        # y_min, y_max = match_index(test_func[1], n)

        input_func = gen_input_func(coords_x[x_min], coords_x[x_max], coords_y[y_min], coords_y[y_max], self.knots_test, self.p_test, test_func, self.quad_points)
        test_val, test_grad = calc_net(input_func, X, coords_x[x_min], coords_x[x_max], coords_y[y_min], coords_y[y_max], self.net)

        trial_grad = shape_func_grad(X, knots=self.knots_trial, p=self.p_trial, spline_nums=trial_func)

        grad_prod = np.sum(trial_grad * test_grad, axis = 0)

        beta_grad_prod = np.sum(self.v * trial_grad, axis = 0)

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
        self.net = kwargs.get('net')

    def __call__(self, x_L, x_R, y_L, y_R, test_func):
        X, W = gen_quadrature(x_L, x_R, y_L, y_R, self.quad_points)

        # x_min, x_max = match_index_C1_quad(test_func[0], n)
        # y_min, y_max = match_index_C1_quad(test_func[1], n)

        # x_min, x_max = match_index(test_func[0], n)
        # y_min, y_max = match_index(test_func[1], n)

        input_func = gen_input_func(coords_x[x_min], coords_x[x_max], coords_y[y_min], coords_y[y_max], self.knots_test,
                                    self.p_test, test_func, self.quad_points)

        test_val, test_grad = calc_net(input_func, X, coords_x[x_min], coords_x[x_max], coords_y[y_min],
                                       coords_y[y_max], self.net)

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
n = 10


peclet = 100
d = 1 / peclet
beta = np.array([[1], [1]])


coords_x, knots_x, _ = knotvector(a, b, n, 1)
coords_y, knots_y, _ = knotvector(a, b, n, 1)

knots = (knots_x, knots_y)

knots_constant_x = knots_x[1: -1]
knots_constant_y = knots_y[1: -1]
knots_constant = (knots_constant_x, knots_constant_y)

knots_quad_x = np.insert(knots_x, [0, -1], [0, 1])
knots_quad_y = np.insert(knots_y, [0, -1], [0, 1])
knots_quad = (knots_quad_x, knots_quad_y)

u_trial = 2
v_trial = 2

u_n = n + u_trial - 1

u_knots = knots_quad
v_knots = knots_quad

quad_points = 10

otf_generator = None

model_path = ''
deeponet = tf.keras.models.load_model(model_path)

lhs = lhs(v = beta, d = d, p_trial = u_trial, knots_trial = u_knots, p_test = v_trial, knots_test = v_knots, source = forcing_term, quad_points = quad_points,
          gen_func = otf_generator, net = deeponet)
rhs = rhs(v = beta, d = d, p_test = v_trial, knots_test = v_knots, source = forcing_term, quad_points = quad_points,
          gen_func = otf_generator, net = deeponet)


A = matrix(u_trial, v_trial, knots, n, lhs)
f = vector(v_trial, knots, n, rhs)

A, f = bound_cond(A, f, u_n)

u = np.linalg.solve(A, f)
N = 100

plot_sol(u, u_knots, u_trial, a, b, a, b, n, peclet, N, u_sol, net = True)
