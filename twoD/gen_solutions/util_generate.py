from twoD.util_otf import *
import numpy as np

def gen_data_points(num_training_points = None, weak_form = None, v = None, d = None, input_dict = None, M = None,
                    n_otf = None, p_otf = None, constant = None, slope = None):

    M_slice = [input_dict['elem'] == M_elem for M_elem in M[:, 1]]

    A = M[M_slice][:, 2][0]

    _, knots_x, _ = knotvector_otf(input_dict['x_a'], input_dict['x_b'], n_otf, p_otf)
    _, knots_y, _ = knotvector_otf(input_dict['y_a'], input_dict['y_b'], n_otf, p_otf)

    knots = (knots_x, knots_y)

    f = vector_otf(p_otf, knots, n_otf, weak_form, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], v=v, d=d,
                   knots_source=input_dict['knots'], p_source=input_dict['degree'], spline_nums=input_dict['spline_nums'],
                   constant=constant, slope=slope)

    A, f = bound_cond_otf(A, f, n_otf, p_otf)

    u = np.linalg.solve(A, f)

    x = np.random.uniform(input_dict['x_a'], input_dict['x_b'], (num_training_points, 1))
    y = np.random.uniform(input_dict['y_a'], input_dict['y_b'], (num_training_points, 1))

    X = np.concatenate([x, y], axis=1)

    sol = fem_otf(u, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], n_otf, p_otf, X)
    sol = sol[..., None]

    X[:, 0] = (X[:, 0] - input_dict['x_a']) / (input_dict['x_b'] - input_dict['x_a'])
    X[:, 1] = (X[:, 1] - input_dict['y_a']) / (input_dict['y_b'] - input_dict['y_a'])

    return X, sol


def gen_data_points_var_disc(num_training_points = None, weak_form = None, v = None, d = None, input_dict = None, M = None, n = None,
                    n_otf = None, p_otf = None, constant = None, slope = None, l_min = None, l_max = None):

    M_slice = [input_dict['elem'] == M_elem for M_elem in M[:, 1]]

    A = M[M_slice][:, 2][0]
    lx = M[M_slice][:, 3][0]
    ly = M[M_slice][:, 4][0]

    _, knots_x, _ = knotvector_otf(input_dict['x_a'], input_dict['x_b'], n_otf, p_otf)
    _, knots_y, _ = knotvector_otf(input_dict['y_a'], input_dict['y_b'], n_otf, p_otf)

    knots = (knots_x, knots_y)

    f = vector_otf(p_otf, knots, n_otf, weak_form, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], v=v, d=d,
                   knots_source=input_dict['knots'], p_source=input_dict['degree'], spline_nums=input_dict['spline_nums'],
                   constant=constant, slope=slope)

    A, f = bound_cond_otf(A, f, n_otf, p_otf)

    u = np.linalg.solve(A, f)

    x = np.random.uniform(input_dict['x_a'], input_dict['x_b'], (num_training_points, 1))
    y = np.random.uniform(input_dict['y_a'], input_dict['y_b'], (num_training_points, 1))

    X = np.concatenate([x, y], axis=1)

    sol = fem_otf(u, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], n_otf, p_otf, X)
    sol = sol[..., None]

    X[:, 0] = (X[:, 0] - input_dict['x_a']) / (input_dict['x_b'] - input_dict['x_a'])
    X[:, 1] = (X[:, 1] - input_dict['y_a']) / (input_dict['y_b'] - input_dict['y_a'])

    lx_array = lx * np.ones((X.shape[0], 1))
    lx_array = (lx_array - l_min) / (l_max - l_min)

    ly_array = ly * np.ones((X.shape[0], 1))
    ly_array = (ly_array - l_min) / (l_max - l_min)

    vars = np.concatenate([X, lx_array, ly_array], axis=1)

    return vars, sol



def gen_data_points_var_peclet(num_training_points = None, weak_form_adv = None, weak_form_diff = None, v = None, d = None, input_dict = None, M = None,
                    n_otf = None, p_otf = None, constant = None, slope = None):

    M_slice = [input_dict['elem'] == M_elem for M_elem in M[:, 1]]

    A = M[M_slice][:, 2][0]

    _, knots_x, _ = knotvector_otf(input_dict['x_a'], input_dict['x_b'], n_otf, p_otf)
    _, knots_y, _ = knotvector_otf(input_dict['y_a'], input_dict['y_b'], n_otf, p_otf)

    knots = (knots_x, knots_y)

    f_diff = vector_otf(p_otf, knots, n_otf, weak_form_diff, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], v=v, d=1,
                   knots_source=input_dict['knots'], p_source=input_dict['degree'], spline_nums=input_dict['spline_nums'],
                   constant=constant, slope=slope)

    f_adv = vector_otf(p_otf, knots, n_otf, weak_form_adv, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], v=v, d=1,
                   knots_source=input_dict['knots'], p_source=input_dict['degree'], spline_nums=input_dict['spline_nums'],
                   constant=constant, slope=slope)

    f_diff = bound_cond_otf_f(f_diff, n_otf, p_otf)
    f_adv = bound_cond_otf_f(f_adv, n_otf, p_otf)
    A = bound_cond_otf_A(A, n_otf, p_otf)

    f_diff = f_diff.reshape((1, -1))
    f_adv = f_adv.reshape((1, -1))
    f_adv = np.repeat(f_adv, num_training_points, axis = 0)

    max_peclet = 400

    diff_vec = 1 / np.random.uniform(1, max_peclet, (num_training_points, 1))
    f_diff = np.matmul(diff_vec, f_diff)

    f = f_diff + f_adv

    A = A.reshape((1, *A.shape))
    A = np.repeat(A, num_training_points, axis = 0)

    u = np.linalg.solve(A, f)

    x = np.random.uniform(input_dict['x_a'], input_dict['x_b'], (num_training_points, 1))
    y = np.random.uniform(input_dict['y_a'], input_dict['y_b'], (num_training_points, 1))

    X = np.concatenate([x, y], axis=1)

    sol = fem_otf_var_peclet(u, input_dict['x_a'], input_dict['x_b'], input_dict['y_a'], input_dict['y_b'], n_otf, p_otf, X)
    sol = sol[..., None]

    diff_vec = (diff_vec - 1) / (max_peclet - 1)
    vars = np.concatenate([X, diff_vec], axis = 1)

    X[:, 0] = (X[:, 0] - input_dict['x_a']) / (input_dict['x_b'] - input_dict['x_a'])
    X[:, 1] = (X[:, 1] - input_dict['y_a']) / (input_dict['y_b'] - input_dict['y_a'])

    return vars, sol
