from twoD.util_otf import shape_func_grad_otf, shape_func_otf_small,shape_func_otf_grad_small, cart_product_otf, gen_quadrature_otf
from numba import jit
import numpy as np
from numpy import polynomial

X_otf, W_bdry = polynomial.legendre.leggauss(10)
W_otf = cart_product_otf(W_bdry, W_bdry)
W_otf = np.prod(W_otf, axis = 1)

@jit(nopython=True)
def h1_inner_product(x_L, x_R, y_L, y_R, trial_func, test_func, x_a = None, x_b = None, y_a = None, y_b = None, knots = None, p_trial = None, p_test = None, n = None):
    X = gen_quadrature_otf(x_L, x_R, y_L, y_R, X_otf)

    trial_dx, trial_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, trial_func, n, p_trial)
    test_dx, test_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, test_func, n, p_test)

    grad_prod = trial_dx * test_dx + trial_dy * test_dy

    trial_val = shape_func_otf_small(X, x_a, x_b, y_a, y_b, trial_func, n, p_trial)
    test_val = shape_func_otf_small(X, x_a, x_b, y_a, y_b, test_func, n, p_test)

    integral1 = np.sum(W_otf * grad_prod)
    integral2 = np.sum(W_otf * trial_val * test_val)

    step = (x_R - x_L) * (y_R - y_L) / 4

    return step * (((x_b - x_a) ** 2 + (y_b - y_a) ** 2) * integral1 + integral2)

@jit(nopython=True)
def weak_form(x_L, x_R, y_L, y_R, test_func, knots = None, p = None, v = None, d = None, x_a = None, x_b = None, y_a = None, y_b = None, knots_source = None, p_source = None, spline_nums = None, n = None, constant = None, slope = None):
    X = gen_quadrature_otf(x_L, x_R, y_L, y_R, X_otf)

    source_dx, source_dy = shape_func_grad_otf(X, knots = knots_source, p = p_source, spline_nums = spline_nums)
    source_dx = source_dx * slope
    source_dy = source_dy * slope
    test_dx, test_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, test_func, n, p)

    test_val = shape_func_otf_small(X, x_a, x_b, y_a, y_b, test_func, n, p)
    grad_prod = source_dx * test_dx + source_dy * test_dy

    grad_beta = v[0] * source_dx + v[1] * source_dy

    step = (x_R - x_L) * (y_R - y_L) / 4

    integral = np.sum(W_otf * (d * grad_prod + grad_beta * test_val))

    return step * integral


@jit(nopython=True) # this function and the next can be used to split up the diffusion and advection part of the weak form so that solutions to multiple values for epsilon can
# be generated more quickly
def weak_form_diffusion(x_L, x_R, y_L, y_R, test_func, knots = None, p = None, v = None, d = None, x_a = None, x_b = None, y_a = None, y_b = None, knots_source = None, p_source = None, spline_nums = None, n = None, constant = None, slope = None):
    X = gen_quadrature_otf(x_L, x_R, y_L, y_R, X_otf)

    source_dx, source_dy = shape_func_grad_otf(X, knots = knots_source, p = p_source, spline_nums = spline_nums)
    source_dx = source_dx * slope
    source_dy = source_dy * slope
    test_dx, test_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, test_func, n, p)

    grad_prod = source_dx * test_dx + source_dy * test_dy

    step = (x_R - x_L) * (y_R - y_L) / 4

    integral = np.sum(W_otf * (d * grad_prod))

    return step * integral

@jit(nopython=True)
def weak_form_adv(x_L, x_R, y_L, y_R, test_func, knots = None, p = None, v = None, d = None, x_a = None, x_b = None, y_a = None, y_b = None, knots_source = None, p_source = None, spline_nums = None, n = None, constant = None, slope = None):
    X = gen_quadrature_otf(x_L, x_R, y_L, y_R, X_otf)

    source_dx, source_dy = shape_func_grad_otf(X, knots = knots_source, p = p_source, spline_nums = spline_nums)
    source_dx = source_dx * slope
    source_dy = source_dy * slope
    test_dx, test_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, test_func, n, p)

    test_val = shape_func_otf_small(X, x_a, x_b, y_a, y_b, test_func, n, p)

    grad_beta = v[0] * source_dx + v[1] * source_dy

    step = (x_R - x_L) * (y_R - y_L) / 4

    integral = np.sum(W_otf * (grad_beta * test_val))

    return step * integral
