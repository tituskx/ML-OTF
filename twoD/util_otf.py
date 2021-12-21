import numpy as np
from numba import jit, njit
import tensorflow as tf
import math


def support_otf_new(i, n, p):
    if i < p:
        support = list(range(0, i + 1))
    elif i >= n - p:
        support = list(range(i - p, n - p))
    else:
        support = list(range(i - p, i + 1))

    return support

def support_otf_nodes(i, n, p):
    if i < p:
        support = list(range(0, i + 2))
    elif i >= n - p:
        support = list(range(i - p, n - p + 1))
    else:
        support = list(range(i - p, i + 2))

    return support

@jit(nopython=True)
def knotvector_otf(a, b, n, p):
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



@jit(nopython=True)
def function_elmat_otf(p, n):
    elmat = np.zeros((n - p, 2), dtype=np.int_)

    for i in np.arange(n - p):
        elmat[i, 0] = p + i
        elmat[i, 1] = p + i + 1

    # print("elmat", elmat)
    return elmat


@jit(nopython=True)
def Bspline_otf(xx, knots, p, i):
    y = np.zeros(xx.shape)  # y has same length as xx array

    if p == 0:
        y[(xx >= knots[i]) & (xx < knots[i + 1])] = 1

    if knots[i + p] != knots[i]:
        c1 = (xx - knots[i]) / (knots[i + p] - knots[i]) * Bspline_otf(xx, knots, p - 1, i)
        y += c1

    if knots[i + p + 1] != knots[i + 1]:
        c2 = (knots[i + p + 1] - xx) / (knots[i + p + 1] - knots[i + 1]) * Bspline_otf(xx, knots, p - 1, i + 1)
        y += c2

    return y


@jit(nopython=True)
def Bspline_der_otf(xx, knots, p, i):
    y_der = np.zeros(xx.shape)

    if p == 0:
        y_der[(xx >= knots[i]) & (xx < knots[i + 1])] = 1

    if knots[i + p] != knots[i]:
        c1 = p / (knots[i + p] - knots[i]) * Bspline_otf(xx, knots, p - 1, i)
        y_der += c1

    if knots[i + p + 1] != knots[i + 1]:
        c2 = p / (knots[i + p + 1] - knots[i + 1]) * Bspline_otf(xx, knots, p - 1, i + 1)
        y_der -= c2

    return y_der

def bound_cond_otf(A, f, n, p_test):
    test_functions = cart_product_otf(np.arange(0, n, 1), np.arange(0, n, 1))

    bdry_rows = [not {0, n - 1}.isdisjoint(set(test_func)) for test_func in test_functions]
    bdry_cols = test_functions[bdry_rows][:, 0] * n + test_functions[bdry_rows][:, 1]

    A[bdry_rows] = np.zeros(A[bdry_rows].shape)
    A[bdry_rows, bdry_cols] = np.ones(A[bdry_rows, bdry_cols].shape)
    f[bdry_rows] = np.zeros((len(test_functions[bdry_rows]),))

    return A, f

def bound_cond_otf_A(A, n, p_test):
    test_functions = cart_product_otf(np.arange(0, n, 1), np.arange(0, n, 1))

    bdry_rows = [not {0, n - 1}.isdisjoint(set(test_func)) for test_func in test_functions]
    bdry_cols = test_functions[bdry_rows][:, 0] * n + test_functions[bdry_rows][:, 1]

    A[bdry_rows] = np.zeros(A[bdry_rows].shape)
    A[bdry_rows, bdry_cols] = np.ones(A[bdry_rows, bdry_cols].shape)

    return A

def bound_cond_otf_f(f, n, p_test):
    test_functions = cart_product_otf(np.arange(0, n, 1), np.arange(0, n, 1))

    bdry_rows = [not {0, n - 1}.isdisjoint(set(test_func)) for test_func in test_functions]

    f[bdry_rows] = np.zeros((len(test_functions[bdry_rows]),))

    return f

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


@njit(fastmath=True)
def isin(b):
    present = False
    for element in b:
        if (element == True):
            return True

    return present

@njit(fastmath = True)
def check_elements_otf(elements, element_indicator):
    supported_elements = []
    for element, indicator in zip(elements, element_indicator):
        if indicator:
            supported_elements.append(element)

    return supported_elements

@jit(nopython=True)
def intersect_otf(trial_function, test_function, n, p_trial, p_test):

    trial_function_support = support_otf(trial_function, n, p_trial)
    test_function_support = support_otf(test_function, n, p_test)

    element_comparison = [[(test_func == trial_func).all() for test_func in test_function_support] for trial_func in trial_function_support]
    intersec_indicator = [isin(element) for element in element_comparison]

    intersection = check_elements_otf(trial_function_support, intersec_indicator)

    return intersection


@jit(nopython=True)
def cart_product_otf(x, y):
    return np.column_stack((np.repeat(x, len(y)), np.transpose(np.repeat(y, len(x)).reshape(-1, len(x))).flatten()))


@jit(nopython=True)
def support_otf(indices, n, p):
    i, j = indices

    if p == 0:
        x_support, y_support = p0_support_otf(i), p0_support_otf(j)
    elif p == 1:
        x_support, y_support = p1_support_otf(n, i), p1_support_otf(n, j)
    else:
        x_support, y_support = p2_support_otf(n, i), p2_support_otf(n, j)

    return cart_product_otf(x_support, y_support)


@jit(nopython=True)
def p0_support_otf(i):
    return [i]


@jit(nopython=True)
def p1_support_otf(n, i):
    if i == 0:
        return [i]
    elif i == (n - 1):
        return [i - 1]
    else:
        return [i - 1, i]


@jit(nopython=True)
def p2_support_otf(n, i):
    if i == 0:
        return [i]
    elif i == 1:
        return [i - 1, i]
    elif i == (n - 1):
        return [i - 2]
    elif i == (n - 2):
        return [i - 2, i - 1]
    else:
        return [i - 2, i - 1, i]


@jit(nopython=True)
def shape_func_otf(Xcross, knots, p, spline_nums):
    func_val = Bspline_otf(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline_otf(Xcross[:, 1], knots[1], p, spline_nums[1])

    return func_val


@jit(nopython=True)
def shape_func_grad_otf(Xcross, knots, p, spline_nums):
    grad = (
            Bspline_der_otf(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline_otf(Xcross[:, 1], knots[1], p, spline_nums[1]),
            Bspline_otf(Xcross[:, 0], knots[0], p, spline_nums[0]) * Bspline_der_otf(Xcross[:, 1], knots[1], p, spline_nums[1])
    )

    return grad


@jit(nopython = True)
def shape_func_otf_small(Xcross, x_min, x_max, y_min, y_max, spline_nums, n, p):
    coords_x_unit, knots_x_unit, _ = knotvector_otf(0, 1, n, p)
    coords_y_unit, knots_y_unit, _ = knotvector_otf(0, 1, n, p)

    knots_unit = [knots_x_unit, knots_y_unit]

    Xcross_new = Xcross.copy()

    Xcross_new[:, 0] = (Xcross[:, 0] - x_min) / (x_max - x_min)
    Xcross_new[:, 1] = (Xcross[:, 1] - y_min) / (y_max - y_min)

    return shape_func_otf(Xcross_new, knots_unit, p, spline_nums)




@jit(nopython = True)
def shape_func_otf_grad_small(Xcross, x_min, x_max, y_min, y_max, spline_nums, n, p):
    coords_x_unit, knots_x_unit, _ = knotvector_otf(0, 1, n, p)
    coords_y_unit, knots_y_unit, _ = knotvector_otf(0, 1, n, p)

    knots_unit = [knots_x_unit, knots_y_unit]

    Xcross_new = Xcross.copy()

    Xcross_new[:, 0] = (Xcross[:, 0] - x_min) / (x_max - x_min)
    Xcross_new[:, 1] = (Xcross[:, 1] - y_min) / (y_max - y_min)

    old_grad = shape_func_grad_otf(Xcross_new, knots_unit, p, spline_nums)

    new_grad = (
            old_grad[0] / (x_max - x_min),
            old_grad[1] / (y_max - y_min)
    )

    return new_grad


@jit(nopython=True)
def gen_quadrature_otf(x_L, x_R, y_L, y_R, X):

    x = (x_R - x_L) / 2 * X + (x_L + x_R) / 2
    y = (y_R - y_L) / 2 * X + (y_L + y_R) / 2

    Xcross = cart_product_otf(x, y)

    return Xcross


# @jit(nopython=True)
def matrix_otf(p_trial, p_test, knots, n, inner_product, x_a, x_b, y_a, y_b):
    K = np.zeros((n ** 2, n ** 2))
    elmat = function_elmat_otf(p_trial, n)

    trial_functions = cart_product_otf(np.arange(n), np.arange(n))
    test_functions = cart_product_otf(np.arange(n), np.arange(n))
    # matr_elem_otf_times = []
    i = 0
    for test_func in test_functions:
        j = 0
        for trial_func in trial_functions:
            # matrix_el_tic = time.perf_counter()
            K[i, j] += matr_elem_otf(inner_product, trial_func, test_func, p_trial, p_test, knots, n, elmat, x_a, x_b, y_a, y_b)
            # matrix_el_toc = time.perf_counter()
            # matr_elem_otf_times.append((matrix_el_toc - matrix_el_tic))
            j += 1
        i += 1
    # print('average matr elem otf time: %.6f' % (sum(matr_elem_otf_times[1:-1]) / len(matr_elem_otf_times[1:-1])))
    # print('total #matr elements: %d' % len(matr_elem_otf_times[1:-1]))
    # print()
    return K


# @jit(nopython=True)
def matr_elem_otf(inner_product, trial_func, test_func, p_trial, p_test, knots, n, elmat, x_a, x_b, y_a, y_b):
    integral = 0
    elements = intersect_otf(trial_func, test_func, n, p_trial, p_test)

    # integral_times = []
    knots_x, knots_y = knots
    for element in elements:
        x_left, x_right = elmat[element[0]]
        y_left, y_right = elmat[element[1]]
        x_L, x_R = knots_x[x_left], knots_x[x_right]
        y_L, y_R = knots_y[y_left], knots_y[y_right]
        # integral_tic = time.perf_counter()
        integral += inner_product(x_L, x_R, y_L, y_R, trial_func, test_func, x_a = x_a, x_b = x_b, y_a = y_a, y_b = y_b, knots = knots, p_trial = p_trial, p_test = p_test, n = n)
        # integral_toc = time.perf_counter()
        # integral_times.append((integral_toc - integral_tic))

    # if elements:
        # print('Average integral computation time: %.8f' % (sum(integral_times) / len(integral_times)))
        # print()

    return integral

@jit(nopython=True)
def vector_otf(p, knots, n, weak_form, x_a, x_b, y_a, y_b, v = None, d = None, knots_source = None, p_source = None, spline_nums = None, constant = None, slope = None):
    K = np.zeros((n ** 2))

    elmat = function_elmat_otf(p, n)
    test_functions = cart_product_otf(np.arange(n), np.arange(n))

    i = 0
    for test_func in test_functions:
        K[i] += vector_elem_otf(test_func, p, knots, n, elmat, weak_form, x_a,
           x_b, y_a, y_b, v, d, knots_source, p_source, spline_nums, constant, slope)
        i += 1

    return K

@jit(nopython=True)
def vector_elem_otf(test_func, p, knots, n, elmat, weak_form, x_a,
           x_b, y_a, y_b, v, d, knots_source, p_source, spline_nums, constant, slope):

    integral = 0
    elements = support_otf(test_func, n, p)
    knots_x, knots_y = knots

    for element in elements:
        x_left, x_right = elmat[element[0]]
        y_left, y_right = elmat[element[1]]
        x_L, x_R = knots_x[x_left], knots_x[x_right]
        y_L, y_R = knots_y[y_left], knots_y[y_right]
        integral += weak_form(x_L, x_R, y_L, y_R, test_func, knots = knots, p = p, v = v, d = d,
                              x_a = x_a, x_b = x_b, y_a = y_a, y_b = y_b, knots_source = knots_source, p_source = p_source, spline_nums = spline_nums, n = n, constant = constant, slope = slope)

    return integral


@jit(nopython = True)
def calc_bdry_term_otf_small(X, W, knots_source, p_source, p_test, spline_nums, test_func, n_test, x_a, x_b, y_a, y_b):
    return np.sum(W * shape_func_otf(X, knots_source, p_source, spline_nums) * shape_func_otf_small(X, x_a, x_b, y_a, y_b, test_func, n_test, p_test))


def coefficient_generator(x_a, x_b, y_a, y_b, v, d, knots_source, p_source, spline_nums, inner_product, weak_form, n_otf = None, p_otf = None, bc = True, constant = None, slope = None):
    if p_otf is None:
        p_otf = 1

    if n_otf is None:
        n_otf = 5

    coords_x, knots_x, _ = knotvector_otf(x_a, x_b, n_otf, p_otf)
    coords_y, knots_y, _ = knotvector_otf(y_a, y_b, n_otf, p_otf)

    knots = (knots_x, knots_y)

    A = matrix_otf(p_otf, p_otf, knots, n_otf, inner_product, x_a, x_b, y_a, y_b)
    f = vector_otf(p_otf, knots, n_otf, weak_form, x_a, x_b, y_a, y_b, v = v, d = d, knots_source = knots_source, p_source = p_source, spline_nums = spline_nums, constant = constant, slope = slope)

    if bc:
        A, f = bound_cond_otf(A, f, n_otf, p_otf)

    u = np.linalg.solve(A, f)

    return u, x_a, x_b, y_a, y_b, n_otf, p_otf


@jit(nopython=True)
def fem_otf(u, x_a, x_b, y_a, y_b, n, p, X): # here x_a, x_b, y_a, and y_b correspond to the boundaries of the element on which the original coefficient_generator computed the otf params
    trial_funcs = cart_product_otf(np.arange(n), np.arange(n))
    u_curve = np.zeros((X.shape[0],))
    for i in range(n ** 2):
        u_curve += u[i] * shape_func_otf_small(X, x_a, x_b, y_a, y_b, trial_funcs[i], n, p)

    return u_curve


@jit(nopython=True)
def fem_otf_grad(u, x_a, x_b, y_a, y_b, n, p, X):
    trial_funcs = cart_product_otf(np.arange(n), np.arange(n))
    u_curve = np.zeros((2, X.shape[0]))

    for i in range(n ** 2):
        trial_dx, trial_dy = shape_func_otf_grad_small(X, x_a, x_b, y_a, y_b, trial_funcs[i], n, p)
        u_curve[0, :] += u[i] * trial_dx
        u_curve[1, :] += u[i] * trial_dy

    return u_curve


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_array_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialise_example(feature_branch, feature_trunk, solution):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.

  feature = {
      'feature_branch': _float_array_feature(feature_branch), #this is the input function
      'feature_trunk': _float_array_feature(feature_trunk), # this is the value of x
      'solution': _float_array_feature(solution) # this is the solution
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

@jit(nopython=True)
def fem_otf_var_peclet(u, x_a, x_b, y_a, y_b, n, p, X):  # here x_a, x_b, y_a, and y_b correspond to the boundaries of the element on which the original coefficient_generator computed the otf params
    trial_funcs = cart_product_otf(np.arange(n), np.arange(n))
    u_curve = np.zeros((X.shape[0],))
    for i in range(n ** 2):
        u_curve += u[:, i] * shape_func_otf_small(X, x_a, x_b, y_a, y_b, trial_funcs[i], n, p)

    return u_curve