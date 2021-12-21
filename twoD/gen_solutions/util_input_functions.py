from twoD.util import *
import matplotlib.pyplot as plt
import numpy as np


def build_input_dict(num_sensors, num_points, x_a, x_b, y_a, y_b, knots, degree, spline_nums, elem_index):
    x = np.arange(x_a, x_b, (x_b - x_a) / num_sensors)
    y = np.arange(y_a, y_b, (y_b - y_a) / num_sensors)
    elem = cartesian_product_itertools([x, y])

    input_func = shape_func(elem, knots, degree, spline_nums)
    input_func = input_func[None, ...]
    input_func = np.repeat(input_func, num_points, axis=0)

    input_dict = {'input_func': input_func, 'knots': knots, 'degree': degree, 'spline_nums': spline_nums,
                  'x_a': x_a, 'x_b': x_b, 'y_a': y_a, 'y_b': y_b, 'elem': elem_index}

    return input_dict

def gen_input_func1(num_sensors, num_training_points, coords_x, coords_y, knots_C0_quad):
    input_C0_quad_dict = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[1], coords_y[0], coords_y[1], knots_C0_quad, 2, (1, 1), (1, 1))

    return [input_C0_quad_dict]

def gen_input_func2(num_sensors, num_training_points, coords_x, coords_y, knots_C0_quad):
    input_C0_quad_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[0],
                                          coords_y[1], knots_C0_quad, 2, (2, 1), (2, 1))

    input_C0_quad_dict2 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[1], coords_y[0],
                                           coords_y[2], knots_C0_quad, 2, (1, 2), (1, 2))

    return [input_C0_quad_dict1, input_C0_quad_dict2]

def gen_input_func4(num_sensors, num_training_points, coords_x, coords_y, knots_C0_lin, knots_C0_quad, knots_C1_quad):

    input_C0_lin_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[0],
                                           coords_y[2], knots_C0_lin, 1, (1, 1), (2, 2))

    input_C0_quad_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[0],
                                          coords_y[2], knots_C0_quad, 2, (2, 2), (2, 2))

    input_C1_quad_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[0],
                                          coords_y[2], knots_C1_quad, 2, (1, 1), (2, 2))

    input_C1_quad_dict2 = build_input_dict(num_sensors, num_training_points, coords_x[-3], coords_x[-1], coords_y[-3],
                                           coords_y[-1], knots_C1_quad, 2, (len(coords_x) - 1, len(coords_y) - 1), (2, 2))

    input_C1_quad_dict3 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[-3],
                                           coords_y[-1], knots_C1_quad, 2, (1, len(coords_y) - 1), (2, 2))

    input_C1_quad_dict4 = build_input_dict(num_sensors, num_training_points, coords_x[-3], coords_x[-1], coords_y[0],
                                           coords_y[2], knots_C1_quad, 2, (len(coords_x) - 1, 1), (2, 2))


    return [input_C0_lin_dict1, input_C0_quad_dict1, input_C1_quad_dict1, input_C1_quad_dict2, input_C1_quad_dict3, input_C1_quad_dict4]

def gen_input_func6(num_sensors, num_training_points, coords_x, coords_y, knots_C1_quad):

    input_C1_quad_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[3], coords_y[0],
                                          coords_y[2], knots_C1_quad, 2, (2, 1), (3, 2))

    input_C1_quad_dict2 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[3], coords_y[-3],
                                           coords_y[-1], knots_C1_quad, 2, (2, len(coords_y) - 1), (3, 2))

    input_C1_quad_dict3 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[2], coords_y[0],
                                           coords_y[3], knots_C1_quad, 2, (1, 2), (2, 3))

    input_C1_quad_dict4 = build_input_dict(num_sensors, num_training_points, coords_x[-3], coords_x[-1], coords_y[0],
                                           coords_y[3], knots_C1_quad, 2, (len(coords_x) - 1, 2), (2, 3))

    return [input_C1_quad_dict1, input_C1_quad_dict2, input_C1_quad_dict3, input_C1_quad_dict4]

def gen_input_func9(num_sensors, num_training_points, coords_x, coords_y, knots_C1_quad):
    input_C1_quad_dict1 = build_input_dict(num_sensors, num_training_points, coords_x[0], coords_x[3], coords_y[0],
                                           coords_y[3], knots_C1_quad, 2, (2, 2), (3, 3))

    return [input_C1_quad_dict1]


def gen_knots(a, b, n):
    coords_x, knots_x, _ = knotvector(a, b, n, 1)
    coords_y, knots_y, _ = knotvector(a, b, n, 1)

    coords = (coords_x, coords_y)
    knots = (knots_x, knots_y)

    knots_constant_x = knots_x[1: -1]
    knots_constant_y = knots_y[1: -1]
    knots_constant = (knots_constant_x, knots_constant_y)

    knots_C0_quad_x = np.repeat(knots_x, 2)[1: -1]
    knots_C0_quad_y = np.repeat(knots_y, 2)[1: -1]
    knots_C0_quad = (knots_C0_quad_x, knots_C0_quad_y)

    knots_C1_quad_x = np.insert(knots_x, [0, -1], [0, 1])
    knots_C1_quad_y = np.insert(knots_y, [0, -1], [0, 1])
    knots_C1_quad = (knots_C1_quad_x, knots_C1_quad_y)

    return coords_x, coords_y, knots_constant, knots, knots_C0_quad, knots_C1_quad

def gen_all_input_functions(num_sensors, num_training_points, coords_x, coords_y, knots_C0_lin, knots_C0_quad, knots_C1_quad):
    input1 = gen_input_func1(num_sensors, num_training_points, coords_x, coords_y, knots_C0_quad)
    input2 = gen_input_func2(num_sensors, num_training_points, coords_x, coords_y, knots_C0_quad)
    input4 = gen_input_func4(num_sensors, num_training_points, coords_x, coords_y, knots_C0_lin, knots_C0_quad, knots_C1_quad)
    input6 = gen_input_func6(num_sensors, num_training_points, coords_x, coords_y, knots_C1_quad)
    input9 = gen_input_func9(num_sensors, num_training_points, coords_x, coords_y, knots_C1_quad)
    total_input = input1 + input2 + input4 + input6 + input9

    return total_input

