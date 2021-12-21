import sys
from twoD.gen_solutions.weak_forms import *
from twoD.util_otf import *
from twoD.gen_solutions.util_input_functions import *
from twoD.gen_solutions.util_generate import *
import time
import numpy as np
import tensorflow as tf
import resource
import matplotlib.pyplot as plt


a = 0
b = 1
n = 10

peclet = 10
d = 1 / peclet
beta = np.array([[1], [1]])

quad_points = 10

n_otf = 26
p_otf = 2

num_training_points = 10000
num_test_points = 400
num_sensors = 20

tic = time.perf_counter()

coords_x, coords_y, knots_constant, knots_C0_lin, knots_C0_quad, knots_C1_quad = gen_knots(a, b, n)
input_functions = gen_all_input_functions(num_sensors, num_training_points, coords_x, coords_y, knots_C0_lin, knots_C0_quad, knots_C1_quad)

matrix_files_location = ''
training_dataset_location = ''
testing_dataset_location = ''

with open(matrix_files_location, 'rb') as file:
        M = np.load(file, allow_pickle=True)

with tf.io.TFRecordWriter(training_dataset_location) as writer:
    for j in range(len(input_functions)):
        input_dict =input_functions[j]
        input_func = input_dict['input_func']
        vars, sol = gen_data_points(num_training_points=num_training_points, weak_form=weak_form, v=beta, d=d,
                                           input_dict=input_dict, M=M,
                                           n_otf=n_otf, p_otf=p_otf, constant=0, slope=1)

        for feature in zip(input_func, vars, sol):
            example = serialise_example(*feature)
            writer.write(example)

print('Training data saved')

test_input_functions = gen_all_input_functions(num_sensors, num_test_points, coords_x, coords_y, knots_C0_lin, knots_C0_quad, knots_C1_quad)

with tf.io.TFRecordWriter(testing_dataset_location) as writer:
    for j in range(len(test_input_functions)):
        input_dict = test_input_functions[j]
        input_func = input_dict['input_func']
        vars, sol = gen_data_points(num_training_points=num_test_points, weak_form=weak_form, v=beta, d=d,
                                    input_dict=input_dict, M=M,
                                    n_otf=n_otf, p_otf=p_otf, constant=0, slope=1)

        for feature in zip(input_func, vars, sol):
            example = serialise_example(*feature)
            writer.write(example)

print('Testing data saved')
print('Max RSS in Kb after running process: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


