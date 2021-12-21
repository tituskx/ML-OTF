from twoD.gen_solutions.weak_forms import h1_inner_product
from twoD.util_otf import matrix_otf, knotvector_otf
import numpy as np
import sys
import time
import resource

a = 0
b = 1

n_otf = 26  # when setting n_otf, we want to make sure that we always use a number that is a multiple of #elements on which the trial function is non-zero. Since
# the Bsplines are non-zero on one, two, and three elements, using a discretisation of 24 elements (i.e. setting n_otf = 26, p_otf = 2) makes sure that
# that is the case.

p_otf = 2
# n_min = int(sys.argv[1])
n_min = 4
# n_max = int(sys.argv[2])
n_max = 6

elements = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)] # only consider elements that do not intersect the domain boundary

x_a = 0
y_a = 0

tic = time.perf_counter()

for n in range(n_min, n_max):
    matrix_tic = time.perf_counter()
    matrices = []
    for j in range(len(elements)):
        coords, _, _ = knotvector_otf(a, b, n, 1)

        element = elements[j]

        x_b = coords[element[0]]
        y_b = coords[element[1]]

        _, knots_x, _ = knotvector_otf(x_a, x_b, n_otf, p_otf)
        _, knots_y, _ = knotvector_otf(y_a, y_b, n_otf, p_otf)

        knots = (knots_x, knots_y)

        small_tic = time.perf_counter()

        A = matrix_otf(p_otf, p_otf, knots, n_otf, h1_inner_product, x_a, x_b, y_a, y_b)

        print('Building matrix in iteration %d took %.2f seconds' % (n - 3, time.perf_counter() - small_tic) * (
                    n == 4 or n == 5))

        matrices.append([n, element, A, x_b - x_a, y_b - y_a])

    print('Size of matrices list: %.2f' % (sys.getsizeof(matrices)))
    print('Max RSS in Kb: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    with open('matrices_notf%d_potf%d_n%d.npy' % (n_otf, p_otf, n), 'wb') as file:
        np.save(file, matrices)

    print('Generating matrices corresponding to disretisation %d took %.2f seconds \n' % (
    n, time.perf_counter() - matrix_tic))

print('Saving all matrices took %.2f seconds' % (time.perf_counter() - tic))
