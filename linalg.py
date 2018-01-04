# -*- coding: utf-8 -*-
import numpy as np

A = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print('A: \n', A)
print('Shape: \n', np.shape(A))
print('A (2,3): \n', A[1, 2])

v = np.array([1, 2, 3])
print('Vector v: \n', v)
print('\n')

# Addition and Scalar Multiplication
A = np.matrix([[1, 2, 4], [5, 3, 2]])
B = np.matrix([[1, 3, 4], [1, 1, 1]])
print('A: \n', A)
print('B: \n', B)

add = np.add(A, B)
print('Add: \n', add)

sub = np.subtract(A, B)
print('Subtract: \n', sub)

div = np.divide(A, B)
print('Divide: \n', div)

add_scalar = A + 2
print('Add matrix and scalar: \n', add_scalar)
print('\n')

# Matrix-Vector Multiplication
A = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)

v = np.array([[1], [1], [1]])
print('Vector v: \n', v)

Av = A * v
print("A * v: \n", Av)
print('\n')

# Matrix Multiplication Properties
# Matrices are not commutative: A*B ≠ B*A
# Matrices are associative: (A*B)*C = A*(B*C)
A = np.matrix([[1, 2], [4, 5]])
B = np.matrix([[1, 1], [0, 2]])
print('A: \n', A)
print('B: \n', B)

I = np.identity(2)
print("I: \n", I)

IA = I * A
print("I * A: \n", IA)

AI = A * I
print("A * I: \n", AI)

AB = A * B
print("A * B: \n", AB)

BA = B * A
print("B * A: \n", BA)
print('\n')

# Inverse and Transpose
# The inverse of a matrix A is denoted A−1
A = np.matrix([[1, 2, 0], [0, 5, 6], [7, 0, 9]])
print('A: \n', A)

At = np.transpose(A)
print('A^1: \n', At)

Ai = np.linalg.inv(A)
print('Inverse of A: \n', Ai)

A_Ai = np.matmul(A, Ai)
print('A * Inverse of A: \n', A_Ai)
