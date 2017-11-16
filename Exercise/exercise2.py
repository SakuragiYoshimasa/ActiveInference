#coding: utf-8
'''
Exercise 2.
Write a computer program finding the most likely size of the food item φ for the situation described in Exercise 1.
Initialize φ = vp, and then find its values in the next 5 time units (you can use Euler’s method,
i.e. update φ(t + ?t) = φ(t) + ?t∂F/∂φ with ?t
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from exercise1 import g

def derivative_F(phai, v_p=3.0):
    sigma_p = 1.0
    sigma_u = 1.0
    u = 2.0

    f_term = (v_p - phai) / sigma_p
    s_term = (u - g(phai)) / sigma_u * 2.0 * phai

    return f_term + s_term

def main():

    phais = [3.0]

    for i in range(500):
        updated = phais[-1] + 0.01 * derivative_F(phais[-1])
        phais.append(updated)

    plt.plot(phais)
    plt.show()
    return

if __name__ == '__main__':
    main()
