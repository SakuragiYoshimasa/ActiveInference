# coding: utf-8
'''
Exercise 1
Assume that our animal observed the light intensity u = 2, the level of noise in its receptor is Σu = 1,
and the mean and variance of its prior expectation of size are vp = 3 andΣp = 1. Write a computer program
that computes the posterior probabilities of sizes from 0.01 to 5, and plots them.
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import integrate

def p_u_v(u, g_v, sigma_u=1.0):
    return norm.pdf(x=u, loc=g_v, scale=sigma_u)

def integrated(v):
    return p_v(v) * p_u_v(2.0, g(v))

def p_u(u):
    result, err = integrate.quad(integrated, 0, 5)
    return result

def p_v(v, v_p=3.0, sigma_p=1.0):
    return norm.pdf(x=v, loc=v_p, scale=sigma_p)

def g(v):
    return np.power(v, 2.0)

def p_v_u(v, u=2):
    return p_v(v) * p_u_v(u, g(v)) / p_u(u)


def main():

    p_v_u_arr = []
    for v in np.arange(0.01, 5.0, 0.01):
        p_v_u_arr.append(p_v_u(v))

    print(p_v_u_arr)
    plt.plot(p_v_u_arr)
    plt.show()
    return

if __name__ == '__main__':
    main()
