#coding: utf-8
'''
Exercise 3. Simulate the model from Fig. 3 for the problem from Exercise 1.
In particular, initialize φ = vp, εp = εu = 0, and find their values for the next 5 units of time.
'''
from exercise1 import g
import matplotlib.pyplot as plt
import numpy as np

def main():
    u = 2.0
    v_p = 3.0
    sigma_u = 1.0
    sigma_p = 1.0
    phais = [3.0]
    eps_ps = [0.0]
    eps_us = [0.0]
    delta = 0.01

    for t in range(500):

        der_phai = eps_us[-1] * 2.0 * phais[-1] - eps_ps[-1]
        der_eps_ps = phais[-1] - v_p - sigma_p * eps_ps[-1]
        der_eps_us = u - np.power(phais[-1], 2.0) - sigma_u * eps_us[-1]

        phais.append(phais[-1] + delta * der_phai)
        eps_ps.append(eps_ps[-1] + delta * der_eps_ps)
        eps_us.append(eps_us[-1] + delta * der_eps_us)

    plt.plot(phais, color='r', label='phais')
    plt.plot(eps_ps, color='g', label='eps_ps')
    plt.plot(eps_us, color='b', label='eps_us')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
