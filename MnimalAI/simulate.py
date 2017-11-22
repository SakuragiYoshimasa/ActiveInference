#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

k = 100
eta = 0.01
n = 16
psi_0 = n / 2.0
k = np.power(4.0, - 1.0 / 16)
omega = np.log(4.0) / 16.0
roh = 0.75
lower_optimise = 0.001

'''
b := brain state. size n.
a := action. -1 or 1.
s := sensory input High or Low.
psi := environment state.
b_star := a sort of target brain state
'''
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

def P_psinext(psi_next, psi, a):
    if psi_next == psi % n:
        return 1.0 - roh
    elif psi_next == (psi + a) % n:
        return roh
    return 0.0

def P_s(s, psi):
    if s == 'High':
        return k * np.exp(-omega * np.abs(psi - psi_0))
    else:
        return 1.0 - k * np.exp(-omega * np.abs(psi - psi_0))

def P_next_psi(psi_next, s, b, a, psi):
    return P_psinext(psi_next, psi, a)

def P_s_(s, b, a, psi):
    return P_s(s, psi)

def P_psi(psi, a, b):
    return softmax(b)


def calc_eps(b_next, b, s, a):

    softmax_psi_next = softmax(b_next)
    softmax_psi = softmax(b)
    eps = 0.0

    for psi_next in range(n):

        in_log = 0.0
        for psi in [psi_next, (psi_next - a) % n]:
            in_log += P_psinext(psi_next, psi, a) * P_s(s, psi) * softmax_psi[psi]

        eps += softmax_psi_next[psi_next] * np.log(in_log)
    return eps

def calc_entrp(b_next):
    entrp = 0
    softmax_psi_next = softmax(b_next)
    for i in range(n):
        entrp -= softmax_psi_next[i] * np.log(softmax_psi_next[i])
    return entrp

def F(b_next, b, s, a):
    eps = calc_eps(b_next, b, s, a)
    entrp = calc_entrp(b_next)
    return - eps - entrp

def derivative_soft_max(psi_next, b_next, ind):
    softmax_psi_next = softmax(b_next)
    delta = 1.0 if psi_next == ind else 0.0
    return softmax_psi_next * (delta - softmax_psi_next[ind])

def p_next_psi_s(psi_next, s, b, a):

    p = 0.0
    for psi in range(n):
        p += P_next_psi(psi_next, s, b, a, psi) * P_s_(s, b, a, psi) * P_psi(psi, a, b)
    return p

def grad_F_on_b_next(b_next, b, s, a):

    grad_F = []
    softmax_psi_next = softmax(b_next)
    for j in range(n):
        grad = 0.0
        for psi_next in range(n):
            grad += derivative_soft_max(psi_next, b_next, j) * (1.0 + np.log(softmax_psi_next[psi_next] / p_next_psi_s(psi_next, s, b, a)))
        grad_F.append(grad)
    return np.array(grad_F)

def argmin_F(b_star, b, s):
    back_F = F(b_star, b, s, - 1)
    forw_F = F(b_star, b, s, 1)
    if back_F <= forw_F:
        return -1
    else:
        return 1

def random_from_P_s(psi):
    r = np.random.rand()
    if r <= k * np.exp(-omega * np.abs(psi - psi_0)):
        return 'High'
    else:
        return 'Low'

def random_from_P_psinext(psi, a):
    r = np.random.rand()
    if r <= 1.0 - roh:
        return psi % n
    else:
        return (psi + a) % n


def optimise(b, s, a):
    b_next = np.zeros(n, dtype='float')
    prev_ = 0
    for i in range(100):
        print('optimise %d, %d' % (i, len(b_next)))
        #print(b_next)

        b_next = b_next - eta * grad_F_on_b_next(b_next, b, s, a)
        if abs(np.linalg.norm(b_next) - prev_) < lower_optimise:
            break
        else:
            prev_ = np.linalg.norm(b_next)
    return b_next

def simulate(psi, b, b_star):

    finished = False

    for i in range(100):
        s = random_from_P_s(psi)
        print(s)
        a = argmin_F(b_star, b, s)
        print(a)
        b_next = optimise(b, s, a)
        print(b_next)
        psi_next = random_from_P_psinext(psi, a)
        print(psi_next)
        b, psi = [b_next, psi_next]
        print(b)
        print(psi)
    return

if __name__ == '__main__':
    simulate(5, np.ones(n, dtype='float') / n, np.array([np.abs(i - n / 2) for i in range(n)]))
