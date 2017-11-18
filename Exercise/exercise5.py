#coding: utf-8
'''
Exercise 5.
Simulate learning of variance Σi over trials. For simplicity,
only simulate the network described by Eqs. (59)–(60), and assume that variables φ are constant.
On each trial generate input φi from a normal distribution with mean 5 and variance 2,
while set gi(φi+1) = 5 (so that the upper level correctly predicts the mean of φi).
Simulate the network for 20 time units, and then update weight Σi with learning rate α = 0.01.
Simulate 1000 trials and plot howΣi changes across trials
'''
import numpy as np
import matplotlib.pyplot as plt


eps_i = 0.0
e_i = 0.0
phai_i = 2.0
mean_phi = 5
sigma_phi = 2
phi_above = 5
dt = 0.01
maxt = 20
trials = 1000
lrate = 0.01

sigmas = [1.0]

for i in range(trials):
    errors = [0.0] #predition error
    es = [0.0] #interneuron
    phai = np.random.normal(mean_phi, sigma_phi)

    for j in range(int(maxt / dt)):
        errors.append( errors[-1] + dt * (phai - phi_above - es[-1]))
        es.append(es[-1] + dt * (sigmas[-1] * errors[-2] - es[-1]))

    sigmas.append(sigmas[-1] + lrate * (errors[-1] * es[-1] -1.0))

plt.plot(sigmas)
plt.show()
