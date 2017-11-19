#coding: utf-8
'''
Having specified the state-space and contingencies,
one can solve the belief updating equations (Eq. (7)) to simulate behaviour.
The (concentration) parameters of the habits were initialised to the sum of all transition probabilities: c =?uB(u).
Prior beliefs about the initial state were initialised to d = 8 for the central location for each context and zero otherwise.
Finally, prior beliefs about policies were initialised to e = 4 with the exception of the habit, where e = 0.
These concentration parameters can be regarded as the num- ber of times each state, transition or policy has been encountered in previous trials
'''

'''
Infernece: Optimising expectations about hidden states.
Learning : Optimising model parameters.
'''

'''
Notation
sequences of variables over time => v_seq
expectations of parameters => p_exp
logarithm => v_log
'''

'''
Output set O
o_tau ∈ {0, 1}
o_tau_exp ∈ [0, 1]

Hidden state set S
s_tau ∈ {0, 1}
s_tau_exp ∈ [0, 1]

Policy set T
pi = (pi_1, .. pi_K) ∈ {0, 1}
pi_exp = (pi_exp_1, .. pi_exp_K) ∈ [0, 1]

Action
u = pi(t) ∈ {1,2,...}

gamma ∈ R                           :Precision of beliefs about policies

A ∈ [0, 1]                          :Likelifood matrix mapping from hidden states to outcomes
B_tau^pi = B(u=pi(tau)) ∈ [0, 1]    :transition probability for hidden states under each action prescribed by a policy at a paticular time.
C := B_tau^0_log ∈ [0, 1]           :transition probability for hidden state under a habit
D ∈ [0, 1]                          :Prior expectation of each state at the begginig of each trial
E ∈ [0, 1]                          :Prior expectation of eacg policy at the begginig of each trial

U_tau = ln P(o_tau) ∈ R             :logarithm of prior preferences or utility over outcomes

F:F_tau = F(pi) = Sum_tau F(pi, tau) ∈ R : Variatinal free energy for each policy
G:G_tau = G(pi) = Sum_tau g(pi, tau) ∈ R : Expected free energy for each policy

H = -diag()
'''

'''
Model parameters of generative model
η = {a,b,c,d,e,β}
a := P(A) = Dir(a)
b := P(B) = Dir(b)
c := P(C) = Dir(c)
d := P(D) = Dir(d)
e := P(E) = Dir(e)
β := P(γ) = Γ(1, β)


'''
