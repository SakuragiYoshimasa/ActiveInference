
<script src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Active Inference

## Notations and Definitions

### Notation
The 〜 notation denotes sequences of variables over time; e.g., $\widetilde{s}=(s_1,...,s_T)$.  
The entropy of a probability distribution $P(s) = Pr(S=s)$ is denoted by $H(S)=H[P(s)]=E_P[-lnP(s)]$.  
The relative entropy or KL divergence is denoted by $D[Q(s)||P(s)]=E_Q[ln Q(s) - ln P(s)]$.  
Products => inner $A\cdot B=A^TB$, outer $A\otimes B$.  
$\hat{s} = ln s$.  
$P(o|s)=Cat(A)$ implies $Pr(o=i|s=j)=Cat(A_{ij})$  
The expectations is denotes as bold style. e.g. $\boldsymbol{s}$


### The tuple Active Inference rest on
$(O, P, Q, R, S, T, U)$  

$O$ := A finite set of outcomes  
$U$ := A finite set of control states or actions  
$S$ := A finite set of hidden states   
$T$ := A finite set of time sensitive policies   
$R(\widetilde{o})$ := Generative process   
$P(\widetilde{o}, \widetilde{s}, \pi, \eta)$ := Generative model  
$Q(\widetilde{s}, \pi, \eta)$ := Approximate posterior  

### Parameters
$K$ means a number of control states or actions.  
$L$ means a number of hidden states.   
The tuple of model parameters $\eta = (a,b,c,d,e,\beta)$.  
T is the end of the current trial.  
t is the current time point.  

#### Hidden State
The state vector at particular time is $s_t$ (The dimension is $L$).  
 e.g. When the state is '1', it is expressed $s_t=(1,0,\dots, 0)^T$.

#### Action
$u\in U$ or $u\in \{1,2,\dots, K\}$.

#### Policy
The policies $\pi \in T$.  
It returns a sequence of actions $u_t=\pi(t)$.   
e.g. $\pi=(0,0,1,0)^T$ means $u_t=3$.

## The generative model
The generative model for partially observable parameterised Markov decision processes.  

$P(\widetilde{o}, \widetilde{s}, \pi, \eta) = P(\pi)P(\eta)\prod_{t=1}^{T}P(o_t|s_t)P(s_t|s_{t-1},\pi)$  

| Eq | Means |
|----|-------|    
|$P(o_t\|s_t)=Cat(A)$.| $A$ is a likelihood matrix mapping from hidden state to outcomes. |     
|$P(s_{t+1}\|s_t, \pi > 0)=Cat(B(u=\pi(t)))$    | $B$ is a transition probability matrix for hidden states under each action prescribed by a policy at a particular time. |   
|$P(s_{t+1}\|s_t, \pi = 0)=Cat(C)$   | $C$ is a transition probability matrix for hidden states under a habit.  |   
|$P(s_1, s_0)=Cat(D)$| $D$ is a prior expectation of each state at the beginning of each trial.|
|$P(\pi)=\sigma(\hat{E}-\gamma \cdot G)$   |   |
|$P(A) = Dir(a)$ same at $B,C,D,E$   |   |  
|$P(\gamma)=\Gamma(1,\beta)$   |  $\beta$ is a prior expectation of temperature of beliefs about policies. And $\gamma$ is precision. |  

## An approximate posterior
$\boldsymbol{x}=(\boldsymbol{s}_{0}^{\pi},\dots, \boldsymbol{s}_{T}^{\pi}, \boldsymbol{\pi}, \boldsymbol{\eta})$ and $\boldsymbol{\eta}=(\boldsymbol{a},\boldsymbol{b},\boldsymbol{c},\boldsymbol{d},\boldsymbol{e},\boldsymbol{\beta})$  


## Control state or action selection
This can be expressed in terms of minimizing (outcome) prediction errors.  

$u_t=min_uE_Q[D[P(o_{t+1}|s_{t+1})]||R(o_{t+1}|s_t,u)]] = min_u \boldsymbol{o}_{t+1} \cdot \epsilon_{t+1}^u$  

where  

$\epsilon_{t+1}^u=\widehat{\boldsymbol{o}}_{t+1} - \widehat{\boldsymbol{o}}_{t+1}^{u}$  

$\widehat{\boldsymbol{o}}_{t+1}=A\boldsymbol{s}_{t+1}$  

$\widehat{\boldsymbol{o}}_{t+1}^{u}=AB(u)\boldsymbol{s}_{t+1}$  

$\boldsymbol{s}_t=\sum_{\pi}\boldsymbol{\pi}_{\pi}\cdot \boldsymbol{s}_t^{\pi}$  

## Free energy and expected free energy
$Q(x)=argmin_{Q(x)}F$

$F=E_Q[ln Q(x)-lnP(x,\widetilde{o})]$

$G(\pi)=\sum_{\tau}G(\pi, \tau)$

$G(\pi, \tau)=E_Q[lnQ(s_{\tau}|\pi) - lnP(s_\tau, o_\tau | \widetilde{o}, \pi)] = \boldsymbol{o}_{\tau}^{\pi} \cdot (\widehat{\boldsymbol{o}}_{\tau}^{\pi} - U_r) + \boldsymbol{s}_\tau^\pi \cdot H$

where,

$\boldsymbol{o}_{\tau}^{\pi}=\breve{A}\cdot\boldsymbol{s}_\tau^\pi$

$\boldsymbol{U}_r=U(o_r)=lnP(o_r)$

$H=-diag(\breve{A}\cdot\widehat{A})$

$\widehat{A}=E_Q[lnA]=\psi(\boldsymbol{a}) - \psi(\boldsymbol{a}_0)$

$\breve{A}=E_Q[A_{ij}]=\boldsymbol{a}\times\boldsymbol{a}_0$

$\boldsymbol{a}_{0ij}=\sum_i \boldsymbol{a}_{ij}$


## Belief updating

### Inference
Optimizing expectations about hidden states.  

#### State estimation  

$\boldsymbol{s}_\tau^\pi=\sigma(\hat{A}\cdot o_\tau + \hat{B}_{\tau - 1}^{\pi} s_{\tau - 1}^{\pi} + \hat{B}_{\tau}^{\pi} s_{\tau + 1}^{\pi})$  

#### Policy selection
using  softmax function   

$\sigma(z)_{j}=\frac{e^{z_j}}{\sum_{k}^{}e^{z_k}}$  


$\boldsymbol{\pi} = \sigma(\hat{E} - F - \gamma \cdot G)$   

#### Precision

$\boldsymbol{\beta} = \beta + (\boldsymbol{\pi} - \boldsymbol{\pi}_0)\cdot G$


### Learning
Optimizing model parameters.  
Using a digamma function,

$\psi(x)=\frac{\partial}{\partial x}ln\Gamma(x)=\frac{\Gamma'(x)}{\Gamma(x)}$

$\hat{A}=\psi(\boldsymbol{a}) - \psi(\boldsymbol{a}_0)$  

$\hat{B}=\psi(\boldsymbol{b}) - \psi(\boldsymbol{b}_0)$  

$\hat{C}=\psi(\boldsymbol{c}) - \psi(\boldsymbol{c}_0)$  

$\hat{D}=\psi(\boldsymbol{d}) - \psi(\boldsymbol{d}_0)$

$\hat{E}=\psi(\boldsymbol{e}) - \psi(\boldsymbol{e}_0)$  

$\boldsymbol{a} = a+\sum_\tau o_\tau \otimes\boldsymbol{s}_\tau$

$\boldsymbol{b}(u)=b(u) + \sum_{\pi(\tau)=u}\boldsymbol{\pi}_\pi\cdot s_\tau^\pi\otimes s_{\tau - 1}^\pi$

$\boldsymbol{c}=c+\sum_\tau s_\tau^0\otimes s_{\tau - 1}^0$

$\boldsymbol{d}=d+\boldsymbol{s}_1$

$\boldsymbol{e}=e+\boldsymbol{\pi}$




# Flow
the (simulated) behavioural and physiological responses over 32 successive trials using a format that will be adopted in subsequent figures.
Each trial comprises two actions following an initial observation.
The top panel shows the initial states on each trial (as coloured circles) and subsequent policy selection (in image format) over the 11 policies considered.
The first 10 (allowable) policies correspond to staying at the center and then moving to each of the four locations, moving to the left or right arm and staying there, or moving to the lower arm and then moving to each of the four locations.
The 11th policy corresponds to a habit (i.e., state-action policy).

- sensory evidence is accumulated to from beliefs about the current state of the world.This evidence accumulation corresponds to state estimation under each policy.
- The quality of each policy is then evaluated in terms of its expected free energy.
-
