# -------------------------------------------------------------- #
# Description: Metropolis-Hastings for Bayesian Beta regression  #
#              with a multivariate Normal as prior for betas and #
#              an Uniform for phi                                #
# Author: Estevão Prado                                          #
# Last modification: 18/12/2018                                  #
# -------------------------------------------------------------- #

import numpy as np
import matplotlib as mt
from scipy.special import gamma as Gamma
import numpy.linalg as np_alg
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import gamma

# Simulate data ---------------------------------------------------------------
np.random.seed(123)
np.set_printoptions(precision=5) # Number of digits of precision for floating point output
Npost = 5000
Burn = int(Npost*0.5)
TotIter = Npost + Burn
j_phi = 0
j_beta = 0
N = 1000
alpha = -1
beta = 0.2
phi = 5

x_0 = np.repeat(1,N)
x_1 = np.random.uniform(0,10,N)
logit_mu = alpha + beta * x_1
mu =  np.exp(logit_mu)/(1+np.exp(logit_mu)) # inverse logit
a = mu * phi
b = (1 - mu) * phi
y = np.random.beta(a,b,N)
db = pd.DataFrame(np.array([y,x_1]).T)
# I export the simulated database to be able to perform the Bayesian analysis in JAGS with exactly the same data
#db.to_csv (r'/Users/estevaoprado/Maynooth University/Andrew Parnell - CDA_PhD/project_4/Python Code/Beta_regression.csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path
y = y.reshape(N,1)
X = np.array([x_0, x_1]).T

# Dimensions ------------------------------------------------------------------
y.shape
d = X.shape

# Prior distributions and the values of the hyperparameters
#------------------------------------------------------------------------------
# Beta ~ N(m, V),
m = np.full((d[1]),0).reshape(d[1],1)
V = np.identity(d[1]) * 100
V_1 = np_alg.inv(V)

# Sigma ~ U(a, b)
a = 0
b = 100

# Starting points
#------------------------------------------------------------------------------
phi_ = 10
beta_ = np.full((d[1]),0).reshape(d[1],1)
sbeta = []
sphi = []
mprop = np.identity(d[1])* 0.0001 # variance of the proposal distribution for beta

def func_mu(beta):
    return(np.exp(X @ beta)/(1+np.exp(X @ beta)))

# Since the proposal distribution for phi is a Gamma, the ideia is as following:
    # mean = alpha/beta
    # var = alpha/beta^2; Thus
    # alpha = mean^2 / var;
    # beta = mean/var.    
# Consider that phi_ = mean and var = var, where var is set up below.
    
var = 0.5 # Controlling the variance of the proposal distribution for phi_. Thus, it is possible to control the acceptance rate of phi_.

# Below, the implementation is slightly different due to the parametrisation of the Gamma distribution in Python.

for i in tqdm(range(TotIter)):
    
    # Proposal distribution for beta (multivariate Normal)
    betac = np.random.multivariate_normal(list(itertools.chain(*beta_)), mprop, 1).T
    
    mu_ = func_mu(beta_)
    muc = func_mu(betac)
    
    # Computing the ratio stated in the Metropolis-Hastings algorithm
    ratio_beta = ((- sum(np.log(Gamma(muc * phi_))) - sum(np.log(Gamma((1 - muc) * phi_))) + sum((muc * phi_ - 1) * np.log(y)) + sum(((1 - muc) * phi_ - 1) * np.log(1 - y)) - 0.5 * (betac - m).T @ V_1 @(betac - m)) - 
                  (- sum(np.log(Gamma(mu_ * phi_))) - sum(np.log(Gamma((1 - mu_) * phi_))) + sum((mu_ * phi_ - 1) * np.log(y)) + sum(((1 - mu_) * phi_ - 1) * np.log(1 - y)) - 0.5 * (beta_ - m).T @ V_1 @(beta_ - m)))                       
        
    # Accept/reject step for betac (beta candidate)
    if (np.random.uniform(0, 1, 1) < np.exp(ratio_beta)):
        beta_ = betac
        if (i > Burn):
            j_beta = j_beta + 1

    # Proposal distribution for phi
    phic = gamma.rvs((phi_**2)/var, scale=1/(phi_ / var), size=1, random_state=None)

    # Computing the ratio stated in the Metropolis-Hastings algorithm
    ratio_phi = ((N * np.log(Gamma(phic)) - sum(np.log(Gamma(mu_ * phic))) - sum(np.log(Gamma((1 - mu_) * phic))) + sum((mu_* phic - 1) * np.log(y)) + sum(((1 - mu_) * phic - 1) * np.log(1 - y)) - np.log(b-a) + gamma.pdf(phi_, (phic**2)/var, scale = 1/(phic / var))) - 
                 (N * np.log(Gamma(phi_)) - sum(np.log(Gamma(mu_ * phi_))) - sum(np.log(Gamma((1 - mu_) * phi_))) + sum((mu_* phi_ - 1) * np.log(y)) + sum(((1 - mu_) * phi_ - 1) * np.log(1 - y)) - np.log(b-a) + gamma.pdf(phic, (phi_**2)/var, scale = 1/(phi_ / var))))
    
    # Accept/reject step for phic (phi candidate)
    if (np.random.uniform(0, 1, 1) < np.exp(ratio_phi)):
        phi_ = phic
        if (i > Burn):
            j_phi = j_phi + 1

    # Storing the results
    if (i >= Burn):
        sbeta.append(beta_.tolist())
        sphi.append(phi_)

sbeta = np.asarray(sbeta)
sphi = np.asarray(sphi)

# Acceptance rate -------------------------------------------------------------
Acc_rate_beta = j_beta/Npost; Acc_rate_beta
Acc_rate_phi = j_phi/Npost; Acc_rate_phi

# Plot of the intercept --------
mt.pyplot.plot(sbeta[:,0])
mt.pyplot.xlabel('Iterations', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Intercept', fontsize=18)
mt.pyplot.grid(True)
mt.pyplot.show()

# Plot of the beta --------
mt.pyplot.plot(sbeta[:,1])
mt.pyplot.xlabel('Iterations', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Beta', fontsize=18)
mt.pyplot.grid(True)
mt.pyplot.show()

# Plot of the phi --------
mt.pyplot.plot(sphi[:,0])
mt.pyplot.xlabel('Iterations', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Phi', fontsize=18)
mt.pyplot.grid(True)
mt.pyplot.show()

# Checking the posterior estimates --------------------------------------------
alpha_est = np.mean(sbeta[:,0]);alpha_est
beta1_est = np.mean(sbeta[:,1]);beta1_est
sphi_est = np.mean(sphi);sphi_est

# Comparing the posterior estimates and true values by using sum of the absolute error -----
logit_mu_est = alpha_est + beta1_est * x_1
mu_est =  np.exp(logit_mu_est)/(1+np.exp(logit_mu_est)) # inverse logit

sum(abs(mu_est.reshape(N,1) - y)) # ESTIMATED - Sum of the absolute error
sum(abs(mu.reshape(N,1)- y))      # TRUE - Sum of the absolute error

# Density plots ---------------------------------------------------------------
n_groups = 3
means_MCMC = (alpha_est, beta1_est, sphi_est)
means_JAGS = (-1.12, 0.22, 4.97)
sd_JAGS = [0.05117907, 0.00901529, 0.2067916]

# Plot of the intercept --------
sns.distplot(sbeta[:,0], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[0],sd_JAGS[0],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Intercept - Density plot', fontsize=18)

# Plot of the beta --------
sns.distplot(sbeta[:,1], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[1],sd_JAGS[1],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Beta - Density plot', fontsize=18)

# Plot of the phi --------
sns.distplot(sphi[:,0], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[2],sd_JAGS[2],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Phi - Density plot', fontsize=18)

# Comparing our MCMC by hand versus JAGS -------------------------------------- 

fig, ax = mt.pyplot.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = mt.pyplot.bar(index, means_MCMC, bar_width,
                 alpha=opacity,
                 color='b',
                 label='MCMC by hand')
 
rects2 = mt.pyplot.bar(index + bar_width, means_JAGS, bar_width,
                 alpha=opacity,
                 color='g',
                 label='JAGS')
 
mt.pyplot.xlabel('Parameters', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Comparison', fontsize=18)
mt.pyplot.xticks(index + bar_width, ('Alpha', 'Beta', 'Phi'))
mt.pyplot.legend()
mt.pyplot.tight_layout()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%.2f'%height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
mt.pyplot.show()

# Boxplot ---------------------------------------------------------------------

a = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Intercept',Npost), 'par' : np.asanyarray(sbeta[:,0]).reshape(Npost,)})
b1 = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Beta1',Npost), 'par' : np.asanyarray(sbeta[:,1]).reshape(Npost,)})
b2 = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Phi',Npost), 'par' : np.asanyarray(sphi[:,0]).reshape(Npost,)})
aJ = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Intercept',Npost), 'par' : np.random.normal(means_JAGS[0], sd_JAGS[0], Npost)})
b1J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Beta1',Npost), 'par' : np.random.normal(means_JAGS[1], sd_JAGS[1], Npost)})
b2J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Phi',Npost), 'par' : np.random.normal(means_JAGS[2], sd_JAGS[2], Npost)})

df=a.append(b1).append(b2).append(aJ).append(b1J).append(b2J)

#sns.boxplot(x='Parameters', y='par', hue='Source', data=df, palette="Set3", dodge=True)
#sns.violinplot(x='Parameters', y='par', hue='Source', data=df, palette="Set3", dodge=True)

a = a.append(aJ)
b1 = b1.append(b1J)
b2 = b2.append(b2J)

sns.boxplot(x='Parameter', y='par', hue='Source', data=a, palette="Set3", dodge=True)
#sns.violinplot(x='Parameter', y='par', hue='Source', data=a, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Intercept', fontsize=18)

sns.boxplot(x='Parameter', y='par', hue='Source', data=b1, palette="Set3", dodge=True)
#sns.violinplot(x='Parameter', y='par', hue='Source', data=b1, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Beta', fontsize=18)

sns.boxplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
#sns.violinplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Phi', fontsize=18)
