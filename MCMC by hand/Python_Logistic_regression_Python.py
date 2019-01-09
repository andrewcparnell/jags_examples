# ------------------------------------------------------------------ #
# Description: Metropolis-Hastings for Bayesian logistic regression  #
#              with a Multivariate Normal as prior for betas.        #
# Author: Estevão Prado                                              #
# Last modification: 18/12/2018                                      #
# ------------------------------------------------------------------ #

import numpy as np
import numpy.linalg as np_alg
import matplotlib as mt
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# Simulate data ---------------------------------------------------------------

np.random.seed(123)
Npost = 10000
Burn = int(Npost*0.5)
TotIter = Npost + Burn
j = 0
k = 1
T = 1000;
x_0 = np.repeat(1,T)
x_1 = np.random.uniform(0, 10, T)
x_2 = np.random.uniform(0, 10, T)
alpha = 1
beta_1 = 0.5
beta_2 = -0.5
logit_p = alpha + beta_1 * x_1 + beta_2 * x_2
p =  np.exp(logit_p)/(1+np.exp(logit_p)) # inverse logit
y = np.random.binomial(1,p,T)

db = pd.DataFrame(np.array([y,x_1,x_2]).T)
# I export the simulated database to be able to perform the Bayesian analysis in JAGS with exactly the same data
#db.to_csv (r'/Users/estevaoprado/Maynooth University/Andrew Parnell - CDA_PhD/project_4/Python Code/Logistic_regression.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
y = y.reshape(T,1)

X = np.array([x_0, x_1,x_2]).T
d = X.shape

# Prior distributions and the values of the hyperparameters
#------------------------------------------------------------------------------
# Beta ~ N(mu, Sigma),

mu = np.full((d[1]),0).reshape(d[1],1)
Sigma = np.identity(d[1]) * 100
Sigma_inv = np_alg.inv(Sigma)

# Starting points
#------------------------------------------------------------------------------

beta_ = np.full((d[1]),-10).reshape(d[1],1)
sbeta = []
mprop = np.identity(d[1])* 0.001 # variance of the proposal distribution

for i in tqdm(range(TotIter)):
    
    # Proposal distribution for beta (multivariate Normal)
    betac = np.random.multivariate_normal(list(itertools.chain(*beta_)), mprop,1).T

    # Computing the ratio stated in the Metropolis-Hastings algorithm
    ratio = ((-k * sum(np.log(1 + np.exp(X @ betac)))  + sum(y * X @ betac) - (1/2) * (betac - mu).T @ Sigma_inv @ (betac - mu)) -
            ((-k * sum(np.log(1 + np.exp(X @ beta_)))) + sum(y * X @ beta_) - (1/2) * (beta_ - mu).T @ Sigma_inv @ (beta_ - mu)))

    # Accept/reject step for betac (beta candidate)
    if (np.random.uniform(0, 1, 1) < np.exp(ratio)):
        beta_ = betac
        if (i > Burn):
            j = j + 1

    # Storing the results
    if (i >= Burn):
        sbeta.append(beta_.tolist())

sbeta = np.asarray(sbeta)

# Acceptance rate ------------- 
Acceptance_rate = j/Npost; Acceptance_rate

# Plot of the intercept --------
mt.pyplot.plot(sbeta[:,0])
mt.pyplot.xlabel('Iterations', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Intercept', fontsize=18)
mt.pyplot.grid(True)
mt.pyplot.show()

# Plot of the beta1 --------
mt.pyplot.plot(sbeta[:,1])
mt.pyplot.xlabel('Iterations')
mt.pyplot.ylabel('Values')
mt.pyplot.title('Beta1')
mt.pyplot.grid(True)
mt.pyplot.show()

# Plot of the beta2 --------
mt.pyplot.plot(sbeta[:,2])
mt.pyplot.xlabel('Iterations')
mt.pyplot.ylabel('Values')
mt.pyplot.title('Beta2')
mt.pyplot.grid(True)
mt.pyplot.show()

# Checking the posterior estimates ------
alpha_est = np.mean(sbeta[:,0])
beta1_est = np.mean(sbeta[:,1])
beta2_est = np.mean(sbeta[:,2])

# Comparing the posterior estimates and true values by using sum of the absolute error -----
logit_p_est = alpha_est + beta1_est * x_1 + beta2_est * x_2
p_est =  np.exp(logit_p_est)/(1+np.exp(logit_p_est))

sum(abs(p_est.reshape(T,1) - y)) # ESTIMATED - Sum of the absolute error
sum(abs(p.reshape(T,1) - y))     # TRUE - Sum of the absolute error

# Density plot ----------------------------------------------------------------
n_groups = 3
means_MCMC = (alpha_est, beta1_est, beta2_est)
means_JAGS = (1.05, 0.57, -0.55)
sd_JAGS = [0.2277933, 0.04452061, 0.04181527]

# Plot of the intercept --------
sns.distplot(sbeta[:,0], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[0],sd_JAGS[0],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Intercept - Density plot', fontsize=18)

# Plot of the beta1 --------
sns.distplot(sbeta[:,1], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[1],sd_JAGS[1],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Beta1 - Density plot', fontsize=18)

# Plot of the beta2 --------
sns.distplot(sbeta[:,2], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[2],sd_JAGS[2],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Beta2 - Density plot', fontsize=18)

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
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.2f'%height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
mt.pyplot.show()

# Boxplot ---------------------------------------------------------------------

a = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Intercept',Npost), 'par' : np.asanyarray(sbeta[:,0]).reshape(Npost,)})
b1 = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Beta1',Npost), 'par' : np.asanyarray(sbeta[:,1]).reshape(Npost,)})
b2 = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Beta2',Npost), 'par' : np.asanyarray(sbeta[:,2]).reshape(Npost,)})
aJ = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Intercept',Npost), 'par' : np.random.normal(means_JAGS[0], sd_JAGS[0], Npost)})
b1J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Beta1',Npost), 'par' : np.random.normal(means_JAGS[1], sd_JAGS[1], Npost)})
b2J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Beta2',Npost), 'par' : np.random.normal(means_JAGS[2], sd_JAGS[2], Npost)})

df=a.append(b1).append(b2).append(aJ).append(b1J).append(b2J)

#sns.boxplot(x='Parameters', y='par', hue='Source', data=df, palette="Set3", dodge=True)
#sns.violinplot(x='Parameters', y='par', hue='Source', data=df, palette="Set3", dodge=True)

a = a.append(aJ)
b1 = b1.append(b1J)
b2 = b2.append(b2J)

sns.boxplot(x='Parameter', y='par', hue='Source', data=a, palette="Set3", dodge=True)
sns.violinplot(x='Parameter', y='par', hue='Source', data=a, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Intercept', fontsize=18)

sns.boxplot(x='Parameter', y='par', hue='Source', data=b1, palette="Set3", dodge=True)
sns.violinplot(x='Parameter', y='par', hue='Source', data=b1, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Beta1', fontsize=18)

sns.boxplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
sns.violinplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Beta2', fontsize=18)
