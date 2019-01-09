# --------------------------------------------------------------- #
# Description: Metropolis-Hastings for Bayesian linear regression #
#              with a Laplace as prior for betas and an Uniform   #
#              for sigma2                                         #
# Author: Estevão Prado                                           #
# Last modification: 18/12/2018                                   #
# --------------------------------------------------------------- #
import numpy as np
import matplotlib as mt
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import gamma

# Simulate data ---------------------------------------------------------------
np.random.seed(123)
Npost = 10000
Burn = int(Npost*0.5)
TotIter = Npost + Burn
js = 0
jb = 0
N = 1000
alpha = -2
beta = 3
sigma = 3

x_0 = np.repeat(1,N)
x_1 = np.random.normal(loc=0, scale=1, size=N) # Draw samples from a uniform distribution
y = np.random.normal(alpha + x_1*beta, sigma, size=N) # Draw random samples from a normal (Gaussian) distribution.
db = pd.DataFrame(np.array([y,x_0,x_1]).T)
# I export the simulated database to be able to perform the Bayesian analysis in JAGS with exactly the same data
#db.to_csv (r'/Users/estevaoprado/Maynooth University/Andrew Parnell - CDA_PhD/project_4/Python Code/Linear_regression.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
y = y.reshape(N,1)
X = np.array([x_0, x_1]).T

# Dimensions ------------------------------------------------------------------
y.shape
d = X.shape

# Prior distributions and the values of the hyperparameters
#------------------------------------------------------------------------------
# Beta ~ Normal(mi, vi) 
mi = np.repeat(0,2).reshape(2,1)
vi = 100

# Sigma2 ~ U(a, b)
a = 0
b = 10

# Starting points
#------------------------------------------------------------------------------
beta_ = np.full((d[1]),0).reshape(d[1],1)
sigma2_ = 1
sbeta = []
ssigm = []
mprop = np.identity(d[1])* 0.001 # variance of the proposal distribution for beta

# Since the proposal distribution for sigma2 is a Gamma, the ideia is as following:
    # mean = alpha/beta
    # var = alpha/beta^2; Thus
    # alpha = mean^2 / var;
    # beta = mean/var.    
# Consider that sigma2_ = mean and var = var, where var is set up below.
    
var = 0.5 # Controlling the variance of the proposal distribution for sigma2. Thus, it is possible to control the acceptance rate of sigma2.

# Below, the implementation is slightly different due to the parametrisation of the Gamma distribution in Python.

for i in tqdm(range(TotIter)):
    
    # Proposal distribution for sigma2 
    sigma2c = gamma.rvs((sigma2_**2)/var, scale=1/(sigma2_ / var), size=1, random_state=None)
    
    # Computing the ratio in the Metropolis-Hastings
    ratio_sigma2 = (((-N/2) * np.log(sigma2c) - 0.5/sigma2c*(y - (X @ beta_)).T @(y - (X @ beta_)) - np.log(b - a) + 1/sigma2_ + gamma.pdf(sigma2_, (sigma2c**2)/var, scale = 1/(sigma2c / var))) -
                    ((-N/2) * np.log(sigma2_) - 0.5/sigma2_*(y - (X @ beta_)).T @(y - (X @ beta_)) - np.log(b - a) + 1/sigma2c + gamma.pdf(sigma2c, (sigma2_**2)/var, scale = 1/(sigma2_ / var))))
                    
    # Accept/reject step for sigma2c (sigma2 candidate)
    if (np.random.uniform(0, 1, 1) < np.exp(ratio_sigma2)):
        sigma2_ = sigma2c
        if (i > Burn):
            js = js + 1

    # Proposal distribution for beta (multivariate Normal)
    betac = np.random.multivariate_normal(list(itertools.chain(*beta_)), mprop,1).T

    # Computing the ratio stated in the Metropolis-Hastings algorithm
    ratio_beta =  ((-0.5/sigma2_*(y - (X @ betac)).T @(y - (X @ betac))  - 0.5/vi*sum(abs(betac - mi)**2)) -
                   (-0.5/sigma2_*(y - (X @ beta_)).T @(y - (X @ beta_))  - 0.5/vi*sum(abs(beta_ - mi)**2)))

    # Accept/reject step for betac (beta candidate)
    if (np.random.uniform(0, 1, 1) < np.exp(ratio_beta)):
        beta_ = betac
        if (i > Burn):
            jb = jb + 1

    # Storing the results
    if (i >= Burn):
        sbeta.append(beta_.tolist())
        ssigm.append(sigma2_.tolist())

sbeta = np.asarray(sbeta)
ssigm = np.asarray(ssigm)

# Acceptance rate ------------- 
Acceptance_rate_betas = jb/Npost; Acceptance_rate_betas
Acceptance_rate_sigma2 = js/Npost; Acceptance_rate_sigma2

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

# Plot of the sigma2 --------
mt.pyplot.plot(ssigm)
mt.pyplot.xlabel('Iterations', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Sigma2', fontsize=18)
mt.pyplot.grid(True)
mt.pyplot.show()

# Checking the posterior estimates ------
alpha_est = np.mean(sbeta[:,0]);alpha_est
beta1_est = np.mean(sbeta[:,1]);beta1_est
sigma_est = np.sqrt(np.mean(ssigm));sigma_est

# Comparing the posterior estimates and true values by using sum of the absolute error -----
y_est = alpha_est + beta1_est * x_1
y_tru = alpha     + beta      * x_1

sum(abs(y_est.reshape(N,1) - y)) # ESTIMATED - Sum of the absolute error
sum(abs(y_tru.reshape(N,1) - y)) # TRUE - Sum of the absolute error

# Density plot ----------------------------------------------------------------
n_groups = 3
means_MCMC = (alpha_est, beta1_est, sigma_est)
means_JAGS = (-1.98, 2.90, 2.87)
sd_JAGS = [0.09098112, 0.09166854, 0.06481725]

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
mt.pyplot.title('Beta1 - Density plot', fontsize=18)

# Plot of the sigma --------
sns.distplot(np.sqrt(ssigm[:,0]), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'MCMC by hand')

sns.distplot(np.random.normal(means_JAGS[2],sd_JAGS[2],Npost), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'JAGS')
mt.pyplot.title('Sigma - Density plot', fontsize=18)

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
b2 = pd.DataFrame({'Source': np.repeat('MCMC by hand',Npost) , 'Parameter' : np.repeat('Sigma',Npost), 'par' : np.asanyarray(np.sqrt(ssigm[:,0])).reshape(Npost,)})
aJ = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Intercept',Npost), 'par' : np.random.normal(means_JAGS[0], sd_JAGS[0], Npost)})
b1J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Beta1',Npost), 'par' : np.random.normal(means_JAGS[1], sd_JAGS[1], Npost)})
b2J = pd.DataFrame({'Source': np.repeat('JAGS',Npost) , 'Parameter' : np.repeat('Sigma',Npost), 'par' : np.random.normal(means_JAGS[2], sd_JAGS[2], Npost)})

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
mt.pyplot.title('Beta', fontsize=18)

sns.boxplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
sns.violinplot(x='Parameter', y='par', hue='Source', data=b2, palette="Set3", dodge=True)
mt.pyplot.xlabel('Parameter', fontsize=14)
mt.pyplot.ylabel('Values', fontsize=14)
mt.pyplot.title('Sigma', fontsize=18)
