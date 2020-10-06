# Header ------------------------------------------------------------------

# A JAGS model for data with compositional covariates

# Compositional covariates
# Andrew Parnell & Uli

# The usual way people fit models with compositional covariates is to not put them directly (as this breaks e.g. lm as they are perfectly correlated) but instead to use ilr or clr and put the reduced dimension version into a regression model. This causes problems with interpretation as its not clear what the posterior parameter values mean. This code tries to implement a ridge regression version which accepts raw compositions, but it might not work.

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
setwd("~/GitHub/jags_examples/jags_scripts")
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# y_i = alpha + beta_1 x_1i + beta_2 x_2i + beta_3 x_3i + epsilon_i
# where x_1i + x_2i + x_3i = 1 for all data points
# Notation
# y_i = response
# x_1i,...,x_3i = covariates (compositional)
# alpha, beta_1, ... beta_3 parameters
# epsilon_i ~ N(0, sigma^2) residuals and residual variance

# Likelihood
# y_i ~ N(alpha + beta_1 x_1i + beta_2 x_2i + beta_3 x_3i, sigma^2)
# Priors
# beta_j ~ N(mu_beta, sigma_beta^2) # Tries to tie the betas together
# Flat of vague on all other parameters

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 100
K = 3
x = matrix(NA, ncol = K, nrow = N)
set.seed(124)
for(i in 1:N) {
  x[i,] = rgamma(K,1,1)
  x[i,] = x[i,]/sum(x[i,])
}
alpha = rnorm(1)
beta = rnorm(3, 0, sd = 5)
sigma = runif(1)
y = rnorm(N, mean = alpha + x%*%beta, sd = sigma)

pairs(cbind(x, y))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(alpha + beta[1]*x[i,1] + beta[2]*x[i,2] + beta[3]*x[i,3],
                  sigma^-2)
    y_sim[i] ~ dnorm(alpha + beta[1]*x[i,1] + beta[2]*x[i,2] + beta[3]*x[i,3],
                  sigma^-2)
  }
  # Priors
  for (j in 1:3) {
    beta[j] ~ dnorm(0, 1)
  }
  alpha ~ dnorm(0, 100^-2)
  sigma ~ dt(0, 10^-2, 1)T(0,)
}
'

# Set up the data
model_data = list(N = N, y = y, x = x)

# Choose the parameters to watch
model_parameters =  c("alpha","beta","sigma","y_sim")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Simulated results -------------------------------------------------------

plot(model_run)

y_sim = model_run$BUGSoutput$mean$y_sim
plot(y, y_sim)
abline(a=0, b=1)

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


