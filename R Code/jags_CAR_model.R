# Header ------------------------------------------------------------------

# A Poisson conditional autoregressive (CAR) model in JAGS

# Much of the maths taken from https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
# Note that that web page has some really cool computational efficiency gains in it which I've not implemented here
# Andrew Parnell

# In this file I:
# 1. Write out the maths for hte CAR model
# 2. Simulate some data from it
# 3. Show the JAGS code
# 4. Fit the JAGS code to the simulated data

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)
library(MASS) # To simulate from MVN

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_i = count (response variable) at location i, i = 1, ..., N
# lambda_i = estimated mean of y at location i
# X_i = vector of explanatory variables at location i, length p
# phi_i = spatially structured random effect at location i
# offset_i = offset for location i (not used in subsequent modelling)
# D is a diagonal matrix with diagonal elements m_i which are the number of neighbours of location i
# W is the adjacency matrix with w_{ij} = 1 if locations i and j are neighbours
# alpha is a (0, 1) parameter that controls spatial dependence. alpha = 0 corresponds to no spatial dependence
# tau is an overall precision parameter

# Likelihood
# y_i ~ Poisson(lambda_i)
# log(lambda_i) = X_i^T beta + phi_i + offset_i
# Prior
# phi ~ MVN(0, tau D (I - alpha * D^{-1} W ) ^{-1} )
# beta_k ~ N(0, 1)
# tau ~ cauchy(0, 1)
# alpha ~ uniform(0, 1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 40 # Locations
p = 2
X = matrix(runif(N*p), ncol = p, nrow = N)
alpha = 0.8
tau = 1
beta = rnorm(p)
# Create random neighbours
set.seed(124)
W1 = matrix(rbinom(N*N, size = 1, prob = 0.6), nrow = N, ncol = N)
# Make is symmetric
W = round(W1%*%t(W1)/max(W1%*%t(W1)))
D = diag(rowSums(W))
D_inv = diag(1/rowSums(W))
I = diag(N)
P = tau*D%*%(I - alpha * D_inv%*% W)
phi = mvrnorm(1, rep(0, N), solve(P))
y = rpois(N, exp(X%*%beta + phi))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dpois(lambda[i])
    log(lambda[i]) = inprod(X[i,], beta) + phi[i]
  }
  phi ~ dmnorm(zeros, P)
  P = tau * D %*% (I - alpha * D_inv %*% W)
  # Priors
  alpha ~ dunif(0, 1)
  for(j in 1:p) {
    beta[j] ~ dnorm(0, 1)
  }
  tau ~ dt(0, 1, 1)T(0,)
}
'

# Set up the data
model_data = list(N = N, y = y, X = X,
                  zeros = rep(0, N),
                  p = p,
                  D_inv = D_inv,
                  D = D,
                  W = W,
                  I = I)

# Choose the parameters to watch
model_parameters =  c("alpha", "tau", "beta")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

plot(model_run)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


