# Header ------------------------------------------------------------------

# A Poisson space-time conditional autoregressive (CAR) model in JAGS

# See the JAGS CAR model for details of the CAR model
# Andrew Parnell

# In this file I:
# 1. Write out the maths for the st-CAR model
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
# y_it = count (response variable) at location i, i = 1, ..., N, at time t = 1, ..., T
# lambda_it = estimated mean of y at location i at time t
# b_t = random effect of time at time t
# X_i = vector of explanatory variables at location i, length p, assumed not time structured, but this could be changed
# phi_i = spatially structured random effect at location i
# offset_i = offset for location i (not used in subsequent modelling)
# D is a diagonal matrix with diagonal elements m_i which are the number of neighbours of location i
# W is the adjacency matrix with w_{ij} = 1 if locations i and j are neighbours
# alpha is a (0, 1) parameter that controls spatial dependence. alpha = 0 corresponds to no spatial dependence
# tau is an overall precision parameter
# gamma is the temporal autoregressive parameter

# Likelihood
# y_it ~ Poisson(lambda_it)
# log(lambda_it) = X_i^T beta + phi_i + offset_i + gamma_t
# Prior
# phi ~ MVN(0, tau D (I - alpha * D^{-1} W ) ^{-1} )
# beta_k ~ N(0, 1)
# tau ~ cauchy(0, 1)
# alpha ~ uniform(0, 1)
# b_t ~ normal(0, sigma_b^2)
# sigma_b ~ cauchy(0, 1)


# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 40 # Locations
T = 20
p = 2
X = matrix(runif(N*p), ncol = p, nrow = N)
alpha = 0.8
gamma = 0.15
tau = 1
sigma_b = 2
b = rnorm(T, 0, sigma_b)
beta = rnorm(p)
# Create random neighbours
set.seed(125)
W1 = matrix(rbinom(N*N, size = 1, prob = 0.6), nrow = N, ncol = N)
# Make is symmetric
W = round(W1%*%t(W1)/max(W1%*%t(W1)))
D = diag(rowSums(W))
D_inv = diag(1/rowSums(W))
I = diag(N)
P = tau*D%*%(I - alpha * D_inv%*% W)
phi = mvrnorm(1, rep(0, N), solve(P))
y = log_lambda = matrix(NA, nrow = T, ncol = N)
for(t in 1:T) {
  log_lambda[t,] = X%*%beta + phi + b[t]
  y[t,] = rpois(N, exp(X%*%beta + phi + b[t]))
}
tail(y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for(t in 1:T) { # Loop over time
    for (i in 1:N) { # Loop over sites
      y[t,i] ~ dpois(lambda[t,i])
      log(lambda[t,i]) = inprod(X[i,], beta) + phi[i] + b[t]
    }
    b[t] ~ dnorm(0, sigma_t^-2)
  }
  phi ~ dmnorm(zeros, P)
  P = tau * D %*% (I - alpha * D_inv %*% W)
  # Priors

  alpha ~ dunif(0, 1)
  for(j in 1:p) {
    beta[j] ~ dnorm(0, 1)
  }
  tau ~ dt(0, 1, 1)T(0,)
  sigma_t ~ dt(0, 1, 1)T(0,)
}
'

# Set up the data
model_data = list(N = N, y = y, X = X,
                  zeros = rep(0, N),
                  p = p,
                  D_inv = D_inv,
                  D = D,
                  W = W,
                  T = T,
                  I = I)

# Choose the parameters to watch
model_parameters =  c("alpha", "tau", "beta", 'sigma_t', 'b')

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

plot(model_run)
stop()

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


