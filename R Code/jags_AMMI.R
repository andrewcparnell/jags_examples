# Header ------------------------------------------------------------------

# An Additive Main effects and Multiplicative Interaction (AMMI) effects model

# A Bayesian version of the AMMI model as specified here: https://link.springer.com/content/pdf/10.1007/s13253-014-0168-z.pdf (JosseE et al JABES 2014)
# Andrew Parnell / Danilo Sarti

# In this file we simulate from the AMMI model specified in the paper above, then fit it using JAGS. Finally we present a worked example
# There are two jags code parts below. The first is for a single interaction decomposition effect. The second is for a general AMMI model which is more realistic

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)
library(ggplot2)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file

# Notation
# Y_ij = response (e.g. yield) for genotype i and environment j, i = 1, ..., I genotypes and j = 1, ..., J environments
# mu is the grand mean
# alpha_i is the genotype effect
# beta_j is the environment effect
# lambda_q is the q-th eigenvalue q = 1,.., Q of the interaction matrix
# Q is the number of components used to model the interaction. Usually Q is fixed at a small number, e.g. 2
# gamma_{iq} is the interaction effect for the q-th eigenvector for genotype i
# delta_{iq} is the interaction effect for the q-th eigenvector for environment j
# E_{ij} is a residual term with E_{ij} ~ N(0, sigma^2_E)
# Usually these models have quite complicated restrictions on the gamma/delta/lambda values but Josse et al show that these are not fully necessary

# Likelihood
# Y_{ij} ~ N(mu_{ij}, sigma^2_E)
# with
# mu_{ij} = mu + alpha_i + beta_j + sum_{q=1}^Q lambda_q*gamma_iq*delta_jq

# Priors
# mu ~ N(m, s_mu^2)
# alpha_i ~ N(0, s_alpha^2)
# beta_j ~ N(0, s_beta^2)
# lambda_q ~ N+(0, s_lambda^2) with lambda_1 < lambda_2 < ... < lambda_Q
# Note: N+ means positive-only normal
# gamma_{1q} ~ N+(0, 1)
# gamma_{iq} ~ N(0, 1)
# delta_{jq} ~ N(0, 1)
# sigma_E ~ U(0, S_ME)
# All of the above hyper-parameter values are fixed

# Simulate data -----------------------------------------------------------

# We will follow the simulation strategy detailed in Section 3.1 of the Josse et al paper

# Specify fixed values
Q = 1 # Number of components
I = 5 # Number of genotypes
J = 9 # Number of environments
N = I*J # Total number of obs
m = 90
s_mu = 20
s_alpha = 10
s_beta = 10
s_lambda = 10
S_ME = 10

# Some further fixed values
mu = 100
sigma_E = 3/2 # Not sure why S_ME was specified if they're also giving sigma_E
alpha = c(-1, -1, 0, 1, 1)
beta = -4:4
lambda_1 = 12
gamma = seq(2, -2)/sqrt(10)
delta = c(0.5, 0.5, rep(0, 5), -0.5, -0.5)

# Now simulate the values
set.seed(123)
G_by_E = expand.grid(1:I, 1:J)
mu_ij = mu + alpha[G_by_E[,1]] + beta[G_by_E[,2]] + lambda_1 * gamma[G_by_E[,1]] * delta[G_by_E[,2]]
Y = rnorm(N, mu_ij, sigma_E)

# Can create some plots
qplot(x = G_by_E[,1], y = Y, geom = 'boxplot', group = G_by_E[,1], xlab = 'Genotype')
qplot(x = G_by_E[,2], y = Y, geom = 'boxplot', group = G_by_E[,2], xlab = 'Environment')


# Q = 1 model -------------------------------------------------------------

# This is the simple Q = 1 model - only here for understanding
# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (k in 1:N) {
    Y[k] ~ dnorm(mu[k], sigma_E^-2)
    mu[k] = mu_all + alpha[genotype[k]] + beta[environment[k]] + lambda_1 * gamma[genotype[k]] * delta[environment[k]]
  }

  # Priors
  mu_all ~ dnorm(0, s_mu^-2) # Prior on grand mean
  for(i in 1:I) {
    alpha[i] ~ dnorm(0, s_alpha^-2) # Prior on genotype effect
  }
  gamma[1] ~ dnorm(0, 1)T(0,) # First one is restriced to be positive
  for(i in 2:I) {
    gamma[i] ~ dnorm(0, 1) # Prior on genotype interactions
  }
  for(j in 1:J) {
    beta[j] ~ dnorm(0, s_beta^-2) # Prior on environment effect
    delta[j] ~ dnorm(0, 1) # Prior on environment interactions
  }
  # Prior on first (and only) eigenvalue
  lambda_1 ~ dnorm(0, s_lambda^-2)T(0,)
  # Prior on residual standard deviation
  sigma_E ~ dunif(0, S_ME)
}
'

# Set up the data
model_data = list(N = N,
                  Y = Y,
                  I = I,
                  J = J,
                  genotype = G_by_E[,1],
                  environment = G_by_E[,2],
                  s_mu = s_mu,
                  s_alpha = s_alpha,
                  s_beta = s_beta,
                  s_lambda = s_lambda,
                  S_ME = S_ME)

# Choose the parameters to watch
model_parameters =  c("alpha", "beta", "lambda_1", "gamma", "delta",
                      'sigma_E')

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Look at the results
plot(model_run)
# Can compare these to the true values - looks ok but not amazing. Seems to have got things right in the main
# Convergence is good


# Second model - general Q ------------------------------------------------

model_code = '
model
{
  # Likelihood
  for (k in 1:N) {
    Y[k] ~ dnorm(mu[k], sigma_E^-2)
    mu[k] = mu_all + alpha[genotype[k]] + beta[environment[k]] + sum(lambda * gamma[genotype[k],1:Q] * delta[environment[k],1:Q])
  }

  # Priors
  mu_all ~ dnorm(0, s_mu^-2) # Prior on grand mean
  for(i in 1:I) {
    alpha[i] ~ dnorm(0, s_alpha^-2) # Prior on genotype effect
  }
  for(j in 1:J) {
    beta[j] ~ dnorm(0, s_beta^-2) # Prior on environment effect
  }

  # Priors on gamma
  for(q in 1:Q) {
    gamma[1, q] ~ dnorm(0, 1)T(0,) # First one is restriced to be positive
    for(i in 2:I) {
      gamma[i, q] ~ dnorm(0, 1) # Prior on genotype interactions
    }
  }

  # Priors on delta
  for(q in 1:Q) {
    for(j in 1:J) {
      delta[j, q] ~ dnorm(0, 1) # Prior on environment interactions
    }
  }

  # Prior on eigenvalues
  for(q in 1:Q) {
    lambda_raw[q] ~ dnorm(0, s_lambda^-2)T(0,)
  }
  lambda = sort(lambda_raw)

  # Prior on residual standard deviation
  sigma_E ~ dunif(0, S_ME)
}
'

# Set up the data
model_data = list(N = N,
                  Y = Y,
                  I = I,
                  J = J,
                  Q = 2, # Set Q to be 2 even though the simulation was for Q = 1
                  genotype = G_by_E[,1],
                  environment = G_by_E[,2],
                  s_mu = s_mu,
                  s_alpha = s_alpha,
                  s_beta = s_beta,
                  s_lambda = s_lambda,
                  S_ME = S_ME)

# Choose the parameters to watch
model_parameters =  c("alpha", "beta", "lambda", "gamma", "delta",
                      'sigma_E')

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Plot the results
plot(model_run)
# Seems to work ok - might need a longer run to get better convergence

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


