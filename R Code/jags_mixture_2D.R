# Header ------------------------------------------------------------------

# A 2D mixture model in JAGS with a fixed number of groups
# Andrew Parnell

# This model fits a mixture of bivariate normal distributions to the data given a fixed number of groups. It might be useful for clustering or other applications where standard probability distributions are not suitable
# This model assumes the mixtures are on the means, but all standard deviations are equal

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)
library(mvtnorm) # for multivariate normal distribution

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_i = M-vector observation i, i= 1, .., N
# mu_g = M-vector mean for group g - these are sorted (in the first dim) to avoid label switching
# Sigma = Covariance matrix of y
# z_i = group member for observation i, z_i = 1, ..., G
# pi_{ig} = probability that observation i belongs to group g. \sum_g \pi_{ig} = 1
# G = total number of groups (fixed)
# Likelihood
# y_i ~ MVN(mu[z_i, 1:M], Sigma[1:M, 1:M])
# z_i ~ categorical(pi)

# Priors
# (pi_{i1}, ..., pi_{iG}) ~ Dir(alpha) # Is common I prefer the below
# pi_{ig} = exp(theta_{ig}) / sum_g exp(theta_{ig}) # Use softmax rather than Dirichlet
# theta_{ig} ~ N(0, 3) # Keep reasonably close to 0. Value outside of -6/6 very extreme
# sigma ~ cauchy(0, 10)
# mu_g ~ N(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
G <- 3
N <- 200
M <- 2
mu_g <- matrix(c(-5, 0, 2, -5, 0, 2), ncol = 2, nrow = G)
Sigma <- diag(2)
theta <- matrix(rnorm(N * G, 0, 3), ncol = G, nrow = N)
pi <- exp(theta) / apply(exp(theta), 1, sum)
Z <- rep(NA, N)
y <- matrix(NA, ncol = M, nrow = N)
for (i in 1:N) {
  Z[i] <- sample(1:G, size = 1, prob = pi[i, ])
  y[i, ] <- rmvnorm(1, mu_g[Z[i], ], Sigma)
}

# Create a quick plot
plot(y) # Should be G modes

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i, 1:M] ~ dmnorm(mu_g[Z[i], 1:M], Sigma_inv)

    Z[i] ~ dcat(pi[i, 1:G])

    for (g in 1:G) {
      exp_theta[i, g] <- exp(theta[i, g])
      pi[i, g] <- exp(theta[i, g]) / sum(exp_theta[i, 1:G])
      theta[i, g] ~ dnorm(0, 6^-2)
    }
  }

  # Prior on variance matrix
  Sigma_inv <- inverse(Sigma)
  Sigma[1,1] <- sigma1^2
  Sigma[2,2] <- sigma2^2
  Sigma[1,2] <- rho * sigma1 * sigma2
  Sigma[2,1] <- Sigma[1,2]

  sigma1 ~ dt(0, 10^-2, 1)T(0,)
  sigma2 ~ dt(0, 10^-2, 1)T(0,)
  rho ~ dunif(-1, 1)

  # Priors on means
  for (g in 1:G) {
    for (m in 1:M) {
      mu_g_raw[g, m] ~ dnorm(0, 100^-2)
    }
    for (m in 2:M) {
      mu_g[g, m] <- mu_g_raw[g, m]
    }
  }
  # Sort first dimension to avoid label switching
  mu_g[1:G, 1] <- sort(mu_g_raw[1:G, 1])

}
"

# Set up the data
model_data <- list(N = N, y = y, G = G, M = M)

# Choose the parameters to watch
model_parameters <- c("mu_g", "Sigma", "Z", "pi")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
