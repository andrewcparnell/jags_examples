# Header ------------------------------------------------------------------

# A sum of two Gaussian process models in JAGS
# Andrew Parnell

# This file fits a sum of Gaussian Process (GP) regression models to data in JAGS, and produces predictions/forecasts
# It is not meant to converge as the two GPs are non-identifiable

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls())
library(R2jags)
library(MASS) # Useful for mvrnorm function

# Maths -------------------------------------------------------------------

# Notation:
# y(x): Response variable at explanatory variable(s) value x, where x is continuous
# y: vector of all observations
# mu: Overall mean parameter
# sigma_k: residual standard deviation parameter for GP k (sometimes known in the GP world as the nugget)
# rho_k: decay parameter for the GP autocorrelation function for GP k
# tau: GP standard deviation parameter for GP k - both fixed ot be the same

# Likelihood:
# y ~ N(mu + g_1 + g_2, sigma^2)
# g_k ~ MVN(0, Sigma_k)
# where MVN is the multivariate normal distribution and
# Sigma_k is a covariance matrix with:
# Sigma_{kij} = tau_k^2 * exp( -rho_k * (x_i - x_j)^2 )
# The part exp( -rho_k * (x_i - x_j)^2 ) is known as the autocorrelation function

# Prior
# mu ~ N(0,100)
# sigma ~ U(0,10)
# tau_k ~ U(0,10)
# rho_k ~ U(0.1, 5) # Need something reasonably informative here

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N <- 25 # can take to N = 100 but fitting gets really slow ...
set.seed(123)
x <- sort(runif(N))
true_mean <- 2 + x + sin(5 * x)
y <- true_mean + rnorm(N, sd = 0.2)
plot(x, y)
lines(x, true_mean, col = "red")


# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(mu + gp_1[i] + gp_2[i], sigma^-2)
    r_1[i] = y[i] - gp_2[i]
    r_2[i] = y[i] - gp_1[i]
  }
  gp_1 ~ dmnorm.vcov(zeros, Sigma_1)
  gp_2 ~ dmnorm.vcov(zeros, Sigma_2)

  # Set up mean and covariance matrix
  for(i in 1:N) {
    Sigma_1[i,i] <- pow(tau, 2) + 0.001
    Sigma_2[i,i] <- pow(tau, 2) + 0.001

    for(j in (i+1):N) {
      Sigma_1[i,j] <- pow(tau, 2) * exp( - rho_1 * pow(x[i] - x[j], 2) )
      Sigma_1[j,i] <- Sigma_1[i,j]
      Sigma_2[i,j] <- pow(tau, 2) * exp( - rho_2 * pow(x[i] - x[j], 2) )
      Sigma_2[j,i] <- Sigma_2[i,j]
    }
  }

  mu ~ dnorm(0, 100^-2)
  sigma ~ dunif(0, 10)
  rho_1 ~ dunif(0.1, 5)
  rho_2 ~ dunif(0.1, 5)
  tau ~ dunif(0, 10)

}
"

# Set up the data
model_data <- list(N = N, y = y, x = x, zeros = rep(0, N))

# Choose the parameters to watch
model_parameters <- c("mu", "sigma", "tau", "rho_1", "rho_2", "r_1", "r_2", "gp_1", "gp_2")

# Run the model - can be slow
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 1
) # Number of different starting positions. Only need 1 here as it won't converge


# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)

# Have a look at the partial residuals
plot(x, y)
lines(x, true_mean, col = "red")
lines(x, model_run$BUGSoutput$mean$r_1)
lines(x, model_run$BUGSoutput$mean$r_2)
lines(x, 2 * y - model_run$BUGSoutput$mean$r_1 - model_run$BUGSoutput$mean$r_2 + model_run$BUGSoutput$mean$mu[1], col = "blue")

# Now create some predictions of new values at new times x_new
# These are based on the formula:
# y_new_hat = mu + gp_1_new + gp_2_new

# I'm just going to use the mean values of the parameters
mu <- model_run$BUGSoutput$mean$mu[1]
sigma <- model_run$BUGSoutput$mean$sigma[1]
tau <- model_run$BUGSoutput$mean$tau[1]
rho_1 <- model_run$BUGSoutput$mean$rho_1[1]
rho_2 <- model_run$BUGSoutput$mean$rho_2[1]
gp_1 <- model_run$BUGSoutput$mean$gp_1
gp_2 <- model_run$BUGSoutput$mean$gp_2

# The GP predictions come from
# gp_k_new | y ~ N( Sigma_k_new^T solve(Sigma, gp_k), Sigma_k_* - Sigma_k_new^T solve(Sigma_k, Sigma_k_new)
# where
# Sigma_k_new[i,j] = tau^2 * exp( -rho_j * (x_new_i - x_j)^2 )
# Sigma_*[i,j] = tau^2 * exp( -rho_k * (x_new_i - x_new_j)^2 )

# Now create predictions
N_new <- 100
x_new <- seq(0, 1, length = N_new)
Sigma_1_new <- tau^2 * exp(-rho_1 * outer(x, x_new, "-")^2)
Sigma_2_new <- tau^2 * exp(-rho_2 * outer(x, x_new, "-")^2)
Sigma_1_star <- tau^2 * exp(-rho_1 * outer(x_new, x_new, "-")^2)
Sigma_2_star <- tau^2 * exp(-rho_2 * outer(x_new, x_new, "-")^2)
Sigma_1 <- tau^2 * exp(-rho_1 * outer(x, x, "-")^2) + diag(0.001, N)
Sigma_2 <- tau^2 * exp(-rho_2 * outer(x, x, "-")^2) + diag(0.001, N)

# Use fancy equation to get predictions
gp_1_new <- t(Sigma_1_new) %*% solve(Sigma_1, gp_1)
gp_2_new <- t(Sigma_2_new) %*% solve(Sigma_2, gp_2)
pred_mean <- mu + gp_1_new + gp_2_new

# Plot output
plot(x, y)
lines(x, true_mean, col = "red")
lines(x_new, pred_mean, col = "blue")
# Pretty good!
