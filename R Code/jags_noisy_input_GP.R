# Header ------------------------------------------------------------------

# Fitting a noise input Gaussian Process model in JAGS
# Andrew Parnell

# In this code we generate some data with noisy inputs then approximate it using the Noisy Input GP method by Andrew Mchutchon and Carl E. Rasmussen from NIPS in 2011

# The steps for fitting it are:
# 1) Ignore the noise uncertainty and fit a standard GP
# 2) Use the slope of the predicted values to estimate the derivative of the slope with respect to the predicted values
# 3) Re-fit the model with an extra variance term modelled as diag(Delta^T %*% Sigma %*% Delta) where Delta is the derivative and Sigma are the time variances
# 4) Use the predictions from this final model for analysis

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(MASS)

# Maths -------------------------------------------------------------------

# Notation:
# y(x): Response variable on continuous input x - I'm going to assume x is univariate for simplicity but this works for multivariate too
# x_obs, x, sigma_x: vectors of observed input, true input and its standard deviation noise. x is not observed, we only know x_obs and sigma_x
# y: vector of all observations
# alpha: Overall mean parameter
# sigma: residual standard deviation parameter (sometimes known in the GP world as the nugget)
# rho: decay parameter for the GP autocorrelation function
# tau: GP standard deviation parameter

# Likelihood:
# y ~ MVN(Mu, Sigma)
# where MVN is the multivariate normal distribution and
# Mu[x] = alpha
# Sigma is a covariance matrix with:
# Sigma_{ij} = tau^2 * exp( -rho * (x_i - x_j)^2 ) if i != j
# Sigma_{ij} = tau^2 + sigma^2 if i=j
# The part exp( -rho * (x_i - x_j)^2 ) is known as the autocorrelation function

# Prior
# alpha ~ N(0,100)
# sigma ~ U(0,10)
# tau ~ U(0,10)
# rho ~ U(0.1, 5) # Need something reasonably informative here

# Noisy inputs. We have
# x_obs[i] ~ N(x[i], sigma_x[i]^2)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N <- 20 # can take to N = 100 but fitting gets really slow ...
alpha <- 0
sigma <- 0.01
tau <- 1
rho <- 1
set.seed(123)
x <- sort(runif(N))
sigma_x <- runif(N, 0, 0.05)
x_obs <- rnorm(N, x, sigma_x)
Sigma <- sigma^2 * diag(N) + tau^2 * exp(-rho * outer(x, x, "-")^2)
y <- mvrnorm(1, rep(alpha, N), Sigma)
plot(x_obs, y) # Black is what you see
points(x, y, col = "green") # Green is the truth

# Jags code ---------------------------------------------------------------

# Jags code for stage 1 with no input noise
model_code_1 <- "
model
{
  # Likelihood
  y ~ dmnorm.vcov(Mu, Sigma)

  # Set up mean and covariance matrix
  for(i in 1:N) {
    Mu[i] <- alpha
    Sigma[i,i] <- pow(sigma, 2) + pow(tau, 2)
    for(j in (i+1):N) {
      Sigma[i,j] <- pow(tau, 2) * exp( - rho * pow(x[i] - x[j], 2) )
      Sigma[j,i] <- Sigma[i,j]
    }
  }
  alpha ~ dnorm(0, 10^-2)
  sigma ~ dunif(0, 10)
  tau ~ dunif(0, 10)
  rho ~ dunif(0.1, 5)
}
"

# Jags code for stage 2 with input noise
model_code_2 <- "
model
{
  # Likelihood
  y ~ dmnorm.vcov(Mu, Sigma)

  # Set up mean and covariance matrix
  for(i in 1:N) {
    Mu[i] <- alpha
    Sigma[i,i] <- pow(sigma, 2) + pow(tau, 2) + pow(extra[i], 2)
    for(j in (i+1):N) {
      Sigma[i,j] <- pow(tau, 2) * exp( - rho * pow(x[i] - x[j], 2) )
      Sigma[j,i] <- Sigma[i,j]
    }
  }
  alpha ~ dnorm(0, 10^-2)
  sigma ~ dunif(0, 10)
  tau ~ dunif(0, 10)
  rho ~ dunif(0.1, 5)
}
"

# Simulated results -------------------------------------------------------

# Run stage 1 with no input noise

# Set up the data
model_data <- list(N = N, y = y, x = x_obs)

# Choose the parameters to watch
model_parameters <- c("alpha", "sigma", "tau", "rho")

# Run the model - can be slow
model_run_1 <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code_1)
)

# Check the fit
plot(model_run_1)

# Stage 2, produce predictions for the data points and calculate derivatives
# y^new | y ~ N( Mu^new + Sigma_new^T solve(Sigma, y - Mu), Sigma_* - Sigma_new^T solve(Sigma, Sigma_new)
post_means <- model_run_1$BUGSoutput$mean

# Calculate predicted mean via a function
pred_mean_calc <- function(x_new) {
  N_new <- length(x_new) # Number of new predictions
  Mu <- rep(post_means$alpha, N) # Original GP mean
  Mu_new <- rep(post_means$alpha, N_new) # New GP mean
  Sigma_new <- post_means$tau[1]^2 * exp(-post_means$rho[1] * outer(x, x_new, "-")^2) # Cross-covariance of new to old
  Sigma <- post_means$sigma[1]^2 * diag(N) + post_means$tau[1]^2 * exp(-post_means$rho[1] * outer(x, x, "-")^2) # Old variance matrix
  return(Mu_new + t(Sigma_new) %*% solve(Sigma, y - Mu)) # Return the predictions
}
# pred_mean_calc(0.5) # Test the function

# Now create derivatives
h <- 0.01
deriv <- (pred_mean_calc(x + h) - pred_mean_calc(x - h)) / (2 * h)

# Add this new term in - this is the extra standard deviation on each term
model_data$extra <- sqrt(deriv^2 * sigma_x^2)[, 1]

# Stage 3 fit this new model
# Run the model - can be slow
model_run_2 <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code_2)
)

# Check the fit
plot(model_run_2)

# Stage 4 produce predictions based on this model
post_means_2 <- model_run_2$BUGSoutput$mean
N_new <- 100
x_new <- seq(min(x), max(x), length = N_new)
N_new <- length(x_new) # Number of new predictions
Mu <- rep(post_means_2$alpha, N) # Original GP mean
Mu_new <- rep(post_means_2$alpha, N_new) # New GP mean

# Note the extra terms in Sigma based on the extra variance needed
Sigma_new <- post_means_2$tau[1]^2 * exp(-post_means_2$rho[1] * outer(x, x_new, "-")^2) # Cross-covariance of new to old
Sigma_star <- post_means_2$sigma[1]^2 * diag(N_new) + post_means_2$tau[1]^2 * exp(-post_means_2$rho[1] * outer(x_new, x_new, "-")^2)
Sigma <- diag(model_data$extra^2) + post_means_2$sigma[1]^2 * diag(N) + post_means_2$tau[1]^2 * exp(-post_means_2$rho[1] * outer(x, x, "-")^2) # Old variance matrix
pred_mean <- Mu_new + t(Sigma_new) %*% solve(Sigma, y - Mu) # Get predicted means
pred_var <- Sigma_star - t(Sigma_new) %*% solve(Sigma, Sigma_new) # Predicted variances

# Plot output
plot(x, y) # True x and y, remember x is unobserved
lines(x_new, pred_mean, col = "red")
points(x_obs, y, col = "green") # Observed x values

# Now add in the uncertainties
pred_low <- pred_mean - 1.96 * sqrt(diag(pred_var))
pred_high <- pred_mean + 1.96 * sqrt(diag(pred_var))
lines(x_new, pred_low, col = "red", lty = 2)
lines(x_new, pred_high, col = "red", lty = 2)
