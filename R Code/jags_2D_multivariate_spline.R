# Header ------------------------------------------------------------------

# A 2D P-spline multivariate output model in JAGS
# Andrew Parnell

# This file fits a 2D spline regression model to data in JAGS, and produces
# predictions/forecasts

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls())
library(R2jags)
library(mvnfast) # Useful for mvn functions
library(akima) # Useful for 2D interpolation
library(splines) # Useful for creating the B-spline basis functions
library(bayesm) # For Wishart distn

# Maths -------------------------------------------------------------------

# Notation:
# y(t, k): Response variable at time t for variable k
# y(t): Multivariate response variable at time t
# y: matrix of all observations
# B: design matrix of spline basis functions - needs to be a tensor produce
# over 2 dimensions. B is the same for every dimension of y
# beta; spline weights - a matrix as one column for each dimension
# Sigma: residual variance matrix
# sigma_b: spline smoothness parameter for each dimension of y

# Likelihood:
# y[t,] ~ N(mu[t,], Sigma)
# mu[,k] <- B%*%beta[,k]
# beta[j,k] ~ N(beta[j-1,k], tau_b[k]^-1)

# Priors
# Sigma ~ Inverse Wishart(I, k+1)
# tau_b[k] ~ gamma(0.5*nu, 0.5*delta[k]*nu)
# delta[k] ~ gamma(a_delta,d_delta)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(123)
N <- 200 # Number of observations
K <- 5 # Dimension of the y output
x1 <- runif(N, 0, 10) # Create some 2D covariate values
x2 <- runif(N, 0, 10) # Create some 2D covariate values

# Create a vector of all the x values
x <- cbind(x1, x2)

# Create the basis functions
nIknots_sim <- 10
knots1 <- quantile(x1, seq(0, 1,
                         length.out = nIknots_sim + 2))[-c(1,
                                                           nIknots_sim + 2)]
knots2 <- quantile(x2, seq(0, 1,
                           length.out = nIknots_sim + 2))[-c(1,
                                                             nIknots_sim + 2)]
B1 <- splines::ns(x = x1, knots = knots1, intercept = FALSE)
B2 <- splines::ns(x = x2, knots = knots2, intercept = FALSE)

# Create the matrix which is now going to be each column of B1 multiplied by each column of B2
# There's perhaps a more elegant way of doing this
B <- matrix(NA, ncol = ncol(B1) * ncol(B2), nrow = N)
count <- 1
for (i in 1:ncol(B1)) {
  for (j in 1:ncol(B2)) {
    B[, count] <- B1[, i] * B2[, j]
    count <- count + 1
  }
}
# If required, plot some of the basis functions
# for(i in 1:ncol(B)) {
#   plot(x[,1], x[,2], cex = B[,i]/max(B[,i]), pch = 19)
#   Sys.sleep(0.1)
# }

tau_b <- rep(1, K)
Sigma <- rwishart(K, diag(K))$W
beta <- matrix(0, ncol = K, nrow = ncol(B))
mu <- matrix(NA, ncol = K, nrow = N)
for(k in 1:K) {
  for(j in 2:nrow(beta)) {
    beta[j,k] <- rnorm(1, beta[j-1, k], sd = sqrt(1/tau_b[k]))
  }
  mu[,k] <- B%*%beta[,k]
}

y <- matrix(NA, ncol = K, nrow = N)
for (i in 1:N) {
  y[i,] <- rmvn(1, mu[i,], Sigma)
}

# Plot the underlying mean surface
pick <- 4
mu_interp <- interp(x1, x2, mu[,1])
with(mu_interp, contour(x, y, z, col = "red"))

# Add in the data points too if reuquired
points(x[, 1], x[, 2], cex = y[,pick] / max(y[,pick]), pch = 19) # Should see bigger dots ad higher countours

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i,] ~ dmnorm(mu[i,], Sigma_Inv)
  }
  for(k in 1:K) {
    mu[1:N,k] <- B%*%beta[1:N_knots,k]
  }

  # Prior on beta
  for (k in 1:K) {
    beta[1,k] ~ dnorm(0, 100^-2)
    for (i in 2:N_knots) {
      beta[i, k] ~ dnorm(beta[i - 1, k], tau_b[k])
    }
    tau_b[k] ~ dgamma(0.5 * nu, 0.5 * delta[k] * nu)
    delta[k] ~ dgamma(a_delta, d_delta)
  }

  # Prior on covariance matrix
  Sigma_Inv ~ dwish(I, K+1)
}
"

# Set up the data
model_data <- list(N = N, K = K,
                   y = y, B = B,
                   a_delta = 0.0001,
                   d_delta = 0.0001,
                   nu = 2,
                   I = diag(K),
                   N_knots = ncol(B))

# Choose the parameters to watch
model_parameters <- c("beta", "Sigma_inv", "tau_b", "mu")

# Run the model - can be slow
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
# print(model_run)
plot(model_run)

# Get the posterior betas and 50% CI
beta_post <- model_run$BUGSoutput$sims.list$beta
beta_median <- model_run$BUGSoutput$median$beta

# Plot the output with uncertainty bands

#First plot the data
pick <- 1
plot(x[, 1], x[, 2], cex = y[,pick] / max(y[,pick]), pch = 19)

# Now plot the true mean surface
with(mu_interp, contour(x, y, z, col = "red", add = TRUE))

# Now plot the estimated median surface
mu_pred <- B %*% beta_median[,pick]
mu_pred_interp <- interp(x1, x2, mu_pred)
with(mu_pred_interp, contour(x, y, z, col = "blue", add = TRUE)) # Red and blue contour lines should look similar

# Just plot the y vs the mean
plot(y[,pick], mu_pred)
abline(a = 0, b = 1)

# Finally create some new predictions on a grid of new values
# Needs to be in the same range as the previous values (if not you need to go back to the creation of B above)
n_grid <- 50
x1_new <- seq(0, 10, length = n_grid)
x2_new <- seq(0, 10, length = n_grid)
x_new <- expand.grid(x1_new, x2_new)

# Create new B matrix
B1_new <- splines::ns(x = x_new[,1], knots = knots1, intercept = FALSE)
B2_new <- splines::ns(x = x_new[,2], knots = knots2, intercept = FALSE)

# Create the matrix which is now going to be each column of B1 multiplied by each column of B2
# There's perhaps a more elegant way of doing this
B_new <- matrix(NA, ncol = ncol(B1_new) * ncol(B2_new), nrow = n_grid^2)
count <- 1
for (i in 1:ncol(B1_new)) {
  for (j in 1:ncol(B2_new)) {
    B_new[, count] <- B1_new[, i] * B2_new[, j]
    count <- count + 1
  }
}

# Plot the new interpolated predictions on top
pick <- 3
mu_interp_pred <- B_new %*% beta_median[,pick]
mu_pred_interp_2 <- interp(x_new[, 1], x_new[, 2], mu_interp_pred)
plot(x[, 1], x[, 2], cex = y[,pick] / max(y[,pick]), pch = 19)
with(mu_pred_interp_2, contour(x, y, z, col = "green", add = TRUE)) # Red and blue contour lines should look similar

# Or just plot them separately
with(mu_pred_interp_2, image(x, y, z)) # Red and blue contour lines should look similar
points(x[, 1], x[, 2], cex = y[,pick] / max(y[,pick]), pch = 19)

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
