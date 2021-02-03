# Header ------------------------------------------------------------------

# Dirichlet Autoregressive model of order 1
# Andrew Parnell

# Some JAGS code to fit a Dirichlet AR(1) model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(bayesm)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = compositional vector of repsonses at time t, t = 1,...,T
# y_r(t) = compositional value for species r, r = 1, ..., R
# alpha_r = overall mean parameter for species r
# beta_r = autocorrelation/autoregressive (AR) parameter for species r
# a_r(t) = dirichlet parameter for species r at time t

# Likelihood
# y[t, 1:R] ~ ddirch(a[t, 1:R])

# Second level log(a[t, r]) = alpha_r + beta_r * log(a[t-1, r])

# Priors
# alpha_r ~ dnorm(0,100)
# beta_r ~ dunif(-1,1) # If you want the process to be stable/stationary

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
# First AR1
set.seed(123)
T <- 100
t_seq <- 1:T
R <- 4
alpha <- rep(1, R)
beta <- runif(R, 0.2, 0.8)
sigma_a <- runif(R, 0.1, 0.3)
y <- log_a <- matrix(NA, nrow = T, ncol = 4)
log_a[1, ] <- 0
y[1, ] <- rdirichlet(exp(log_a[1, ]))
for (t in 2:T) {
  for (r in 1:R) {
    log_a[t, r] <- rnorm(
      1, alpha[r] + beta[r] * log_a[t - 1, r],
      sigma_a[r]
    )
  }
  y[t, ] <- rdirichlet(exp(log_a[t, ]))
}
# plot
plot(t_seq, y[, 1], type = "l", ylim = c(0, 1))
lines(t_seq, y[, 2], col = 2)
lines(t_seq, y[, 3], col = 3)
lines(t_seq, y[, 4], col = 4)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
# This code is for a general AR(p) model
model_code <- "
model
{
  for (r in 1:R) {
    log(a[1, r]) <- alpha[r]
  }

  # Likelihood
  for (t in 2:T) {
    y[t, 1:R] ~ ddirch(a[t, 1:R])
    for (r in 1:R) {
      a[t, r] <- exp(log_a[t, r])
      log_a[t, r] ~ dnorm(alpha[r] + beta[r] * log_a[t-1, r], sigma_a[r])
    }
  }

  for(t in 1:T) {
    for(r in 1:R) {
      mean_a[t,r] = a[t, r] / sum(a[t, 1:R])
    }
  }

  # Priors
  for (r in 1:R) {
    log_a[1, r] ~ dnorm(alpha[r], sigma_a[r])
    alpha[r] ~ dnorm(0, 10^-2)
    beta[r] ~ dnorm(0, 10^-2)
    sigma_a[r] ~ dunif(0, 100)
  }
}
"

# Set up the data
model_data <- list(T = T, R = R, y = y)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta", "mean_a")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.iter = 10000,
  n.burnin = 2000,
  n.thin = 8
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
print(model_run)

# Look at alpha
par(mfrow = c(2, 2))
hist(model_run$BUGSoutput$sims.list$alpha[, 1])
hist(model_run$BUGSoutput$sims.list$alpha[, 2])
hist(model_run$BUGSoutput$sims.list$alpha[, 3])
hist(model_run$BUGSoutput$sims.list$alpha[, 4])

par(mfrow = c(2, 2))
hist(model_run$BUGSoutput$sims.list$beta[, 1])
hist(model_run$BUGSoutput$sims.list$beta[, 2])
hist(model_run$BUGSoutput$sims.list$beta[, 3])
hist(model_run$BUGSoutput$sims.list$beta[, 4])

# Get the mean over time
post_mean_a <- model_run$BUGSoutput$sims.list$mean_a
post_mean_a_mean <- apply(post_mean_a, c(2, 3), "mean")

# plot
par(mfrow = c(2, 2))
plot(t_seq, y[, 1], type = "l", ylim = c(0, 1))
lines(t_seq, post_mean_a_mean[, 1], lty = "dotted")
plot(t_seq, y[, 2], type = "l", ylim = c(0, 1), col = 2)
lines(t_seq, post_mean_a_mean[, 2], lty = "dotted", col = 2)
plot(t_seq, y[, 3], type = "l", ylim = c(0, 1), col = 2)
lines(t_seq, post_mean_a_mean[, 3], lty = "dotted", col = 3)
plot(t_seq, y[, 4], type = "l", ylim = c(0, 1), col = 2)
lines(t_seq, post_mean_a_mean[, 4], lty = "dotted", col = 4)
