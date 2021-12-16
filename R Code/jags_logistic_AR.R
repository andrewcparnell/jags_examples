# Header ------------------------------------------------------------------

# Fitting a logistic autoregression in JAGS
# Andrew Parnell

# In this file we fit a Bayesian Generalised Linear Model (GLM) in the form
# of a logistic time series autoregression

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(boot) # Package contains the logit transform

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = binomial (often binary) response variable for time t=1, ..., T
# p_t = probability of y_t being 1 for observation t
# alpha = intercept term
# beta = autoregressive parameter value (should be between 0 and 1 for a standard model)

# Likelihood
# y_t ~ Binomial(K,p_t), or Binomial(1,p_t) if binary
# logit(p_t) ~ N(alpha + beta * logit(p_{t-1}), sigma^2)
# So it's an AR model on the logit term with extra dispersion captured by sigma
# where logit(p_t) = log( p_t / (1 - p_t ))
# Note that p_t has to be between 0 and 1, but logit(p_t) has no limits

# Priors - all vague
# alpha ~ normal(0,100)
# beta_1 ~ normal(0,100)
# beta_2 ~ normal(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
set.seed(123)
alpha <- -0.3
beta <- 0.8
sigma <- 0.5
logit_p <- rep(NA, T)
logit_p[1] <- 0.2
for(t in 2:T) logit_p[t] <- alpha + beta * logit_p[t-1] + rnorm(1, 0, sigma)
p <- inv.logit(logit_p)
y <- rbinom(T, 1, p)

# Have a quick look at the effect of x_1 and x_2 on y
plot(1:T, y)
lines(1:T, p) # The true value of p

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  logit_p[1] ~ dnorm(alpha, sigma^-2)
  for (t in 2:T) {
    y[t] ~ dbin(p[t], K)
    p[t] <- exp(logit_p[t]) / (1 + exp(logit_p[t]))
    logit_p[t] ~ dnorm(alpha + beta * logit_p[t-1], sigma^-2)
  }

  # Priors
  alpha ~ dnorm(0, 5^-2)
  beta ~ dunif(0, 1)
  sigma ~ dgamma(1,1)
}
"

# Set up the data
model_data <- list(T = T, y = y, K = 1)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta", "sigma", "p")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.iter = 10000,
  n.burnin = 2000,
  n.thin = 5
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)

# Create a plot of the posterior model run
post <- model_run$BUGSoutput$sims.list$p
p_median <- apply(post, 2, 'quantile', 0.5)
p_low <- apply(post, 2, 'quantile', 0.05)
p_high <- apply(post, 2, 'quantile', 0.95)
plot(1:T, y)
lines(1:T, p) # The true value of p
lines(2:T, p_median, col = 'red') # The true value of p
lines(2:T, p_low, col = 'red', lty = 'dotted')
lines(2:T, p_high, col = 'red', lty = 'dotted')
