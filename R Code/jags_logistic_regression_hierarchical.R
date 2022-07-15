# Header ------------------------------------------------------------------

# Fitting a hierarchical logistic regression in JAGS
# Andrew Parnell

# In this file we fit a Bayesian Hierarchical Generalised Linear
# (GLM) in the form of a logistic regression with two categorical variables

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(boot) # Package contains the logit transform
library(ggplot2)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_i = binomial (often binary) response variable for observation i=1,...,N
# x_{1i} = first categorical explanatory variable for observation i
# x_{2i} = second " " " " " " " " "
# p_i = probability of y_i being 1 for observation i
# alpha = intercept term
# beta_1 = parameter value for explanatory variable 1
# beta_2 = parameter value for explanatory variable 2

# Likelihood
# y_i ~ Binomial(K,p_i), or Binomial(1,p_i) if binary
# logit(p_i) = mu + beta_1[x_[i]] + beta_2[x_2[i]]
# where logit(p_i) = log( p_i / (1 - p_i ))
# Note that p_i has to be between 0 and 1, but logit(p_i) has no limits

# Priors - all vague
# mu ~ normal(0,10)
# beta_1 ~ normal(0, sigma_1^2)
# beta_2 ~ normal(0, sigma_2^2)
# sigma_1 ~ half-cauchy(0, 1)
# sigma_2 ~ half-cauchy(0, 1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N <- 200
set.seed(100)
# Set the number of levels for each factor
# Note if you make these too small the model will work fine
# but the sample means of beta_1 and beta_2 will not be zero and so
# the mu parameter will need to be shifted to be the overall mean
N_levels_1 <- 10
N_levels_2 <- 8
x_1 <- sample(1:N_levels_1, size = N, replace = TRUE)
x_2 <- sample(1:N_levels_2, size = N, replace = TRUE)
mu <- 1
sigma_1 <- 2
sigma_2 <- 1
beta_1 <- rnorm(N_levels_1, 0, sigma_1)
beta_2 <- rnorm(N_levels_2, 0, sigma_2)
logit_p <- mu + beta_1[x_1] + beta_2[x_2]
p <- inv.logit(logit_p)
y <- rbinom(N, 1, p)

# Have a quick look at the effect of x_1 and x_2 on y
table(x_1, y)
table(x_2, y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dbin(p[i], 1)
    logit(p[i]) <- mu + beta_1[x_1[i]] + beta_2[x_2[i]]
  }

  # Priors
  mu ~ dnorm(0, 10^-2)
  for (j in 1:N_levels_1) {
    beta_1[j] ~ dnorm(0, sigma_1^-2)
  }
  for (j in 1:N_levels_2) {
    beta_2[j] ~ dnorm(0, sigma_2^-2)
  }
  sigma_1 ~ dt(0, 1^-2, 1)T(0,)
  sigma_2 ~ dt(0, 1^-2, 1)T(0,)
}
"

# Set up the data
model_data <- list(N = N, y = y, x_1 = x_1, x_2 = x_2,
                   N_levels_1 = N_levels_1, N_levels_2 = N_levels_2)

# Choose the parameters to watch
model_parameters <- c("mu", "beta_1", "beta_2", "sigma_1", "sigma_2")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)

# Compare the values of beta_1 with the truth
qplot(model_run$BUGSoutput$mean$beta_1, beta_1) + geom_abline()
qplot(model_run$BUGSoutput$mean$beta_2, beta_2) + geom_abline()

