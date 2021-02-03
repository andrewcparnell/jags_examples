# Header ------------------------------------------------------------------

# t-distributed regression for outlier detection

# Andrew Parnell

# This model fits a slight variation on the traditional stack loss model made famous (at least to me) by WinBUGS. I think it comes from Spiegelhalter et al (1996).

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# N = number of observations
# y = response
# x = explanatory variable
# alpha, beta = regression intercept and slope
# sigma = residual sd
# df = degrees of freedom parameter (possibly varying by observation)
# p = probability of an observation being from degrees of freedom 1, 2, ...

# Likelihood
# y_i ~ dt(alpha + beta * x_i, sigma, df_i)

# Prior
# alpha, beta ~ N(0, 1)
# sigma ~ half_cauchy(0, 1)
# df_i ~ dcat(p) - dcat categorical distribution
# p is fixed - see below

# Simulate data -----------------------------------------------------------


# Some R code to simulate data from the above model
N <- 100
set.seed(123)
x <- runif(N)
y <- rt(N, df = 3) * 0.8 + 2 - 2 * x
# plot(x, y)


# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
jags_code <- "
model{
  # Likelihood
  for(i in 1:N) {
    y[i] ~ dt(alpha + beta * (x[i] - mean(x)),
                sigma^-2, df[i])
    df[i] ~ dcat(p)
  }
  alpha ~ dnorm(0, 1^-2)
  beta ~ dnorm(0, 1^-2)
  sigma ~ dt(0,1,1)T(0,)
}
"

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
jags_run <- jags(
  data = list(
    N = N,
    p = rep(1, 10) / 10,
    y = y,
    x = x
  ),
  parameters.to.save = c(
    "alpha",
    "beta",
    "df"
  ),
  model.file = textConnection(jags_code)
)

dfs <- jags_run$BUGSoutput$median$df
pars <- jags_run$BUGSoutput$mean
cols <- heat.colors(10)
plot(x, y, col = cols[dfs])
lines(x, as.numeric(pars$alpha) +
  as.numeric(pars$beta) * (x - mean(x)))
# Different colours for observations more likely to be outliers

# Real example ------------------------------------------------------------

# Use just one of the covariates
data(stackloss)
plot(stackloss)

jags_run <- jags(
  data = list(
    N = nrow(stackloss),
    p = rep(1, 10) / 10,
    y = scale(stackloss$stack.loss)[, 1],
    x = stackloss$Air.Flow
  ),
  parameters.to.save = c(
    "alpha",
    "beta",
    "df"
  ),
  model.file = textConnection(jags_code)
)

dfs <- jags_run$BUGSoutput$median$df
pars <- jags_run$BUGSoutput$mean
cols <- heat.colors(10)
with(stackloss, plot(Air.Flow, scale(stack.loss)[, 1], col = cols[dfs]))
with(stackloss, lines(Air.Flow, as.numeric(pars$alpha) +
  as.numeric(pars$beta) * (Air.Flow - mean(Air.Flow))))
# One larger outlier


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
