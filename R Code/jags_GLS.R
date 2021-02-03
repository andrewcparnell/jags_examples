# Header ------------------------------------------------------------------

# An example of a generalised least squares (GLS) model in Jags

# Andrew Parnell

# In this file I simulate some data from a GLS type model, then fit it using JAGS, and evaluate the fit
# Much of the interest in GLS comes from the fact that you can prescribe patterns to the variance matrix which can yield some interesting model specifications. I'm going to leave it as a general unknown matrix

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)
library(MASS) # To simualte from MVN

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# Let y_i be the response variable for observation i
# Let x_i be the explanatory variable for observation i
# Let alpha and beta be parameters representing slope and intercept respectively
# Let V be the variance matrix of the residuals. This induces correlations between the parameters
# Let Omega = V^{-1} be the precision matrix; the inverse of V
# Sometimes we might like to specify the structure underlying Omega or V. I'm going to use a time series structure where V_ij = rho^|i - j|
# rho is now an autocorrelation parameter

# Likelihood
# y_i = alpha + beta * x_i + epsilon_i
# Key is that, unlike standard linear regression, Cov(epsilon_i, epsilon_j) \ne 0
# In general: y ~ MVN(X%*%Theta, V)
# where X is cbind(1, x) and Theta = (alpha, beta)
# Prior - using vague ones here
# alpha ~ N(0, 10)
# beta ~ N(0, 10)
# V_ij = rho^|i - j| Other correlation structures are available. This is an AR1 structure

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N <- 50
alpha <- 2
beta <- 3
Theta <- c(alpha, beta)
set.seed(123)
x <- sort(runif(N))
X <- cbind(1, x)
rho <- 0.7
V <- matrix(NA, ncol = N, nrow = N)
for (i in 1:N) {
  for (j in 1:N) {
    V[i, j] <- rho^abs(i - j)
  }
}
y <- mvrnorm(1, mu = X %*% Theta, Sigma = V)
# plot(x, y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model{
  # Likelihood
  y ~ dmnorm.vcov(X%*%Theta, V)
  # Priors
  for(i in 1:p) {
    Theta[i] ~ dnorm(0, 10^-2)
  }
  for(i in 1:N) {
    for(j in 1:N) {
      V[i,j] = rho^abs(i - j)
    }
  }
  rho ~ dunif(0, 1)
}
"

model_parameters <- c("Theta", "rho")
model_data <- list(
  y = y,
  N = length(y),
  X = X,
  p = ncol(X)
)

# Run the model - can be slow
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

plot(model_run)
# Works beautifully - looks just like the true values

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory
# Example taken from here: http://halweb.uc3m.es/esp/Personal/personas/durban/esp/web/notes/gls.pdf

data(longley)
# In GLS you'd run something like this:
# library(nlme)
# g<-gls(Employed ~ GNP,correlation=corAR1(form=~Year),data=longley)

# In Jags we're doing to run
model_data <- list(
  y = longley$Employed,
  N = nrow(longley),
  X = cbind(1, longley$GNP),
  p = 2
)

model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)
plot(model_run)

# Quite a strong degree of autocorrelation from rho, and interesting estimates of effect of GSP (not shown well on plot)
