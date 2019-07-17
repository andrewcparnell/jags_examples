# Header ------------------------------------------------------------------

# Fitting a multiple response regression in JAGS
# Andrew Parnell

# In this file we fit a Bayesian muliple response regression model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)
library(MASS) # For the multivariate normal distribution

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_{i} = J-vector of response variables for observation i, i = 1, ..., N
# x_{i} = K-vector of explanatory variables for observation i
# Together Y is an N by J matrix of response variables (each observation is of dimenstion J) and X is an N by K matrix of explanatory variables (each observation has K response variables)

# Parameters
# A = intercept vector of length J
# B = slope matrix of dimension K by J, i.e. this is the matrix of slopes for explanatory variable k on dimensino j
# Sigma = residual variance matrix of dimension J by J

# Likelihood
# y[i,] ~ N(A + B*x[i,], Sigma)

# Priors - all vague
# A[j] ~ normal(0, 100)
# B[j,k] ~ normal(0,100)
# Smarter priors might tie these values, especially the slopes together
# This borrowing of strength might take place over dimensions (j) or covariates (k)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 100
J = 3
K = 5
set.seed(123)
X = matrix(rnorm(N*K), nrow = N, ncol = K)

# Simulate parameters
Sigma = rWishart(1, df = J+1, Sigma = diag(J))[,,1]
A = rnorm(J)
B = matrix(rnorm(J*K)*5, ncol = J, nrow = K)

# Get the means and simulate data
mean = y = matrix(NA, ncol = J, nrow = N)
for(i in 1:N) {
  mean[i,] = A + X[i,]%*%B
  y[i,] = mvrnorm(1, mean[i,], Sigma)
}

# Very hard to visualise!
pairs(data.frame(y = y,
                 X = X))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i, ] ~ dmnorm(mu[i, ], Sigma.Inv)
    mu[i, 1:J] <- A + X[i, ]%*%B
  }
  Sigma.Inv ~ dwish(I, J+1)
  Sigma <- inverse(Sigma.Inv)

  # Priors
  for(j in 1:J) {
    A[j] ~ dnorm(0, 100^-2)
    for(k in 1:K) {
      B[k,j] ~ dnorm(0, 100^-2)
    }
  }
}
'

# Set up the data
model_data = list(N = N, y = y, X = X, K = ncol(X), J = ncol(y),
                  I = diag(J))

# Choose the parameters to watch
model_parameters =  c("A", "B", "Sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
#traceplot(model_run)

# Check whether the values match the truth
B_post = model_run$BUGSoutput$mean$B
plot(B, B_post); abline(a=0, b=1)
A_post = model_run$BUGSoutput$mean$A
plot(A, A_post); abline(a=0, b=1)

# Real example ------------------------------------------------------------

# Not done yet

