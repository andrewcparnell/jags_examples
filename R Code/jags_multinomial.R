# Header ------------------------------------------------------------------

# Multinomial JAGS example
# Andrew Parnell

# Demostrates a soft-maxmultinomial model

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y[i,j] = matrix of counts with each row i having set row sum S on j = 1 to M categories, i = 1, to N
# x[i,k] = covariate matrix with k = 1 to K covariate values
# p[i,j] = probability of observation i being in variable k
# beta[j,k] = regression parameter of variable k in category j

# Likelihood
# y ~ multinomial(S, p)
# p[i,j] = softmax(x * beta)
# where softmax(r) = exp(r_1)/sum(exp(r), ..., exp(r_S)/sum(exp(r))

# Priors
# beta[j,k] ~ N(0, 1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 500 # Number of observations
M = 3 # Number of categories
S = rpois(N, 20) + 1 # Sum of values (different for each observation)
K = 2 # Number of covariates

softmax = function(x) exp(x)/sum(exp(x))

beta = matrix(rnorm(M*K), nrow = M, ncol = K)
x = matrix(rnorm(N*K), nrow = N, ncol = K)
p = y = matrix(NA, nrow = N, ncol = M)
for(i in 1:N) {
  p[i,] = softmax(beta%*%x[i,])
  y[i,] = rmultinom(1, size = S[i], prob = p[i,])
}
# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) { # Observaton loops
    y[i,] ~ dmulti(p[i,], S[i])
    for(j in 1:M) { # Category loop
      exp_z[i,j] <- exp(z[i,j])
      p[i,j] <- exp_z[i,j]/sum(exp_z[i,])
      z[i,j] <- beta[j,]%*%x[i,]
    }
  }
  # Prior
  for(j in 1:M) {
    for(k in 1:K) {
      beta[j,k] ~ dnorm(0, 0.1^-2)
    }
  }
}
'

# Set up the data
model_data = list(N = N, y = y, x = x, S = S, K = K, M = M)

# Choose the parameters to watch
model_parameters =  c("beta", "p")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Compare the predicted vs true values of beta
model_run$BUGSoutput$mean$beta
beta

# However you're better off lookin at the predicted probabilities as these
# will be more directly comparable
p_pred = model_run$BUGSoutput$mean$p
head(cbind(p[,1], p_pred[,1]), 20)

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory

# This is some real pollen data
pollen = read.csv('https://raw.githubusercontent.com/andrewcparnell/bhm_course/master/data/pollen.csv')
pollen$sum = rowSums(pollen[,3:ncol(pollen)])
str(pollen)

# Idea is to predict counts of Abies to Graminaea based on covariates of
# GDD5 and MTCO

# Set up the data

model_data = list(N = nrow(pollen[1:500,]), y = pollen[1:500,3:9],
                  x = cbind(1, scale(cbind(pollen[1:500,1:2],pollen[1:500,1:2]^2))),
                  S = pollen[1:500,10],
                  K = 5, # Number of covars
                  M = 7) # Number of categories

# Choose the parameters to watch
model_parameters =  c("beta", "p")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

# Create a quick plot to see how well it fits
p3_model = model_run$BUGSoutput$mean$p[,3]
p3_data = pollen$Betula/pollen$sum
plot(p3_data[1:500], p3_model,
     xlab = 'True proportion of Betula',
     ylab = 'Estimated proportion of Betula')
abline(a=0, b=1)
# Model does not fit well!

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) The main challenge is to find a model that fits these data well. It's really hard as the data suffer from zero inflation and complex non-linear dependence on the two covariates. Maybe higher powers of the covariates will help? (Or if you want to try somethng more complex, splines or Gaussian processes)
# 2) It might also be worth running the modle on more of the data (rather than just the first 5 observations)
# 3) Alternatively you could try removing some of the counts and only running the model on a subset of them - perhaps some of these will then fit better

