
# Get everything set up
rm(list=ls())
require(R2jags)
library(loo)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
# First an AR(1)

set.seed(123)
T = 100
t_seq = 1:T
sigma = 1
alpha = 1    
beta = 0.6    # Constrain beta to (-1,1) so the series doesn't explode
y = rep(NA,T)
y[1] = rnorm(1,0,sigma)
for(t in 2:T) y[t] = rnorm(1, alpha + beta * y[t-1], sigma)
# plot
plot(t_seq, y, type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code = '
model
{
  # Likelihood
  for (t in (p+1):T) {
    y[t] ~ dnorm(mu[t], tau)
    mu[t] <- alpha + inprod(beta, y[(t-p):(t-1)])
    log_lik[t] <- logdensity.norm(y[t], mu[t], tau)
  }
  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:p) {
    beta[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

# Set up the data
model_data = list(T = T, y = y, p = 1)

# Choose the parameters to watch
model_parameters =  c("alpha","beta","sigma","log_lik")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code),
                 n.chains = 4, # Number of different starting positions
                 n.iter = 1000, # Number of iterations
                 n.burnin = 200, # Number of iterations to remove at start
                 n.thin = 2) # Amount of thinning

# Get the log likelihood
log_lik = model_run$BUGSoutput$sims.list$log_lik
waic(log_lik)
