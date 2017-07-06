# Header ------------------------------------------------------------------

# Change point modelling in JAGS
# Andrew Parnell

# This file implements different versions of change point modelling in jags. The basic versionhas constant periods and discontinuous jumps. A more advanced version contains jumps in the slope rather than the mean. I call the former discontinuous change point regression (DCPR), and the latter continuous change point regression (CCPR)

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls())
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable observated at times t. In these files time can be discrete or continuous
# t_k = time of change point k - these are the key parameters to be estimated, k = 1, .., K - K is the number of change points
# alpha_k = intercept term - possibly variging
# beta_k = slope value for period k
# sigma = overall residual standard deviation

# Likelihood:
# Top level likelihood is always:
# y(t) ~ normal(mu[t], sigma^2)

# Then for DCPR with one change point:
# mu[t] = alpha[1] if t < t_1, or mu[t] = alpha[2] if t>=t_1

# For CCPR with one change point:
# mu[t] = alpha + beta[1] * (t - t_1) if t < t_1, or mu[t] = alpha + beta[2] * (t - t_1) if t>=t_1
# Note that this is a clever way of expresssing the model as it means that alpha is the mean of y at the change point

# For CCPR with two change points:
# mu[t] = alpha[1] + beta[1] * (t - t_1) if t < t_1
# or alpha[1] + beta[2] * (t - t_1) if t_1 <= t < t_2
# or alpha[2] + beta[3] * (t - t_2) if t >= t_2

# To achieve this kind of model in jags we use the step function which works via:
# step(x) = 1 if x>0 or 0 otherwise.
# We can use it to pick out which side of the change point(s) we're on

# Priors
alpha ~ normal(0, 100)
beta ~ normal(0, 100)
sigma ~ uniform(0, 100)
t_1 ~ uniform(t_min, t_max) # Restrict the change point to the range of the data

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model

# DCPR-1 model
T = 100
sigma = 1
alpha = c(-1,1) # Mean values before and after change point
set.seed(123)
t_1 = runif(1) # Time of change point
t = sort(runif(T))
mu = rep(NA,T)
mu[t<t_1] = alpha[1]
mu[t>=t_1] = alpha[2]
y = rnorm(T, mu, sigma)
plot(t,y)
lines(t[t<t_1], mu[t<t_1], col='red')
lines(t[t>=t_1], mu[t>=t_1], col='red')
abline(v=t_1,col='blue')

# Store this all together for later use
DCPR_1 = list(t=t, y=y, T=T, t_min = min(t), t_max = max(t))

# CCPR-1 model
T = 100
sigma = 1
alpha = 0
beta = c(2,-2) # Slopes before and after change point
set.seed(123)
t_1 = runif(1) # Time of change point
t = sort(runif(T))
mu = rep(NA,T)
mu[t<t_1] = alpha + beta[1] * (t[t<t_1] - t_1)
mu[t>=t_1] = alpha + beta[2] * (t[t>=t_1] - t_1)
y = rnorm(T, mu, sigma)
plot(t,y)
lines(t[t<t_1], mu[t<t_1], col='red')
lines(t[t>=t_1], mu[t>=t_1], col='red')
abline(v=t_1,col='blue')

# Store this all together for later use
CCPR_1 = list(t=t, y=y, T=T, t_min = min(t), t_max = max(t))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

# Code for DCPR-1
model_code_DCPR_1="
model
{
  # Likelihood
  for(i in 1:T) {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha[J[i]]
    # This is the clever bit - only pick out the right change point when above t_1
    J[i] <- 1 + step(t[i] - t_1)
  }

  # Priors
  alpha[1] ~ dnorm(0.0, 0.01)
  alpha[2] ~ dnorm(0.0, 0.01)
  t_1 ~ dunif(t_min, t_max)

  tau <- 1/pow(sigma, 2)
  sigma ~ dunif(0, 100)
}
"

# Code for CCPR-1
model_code_CCPR_1="
model
{
  # Likelihood
  for(i in 1:T) {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha + beta[J[i]]*(t[i]-t_1)
    # This is the clever bit - only pick out the right change point when above t_1
    J[i] <- 1 + step(t[i] - t_1)
  }

  # Priors
  alpha ~ dnorm(0.0, 0.01)
  beta[1] ~ dnorm(0.0, 0.01)
  beta[2] ~ dnorm(0.0, 0.01)
  t_1 ~ dunif(t_min, t_max)

  tau <- 1/pow(sigma, 2)
  sigma ~ dunif(0, 100)
}
"

# Choose the parameters to watch
model_parameters =  c("t_1", "alpha", "sigma")

# Run the model
model_run_DCPR_1 = jags(data = DCPR_1,
                        parameters.to.save = model_parameters,
                        model.file=textConnection(model_code_DCPR_1),
                        n.chains=4,
                        n.iter=1000,
                        n.burnin=200,
                        n.thin=2)


# Choose the parameters to watch
model_parameters =  c("t_1", "alpha", "beta", "sigma")

# Run the model
model_run_CCPR_1 = jags(data = CCPR_1,
                        parameters.to.save = model_parameters,
                        model.file=textConnection(model_code_CCPR_1),
                        n.chains=4,
                        n.iter=1000,
                        n.burnin=200,
                        n.thin=2)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run_DCPR_1)
plot(model_run_DCPR_1)

# Plot the data agin with the estimated change point
t_1 = model_run_DCPR_1$BUGSoutput$sims.list$t_1
with(DCPR_1, plot(t, y))
abline(v = mean(t_1), col='red')

# Now the CCPR run
print(model_run_CCPR_1) # Not a great job here - CP is pretty uncertain
plot(model_run_CCPR_1) # Not a great job here - CP is pretty uncertain

# Plot the data agin with the estimated change point
t_1 = model_run_CCPR_1$BUGSoutput$sims.list$t_1
with(CCPR_1, plot(t, y))
abline(v = mean(t_1), col='red')

# Real example ------------------------------------------------------------

# Run the CCPR-1 model on the HadCrut data
hadcrut = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/hadcrut.csv')
head(hadcrut)
with(hadcrut,plot(Year,Anomaly,type='l'))
# Where is the change point?

# Set up the data
real_data = with(hadcrut,
                 list(T = nrow(hadcrut),
                      y = hadcrut$Anomaly,
                      t = hadcrut$Year,
                      t_min = min(hadcrut$Year),
                      t_max = max(hadcrut$Year)))

# Run the model - this can struggle to converge so needs a longer run
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code_CCPR_1),
                     n.chains=4,
                     n.iter=10000,
                     n.burnin=2000,
                     n.thin=8)

# Plot output
print(real_data_run) # Not great R-hat values - might need a longer run

# Plot the change point
with(hadcrut,
     plot(Year,
          Anomaly,
          type='l'))
t_1_mean = mean(real_data_run$BUGSoutput$sims.list$t_1)
abline(v = t_1_mean, col='red')

# Include the slopes before and after
alpha_mean = mean(real_data_run$BUGSoutput$sims.list$alpha)
beta_1_mean = mean(real_data_run$BUGSoutput$sims.list$beta[,1])
beta_2_mean = mean(real_data_run$BUGSoutput$sims.list$beta[,2])

with(hadcrut,
     lines(Year[Year<t_1_mean], alpha_mean + beta_1_mean * ( Year[Year<t_1_mean] - t_1_mean ) ) )
with(hadcrut,
     lines(Year[Year>=t_1_mean], alpha_mean + beta_2_mean * ( Year[Year>=t_1_mean] - t_1_mean ) ) )

# Other tasks -------------------------------------------------------------

# 1) For the plots of the output from the simulated data runs, see if you can add in more information than just the posterior means. You might try to include the 95% CI, or see if you can put in a full histogram or density plot of the posterior distribution. For the real data example, see if you can add in uncertainty in the predicted lines either side of the change point
# 2) (Harder) Jags code for 2, 3, and 4 change point models is available at http://iopscience.iop.org/1748-9326/10/8/084002/media/Rcode_CPA.R. Use this code (or try yourself if you fancy a challenge) and see if you can get multiple change point models running on the real data.
# 3) (Much harder) It would be great if were able to estimate K (the number of change points) as another parameter in the model. Can you think of a way of doing this?
