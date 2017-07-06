# Header ------------------------------------------------------------------

# Moving average (MA) models
# Andrew Parnell

# Some jags code to fit a moving average (MA) model of order q.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = response variables, times t = 1,...,T
# alpha = mean parameter
# eps_t = residual at time t
# theta = MA parameters
# sigma = residual standard deviation
# q = order of the moving average (fixed)
# Likelihood for an MA(q) model:
# y_t ~ N(alpha + theta_1 ept_{t-1} + ... + theta_q eps_{t-q}, sigma)
# Prior
# alpha ~ normal(0,100) # Vague
# sigma ~ uniform(0,10)
# theta[q] ~ normal(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
q = 1 # Order
T = 100
sigma = 1
alpha = 0
set.seed(123)
theta = runif(q)
y = rep(NA,T)
y[1:q] = rnorm(q,0,sigma)
eps = rep(NA,T)
eps[1:q] = y[1:q] - alpha
for(t in (q+1):T) {
  y[t] = rnorm(1, mean = alpha + sum(theta*eps[(t-q):(t-1)]), sd = sigma)
  eps[t] = y[t] - alpha - sum(theta*eps[(t-q):(t-1)])
}
plot(1:T,y,type='l')

# Jags code ---------------------------------------------------------------

# This code to fit a general MA(q) model
model_code = '
model
{
  # Set up residuals
  for(t in 1:q) {
    eps[t] <- y[t] - alpha
  }
  # Likelihood
  for (t in (q+1):T) {
    y[t] ~ dnorm(mean[t], tau)
    mean[t] <- alpha + inprod(theta, eps[(t-q):(t-1)])
    eps[t] <- y[t] - alpha - inprod(theta, eps[(t-q):(t-1)])
  }

  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:q) {
    theta[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

# Set up the data
model_data = list(T = T, y = y, q = 1)

# Choose the parameters to watch
model_parameters =  c("alpha","theta","sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=1000, # Number of iterations
                 n.burnin=200, # Number of iterations to remove at start
                 n.thin=2) # Amount of thinning


# Jags code to fit the model to the simulated data

# Simulated results -------------------------------------------------------

print(model_run) # Parameter theta should match the true value

# Real example ------------------------------------------------------------

hadcrut = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/hadcrut.csv')
head(hadcrut)
with(hadcrut,plot(Year,Anomaly,type='l'))

# Look at the ACF/PACF
acf(hadcrut$Anomaly)
pacf(hadcrut$Anomaly)

# Set up the data
real_data = with(hadcrut,
                 list(T = nrow(hadcrut),
                      y = hadcrut$Anomaly,
                      q = 5))

# Run the model
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
                     n.iter=1000,
                     n.burnin=200,
                     n.thin=2)

# Plot output
print(real_data_run) # Very high degree of autocorrelation

# Plot some of the fitted values (also known as one-step-ahead predictions)
post = print(real_data_run)
alpha_mean = post$mean$alpha
theta_mean = post$mean$theta

# Create fitted values
eps_fit = y_fit = rep(NA,real_data$T)
eps_fit[1:real_data$q] = y[1:real_data$q] - alpha_mean
y_fit[1:real_data$q] = alpha_mean
for (t in (real_data$q+1):real_data$T) {
  eps_fit[t] = real_data$y[t] -
    alpha_mean -
    sum(theta_mean * eps_fit[(t-real_data$q):(t-1)])
  y_fit[t] = alpha_mean + sum(theta_mean * eps_fit[(t-real_data$q):(t-1)])
}

# Create fitted line
with(hadcrut, plot(Year, Anomaly, type='l'))
with(hadcrut, lines(Year, y_fit, col='red'))
# First bit might look strange due to warm-up but then should be ok

# Create some predictions off into the future
T_future = 2050
future_values = rep(NA, T_future-max(hadcrut$Year))
# To create future values, need to append on residuals from end of data set
future_eps = rep(NA, T_future-max(hadcrut$Year) + real_data$q)
future_eps[1:real_data$q] = eps_fit[real_data$T:(real_data$T-real_data$q+1)]
for (t in 1:length(future_values)) {
  future_values[t] = alpha_mean + sum(theta_mean * future_eps[(t+real_data$q-1):(t)])
  future_eps[t+real_data$q] = future_values[t] - alpha_mean - sum(theta_mean * future_eps[(t+real_data$q-1):(t)])
}
# Most of the above code, isn't necessary, but shows why the MA(q) model quickly hits its mean as the future prediction

# Plot these all together
with(hadcrut,
     plot(Year,
          Anomaly,
          type='l',
          xlim=c(min(hadcrut$Year),T_future),
          ylim=range(c(hadcrut$Anomaly,future_values))))
lines(((max(hadcrut$Year)+1):T_future),future_values,col='red')
# Will quickly head down to the mean in q steps

# Other tasks -------------------------------------------------------------

# 1) Try changing the order q in the real data example. How do the fitted values change? What seems to be a good value of q?
# 2) A good way to test the model is to remove some of the last few data points and check the predictions. Try remove the last 5 data points, fitting the model to the remaining data, and then producing the predictions.
# 3) (Harder) See if you can combine the AR and MA models together into an ARMA model. Note that this is covered in the jags_ARMA.R code file so try not to look at that first!