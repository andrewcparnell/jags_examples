# Header ------------------------------------------------------------------

# Ornstein–Uhlenbeck (OU) process model in jags
# Andrew Parnell

# The OU process isthe continuous time equivalent of the AR(1) discrete time process

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Notation:
# y(t) = response variable observated at continuous times t
# alpha = optional drift parameter
# theta = autoregression parameter
# sigma = standard deviation/volatility parameter

# Likelihood:
# Sometimes written as dy = theta * ( alpha - y ) dt + sigma * dW(t)
# More helpful version:
# y(t) - y(t-s) ~ N( theta * ( alpha - y(t-s) ) * s, s * sigma^2 )
# where s is any positive value

# Prior
# alpha ~ N(0, 100)
# theta ~ U(0, 100) - can be outside the range 0, 1
# sigma ~ U(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
alpha = 0
theta = 0.6 # Note if theta = 0 you end up with BM
sigma = 0.1
y = rep(NA, T)
y[1] = 0
set.seed(123)
t = sort(runif(T, 0, 1)) # Assume time runs from 0 to 1
for(i in 2:T) y[i] = y[i-1] + rnorm(1, theta * (alpha - y[i-1]) * ( t[i] - t[i-1] ), sigma * sqrt(t[i] - t[i-1]))
plot(t, y, type = 'l')
axis(side=1,at=t,labels=rep('',length(t))) # Add in ticks for time values

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 2:T) {
    y[i] ~ dnorm( theta * (alpha - y[i-1]) * (t[i] - t[i-1]) + y[i-1], tau[i] )
    tau[i] <- 1/( pow(sigma,2) * (t[i] - t[i-1]) )
  }

  # Priors
  alpha ~ dnorm(0, 0.01)
  theta ~ dunif(0, 100)
  sigma ~ dunif(0.0, 10.0)
}
'

# Set up the data
model_data = list(T = T, y = y, t = t)

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


# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run) # Model seems to struggle to estimate both theta and alpha
plot(model_run)

# Check correlation of parameters
cor(model_run$BUGSoutput$sims.matrix) # Not that strong a correlation between theta and alpha

# Real example ------------------------------------------------------------

# Run on the Monticchio mean temperature of coldest month data
mont = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/Monticchio_MTCO.csv')
with(mont, plot(Age, MTCO, type='l'))

# Remove duplpicate times - model will fail
dup_times = which(diff(mont$Age)==0)
mont2 = mont[-dup_times,]

# Use the trick in jags_BM to estimate the model and get predictions on a new
# grid
t_ideal = seq(100+0.5,max(mont2$Age)+0.5, by = 500) # 500 year regular grid
# Note added on 0.01 to the above to stop there being some zero time differences
y_ideal = rep(NA, length(t_ideal))
t_all = c(mont2$Age, t_ideal)
y_all = c(mont2$MTCO, y_ideal)
o = order (t_all)

# Create new data set
real_data = with(mont,
                 list(y = y_all[o], T = length(y_all), t = t_all[o]))

# Save all the values of y
model_parameters = c('y', 'alpha', 'theta', 'sigma')

# Run the model - if the below is slow to run try reducing the time grid above
real_data_run = jags(data = real_data,
                       parameters.to.save = model_parameters,
                       model.file=textConnection(model_code),
                       n.chains=4,
                       n.iter=10000,
                       n.burnin=2000,
                       n.thin=8)

plot(real_data_run)

par(mfrow=c(1,3))
hist(real_data_run$BUGSoutput$sims.list$alpha, breaks=30)
hist(real_data_run$BUGSoutput$sims.list$theta, breaks=30)
hist(real_data_run$BUGSoutput$sims.list$sigma, breaks=30)
par(mfrow=c(1,1))

# Now create a plot of the gridded predicted values
pick_out = which( is.na(real_data$y) )
pred_y = apply(real_data_run$BUGSoutput$sims.list$y[, pick_out], 2, 'mean')

plot(t_ideal, pred_y, type = 'l')


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Play around with simulating series with different values of theta. What happens when theta get patricularly large or close to zero? (hint: look at the equation - what happens when theta=0?)
# 2) Try fitting the model to the real data using different subsets (e.g. the last 10,000 years). Do the results change substantially between periods?
# 3) (harder) The posterior distribution of theta here is almost zero. Use DIC to compare between the full O-U model and a model with theta = 0



