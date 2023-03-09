# Header ------------------------------------------------------------------

# A censored random walk model in continuous time
# Andrew Parnell

# In this code I fit a random walk model where some of the data are left and/or right censored
# The data are fixed so that both types of censoring occur
# A useful resource for this file was:
# https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-021-04496-8.pdf
# and https://github.com/xinyue-qi/Censored-Data-in-JAGS
# But their files seem a bit strange so I amended it and it seems to work better

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t ~ U(0, 1)
# mu(t) = mean at time t
# sigma_b = RW standard deviation per unit time
# sigma = observation standard deviation
# h = arbitrary time gap between observations

# Likelihood:
# y(t) ~ N(mu(t), sigma^2)
# mu(t) ~ N(mu(t - h), h * sigma_b^2)
# with also the censoring pattern:
# If y(t) > r then y(t) = r
# If y(t) < l then y(t) = l
# l and r are known here
# Prior:
# sigma ~ unif(0, 100) - vague
# sigma_b ~ dgamma(a, b) ~ informative if high values for a and b provided

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(1)
T <- 100
sigma <- 0.1
sigma_b <- 1.5
r <- 1.5
l <- 0
t <- sort(runif(T))
h <- diff(t)
mu <- cumsum(rnorm(T, 0, sqrt(h)*sigma_b))
y_true <- rnorm(T, mu, sigma)
y <- y_true
y[y > r] <- r # Right censoring
y[y < l] <- l # Left censoring
plot(t, y_true, pch = 19)
points(t, y, col = 'blue')
lines(t, mu, col = 'red')
r_censored <- as.integer(y_true > r)
l_censored <- as.integer(y_true < l)
obs <- 1 - r_censored - l_censored

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood for observations
  for (t in 1:T) {
    y[t] ~ dnorm(mu[t_order[t]], sigma^-2)
  }

  # Left censored data has Z = 1, right censored has Z = 0
  for (t in 1:T_cens){
    Z[t] ~ dbern(p[t])
		p[t] <- pnorm(cuts[t], mu[t_order[T_obs + t]], sigma^-2)
  }

  # Prior on mu
  mu[1] ~ dnorm(0, 1^-2)
  for (t in 2:T) {
    mu[t] ~ dnorm(mu[t-1], (h[t-1]^-1) * (sigma_b^-2))
  }

  # Priors on hyper-parameters
  sigma ~ dgamma(1, 1)
  sigma_b ~ dgamma(15, 10)
}
"

# Set up the data
cuts <- c(rep(l, sum(l_censored)), rep(r, sum(r_censored)))
Z <- c(rep(1, sum(l_censored)), rep(0, sum(r_censored)))

# Get the time order right
t_order <- order(order(c(t[obs == 1], t[obs == 0])))

model_data <- list(y = c(y[obs == 1], rep(NA, sum(obs == 0))),
                   T_obs = length(y[obs == 1]),
                   T = length(y),
                   cuts = cuts,
                   T_cens = length(cuts),
                   Z = Z,
                   h = diff(t),
                   t_order = t_order)

# Choose the parameters to watch
model_parameters <- c("mu", "sigma", "sigma_b", "y")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)
plot(model_run)

plot(t, y_true, pch = 19)
lines(t, model_run$BUGSoutput$mean$mu, col = 'blue')
lines(t, apply(model_run$BUGSoutput$sims.list$mu, 2, 'quantile', 0.25), col = 'blue', lty = 'dotted')
lines(t, apply(model_run$BUGSoutput$sims.list$mu, 2, 'quantile', 0.75), col = 'blue', lty = 'dotted')
lines(t, mu, col = 'red')

# Real example ------------------------------------------------------------

