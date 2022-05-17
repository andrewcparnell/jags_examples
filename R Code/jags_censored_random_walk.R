# Header ------------------------------------------------------------------

# A censored random walk model in continuous time
# Andrew Parnell

# In this code I fit a random walk model where some of the data are left and/or right censored
# The data are fixed so that both types of censoring occur
# A useful resource for this file was:
# https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-021-04496-8.pdf
# and https://github.com/xinyue-qi/Censored-Data-in-JAGS
# But their files seem wrong so I ammended it and it seems to work better

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
# mu ~ dnorm(0, 100) - vague

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

  # Left censored data
  for (t in 1:T_lcens){
    Zl[t] ~ dbern(pl[t])
		pl[t] <- pnorm(l_cuts[t], mu[t_order[T_obs + t]], sigma^-2)
  }

	# Right censored data
  for (t in 1:T_rcens){
    Zr[t] ~ dbern(pr[t])
		pr[t] <- 1 - pnorm(r_cuts[t], mu[t_order[T_obs + T_lcens + t]], sigma^-2)
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
# lim <- matrix(NA, nrow = sum(obs == 0), ncol = 2)
# for (j in 1:nrow(lim)) {
#   if(l_censored[obs==0][j] == 1) lim[j,] <- c(l, Inf)
#   if(r_censored[obs==0][j] == 1) lim[j,] <- c(-Inf, r)
# }
l_cuts <- l * l_censored[l_censored == 1]
r_cuts <- r * r_censored[r_censored == 1]

# Get the time order right
t_order <- order(order(c(t[obs == 1], t[obs == 0])))

model_data <- list(y = c(y[obs == 1], rep(NA, sum(obs == 0))),
                   T = length(y),
                   T_obs = length(y[obs == 1]),
                   l_cuts = l_cuts,
                   r_cuts = r_cuts,
                   T_lcens = length(l_cuts),
                   T_rcens = length(r_cuts),
                   Zl = rep(1, length(l_cuts)),
                   Zr = rep(1, length(r_cuts)),
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

