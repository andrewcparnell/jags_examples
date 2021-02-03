# Header ------------------------------------------------------------------

# P-spline model in JAGS - alternative parameterisation
# Andrew Parnell and Niamh Cahill

# This file fits a spline regression model to data in JAGS, and produces predictions/forecasts
# It uses an alternative method for calculating the penalty

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls())
library(R2jags)
library(MASS) # Useful for mvrnorm function
library(splines) # Useful for creating the B-spline basis functions

# Maths -------------------------------------------------------------------

# Notation:
# y(t): Response variable at time t, defined on continuous time
# y: vector of all observations
# B: design matrix of spline basis functions
# beta; spline weights
# sigma: residual standard deviation parameter (sometimes known in the GP world as the nugget)
# sigma_b: spline random walk parameter

## This uses a different implementation for the penalisation so that we can put a prior directly on the coefficient differences
## Eilers (1999) proposed Z = BD'(DD')^(-1) (where D is the differencing matrix)
## then instead of y = B*beta, we have y = alpha + Z*delta
## where delta are the first order differences
## this works much better for convergence

# Likelihood:
# y ~ N(alpha + Z%*%delta, sigma^2)
# delta ~ N(0, sigma_b^2)

# Priors
# sigma ~ cauchy(0, 10)
# sigma_b ~ cauchy(0, 10)

# Useful function ---------------------------------------------------------

# These functions create the B-spline basis functions

# A function that uses the bs() function to generate the B-spline basis functions
# following Eilers and Marx 'Craft of smoothing' course. This bs_bbase() function
# is equivalent to the bbase() function available at http://statweb.lsu.edu/faculty/marx/

bs_bbase <- function(x, xl = min(x), xr = max(x), nseg = 10, deg = 3) {
  # Compute the length of the partitions
  dx <- (xr - xl) / nseg
  # Create equally spaced knots
  knots <- seq(xl - deg * dx, xr + deg * dx, by = dx)
  # Use bs() function to generate the B-spline basis
  get_bs_matrix <- matrix(bs(x, knots = knots, degree = deg, Boundary.knots = c(knots[1], knots[length(knots)])), nrow = length(x))
  # Remove columns that contain zero only
  bs_matrix <- get_bs_matrix[, -c(1:deg, ncol(get_bs_matrix):(ncol(get_bs_matrix) - deg))]

  return(bs_matrix)
}

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(123)
N <- 100 # Number of observations
x <- sort(runif(N, 0, 10)) # Create some covariate values
B <- bs_bbase(x, nseg = 30)
alpha <- 1
sigma_b <- 0.5 # Parameters as above
sigma <- 0.2
delta <- rnorm(ncol(B) - 1, 0, sigma_b)

# Create the differencing matrix
D <- diff(diag(ncol(B)), diff = 1)
Q <- t(D) %*% solve(D %*% t(D))
Z <- B %*% Q

y <- rnorm(N, mean = alpha + Z %*% delta, sd = sigma)
plot(x, y)
lines(x, alpha + Z %*% delta, col = "red") # True line

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(alpha + inprod(Z[i,], delta), sigma^-2)
  }

  # RW prior on delta
  alpha ~ dnorm(0, 10^-2)
  for (i in 1:N_knots) {
    delta[i] ~ dnorm(0, sigma_b^-2)
  }

  # Priors on beta values
  sigma ~ dt(0, 10^-2, 1)T(0,)
  sigma_b ~ dt(0, 10^-2, 1)T(0,)

}
"

# Set up the data
model_data <- list(N = N, y = y, Z = Z, N_knots = ncol(Z))

# Choose the parameters to watch
model_parameters <- c("alpha", "delta", "sigma", "sigma_b")

# Run the model - can be slow
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)
plot(model_run)

# Get the posterior delta and 50% prediction interval
alpha_post <- model_run$BUGSoutput$sims.list$alpha
delta_post <- model_run$BUGSoutput$sims.list$delta
sigma_post <- model_run$BUGSoutput$sims.list$sigma

# object to store prediction interval
PI <- matrix(nrow = length(alpha_post),
               ncol = N)
for(i in 1:nrow(pred)) {
  PI[i,] <- rnorm(N,
                  mean = alpha_post[i] + Z%*%delta_post[i,],
                  sd = sigma_post[i])
}
PI_quantile <- apply(PI, 2, quantile, prob = c(0.25,0.5,0.75))

# Plot the output with uncertainty bands
plot(x, y)
lines(x, alpha + Z %*% delta, col = "red") # True line
lines(x, PI_quantile[2,], col = "blue") # Predicted line
lines(x, PI_quantile[1,], col = "blue", lty = 2) # Predicted low
lines(x, PI_quantile[3,], col = "blue", lty = 2) # Predicted high
legend("topleft", c(
  "True line",
  "Posterior lines (with 50% CI)",
  "Data"
),
lty = c(1, 1, -1),
pch = c(-1, -1, 1),
col = c("red", "blue", "black")
)

# Create some new predictions on a grid of new values
# Needs to be in the same range as the previous values (if not you need to go back to the creation of B above)
x_new <- seq(min(x), max(x), length = 1000)
B_new <- bs_bbase(x_new, xl = min(x), xr = max(x), nseg = 30)
D_new <- diff(diag(ncol(B_new)), diff = 1)
Q_new <- t(D_new) %*% solve(D_new %*% t(D_new))
Z_new <- B_new %*% Q_new

plot(x, y)
lines(x_new, alpha_quantile[2] + Z_new %*% delta_quantile[2, ], col = "blue") # Beautifully smooth

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
