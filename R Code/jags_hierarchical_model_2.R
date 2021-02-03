# Header ------------------------------------------------------------------

# Fitting a nested hierarchical linear model in JAGS
# Andrew Parnell

# In this code we generate some data from a nested hierarchical model and fit it using JAGS. We then interpret the output

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_{ijk} = repsonse variable for observation i=1,..,n_{jk} in group j = 1,..,M_1, and sub-group k = 1, .., M_2
# The observations are nested, i.e. all observations in the same sub-group must also be in the same group
# N = total number of observation = sum_{j,k} n_{j,k}
# alpha = overall mean parameter
# b_j = random effect for group j
# c_{jk} = nested random effect for sub-group k in group j
# sigma = residual standard deviation
# sigma_b = standard deviation between groups
# sigma_c = standard deviation between sub-groups

# Likelihood:
# y_{ijk} ~ N(alpha + b_j + c_{jk}, sigma^2)
# Prior
# alpha ~ N(0, 100) - vague priors
# b_j ~ N(0, sigma_b^2)
# c_{jk} ~ N(0, sigma_c^2)
# sigma ~ half-cauchy(0, 10)
# sigma_b ~ half-cauchy(0, 10)
# sigma_c ~ half-cauchy(0, 10)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
M1 <- 5 # Number of groups
M2 <- 3 # Number of sub-groups
alpha <- 2
sigma <- 1
sigma_b <- 3
sigma_c <- 2
# Set the seed so this is repeatable
set.seed(123)
# The below provides the number of observations in each group and sub-group
njk <- matrix(sample(10:20, M1 * M2, replace = TRUE), ncol = M2, nrow = M1)
N <- sum(njk)
b <- rnorm(M1, 0, sigma_b)
c <- matrix(rnorm(M1 * M2, 0, sigma_c), ncol = M2, nrow = M1)
group <- rep(1:M1, times = rowSums(njk))
# Now create a vector which links the sub-groups to the observations
subgroup <- NULL
for (i in 1:M1) {
  subgroup <- c(subgroup, rep(1:M2, times = njk[i, ]))
}
y <- rep(NA, N)
for (i in 1:N) {
  y[i] <- rnorm(1, mean = alpha + b[group[i]] + c[group[i], subgroup[i]], sd = sigma)
}

# Also creat a plot
boxplot(y ~ group + subgroup)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(alpha + b[group[i]] + c[group[i], subgroup[i]], sigma^-2)
  }

  # Priors
  alpha ~ dnorm(0, 100^-2)
  for (j in 1:M1) {
    b[j] ~ dnorm(0, sigma_b^-2)
    for(k in 1:M2) {
      c[j, k] ~ dnorm(0, sigma_c^-2)
    }
  }
  sigma ~ dt(0, 10^-2, 1)T(0,)
  sigma_b ~ dt(0, 10^-2, 1)T(0,)
  sigma_c ~ dt(0, 10^-2, 1)T(0,)
}
"

# Set up the data
model_data <- list(N = N, y = y, M1 = M1, M2 = M2, group = group, subgroup = subgroup)

# Choose the parameters to watch
model_parameters <- c("alpha", "b", "c", "sigma", "sigma_b", "sigma_c")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
# traceplot(model_run)

# Get the posterior samples
post <- model_run$BUGSoutput$sims.list

# Compare alpha with true value
hist(post$alpha, breaks = 30)
abline(v = alpha, col = "red")

# Comapre b with true values
par(mfrow = c(M1, 1))
for (i in 1:M1) {
  hist(post$b[, i], breaks = 30)
  abline(v = b[i], col = "red")
}

# Comapre b with true values
par(mfrow = c(M1, M2))
for (i in 1:M1) {
  for (j in 1:M2) {
    hist(post$c[, i, j], breaks = 30)
    abline(v = c[i, j], col = "red")
  }
}

# Real example ------------------------------------------------------------
