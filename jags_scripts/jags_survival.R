# Header ------------------------------------------------------------------

# Fitting a exponential regression in JAGS
# Bruna Wundervald 

# In this file we fit a Exponential model for survival analysis
library(R2jags)
library(tidyverse)

# Description of the Bayesian Poisson model
# Notation:
# t_i = exponencial response variable for observation i = 1,...,N
# x_{1i} = first explanatory variable for observation i
# x_{2i} = second " " " " " " " " "
# mu_t = the rate of occurence for each random variable t_i
# beta_1 = parameter value for explanatory variable 1
# beta_2 = parameter value for explanatory variable 2

# Likelihood
# t_i ~ Exp(mu_i)
# log(mu_i) = beta_1 * x_1[i] + beta_2 * x_2[i]

# Priors - all vague
# lambda_0 ~ gamma(1, 1)
# beta_1 ~ normal(0, 100)
# beta_2 ~ normal(0, 100)

# Simulate data -----------------------------------------------------------

T = 1000
x_1 <- rnorm(n = T, mean = 1, sd = 1)
x_2 <- rnorm(n = T, mean = 1, sd = 1)
lambda_0 <- 0.3
beta_1 <- 1.2
beta_2 <- 1
mu <- exp(beta_1 * x_1 + beta_2 * x_2)
t <- rexp(n = T, rate = lambda_0 * mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:T) {
    mu[i] = exp(beta_1 * x_1[i] + beta_2 * x_2[i])
    t[i] ~ dexp(mu[i] * lambda_0)
    
  }
  # Priors
  lambda_0 ~ dgamma(1, 1)
  beta_1 ~ dnorm(0.0, 0.01)
  beta_2 ~ dnorm(0.0, 0.01)
}
'
# Set up the data
model_data = list(T = T, t = t, x_1 = x_1, x_2 = x_2)

# Choose the parameters to watch
model_parameters =  c("beta_1", "beta_2", "lambda_0")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code),
                 n.chains = 4,
                 n.iter = 1000,
                 n.burnin = 200,
                 n.thin = 2)

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# -------------------
pars <- model_run$BUGSoutput$summary[c(1, 2, 4)]
H <- t*pars[3] * exp(pars[1] * x_1 + pars[2] * x_2)
# Survival function 
S <- 1 - exp(-H)

df <- data.frame(S, t, x_1, x_2, H, pred_1 = pars[3] * exp(pars[1] * x_1)) 

df %>% 
  ggplot(aes(t, S)) +
  geom_point(colour = 'orange') +
  theme_bw()