# Header ------------------------------------------------------------------

# Fitting a exponential regression in JAGS
# Bruna Wundervald

# In this file we fit a Exponential model for survival analysis
library(R2jags)
library(tidyverse)
library(patchwork) # devtools::install_github("thomasp85/patchwork")

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
set.seed(123)
T <- 1000
x_1 <- rnorm(n = T, mean = 1, sd = 1)
x_2 <- sample(c(0:1), size = T, replace = TRUE)
lambda_0 <- 0.3
beta_1 <- 1.2
beta_2 <- 1
mu <- exp(beta_1 * x_1 + beta_2 * x_2)
t <- rexp(n = T, rate = lambda_0 * mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
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
"
# Set up the data
model_data <- list(T = T, t = t, x_1 = x_1, x_2 = x_2)

# Choose the parameters to watch
model_parameters <- c("beta_1", "beta_2", "lambda_0")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Plotting the results -------------------
pars <- model_run$BUGSoutput$summary[c(1, 2, 4)]

x_1 <- sort(x_1)

surv_func <- function(i, trt) {
  if (trt == 0) {
    H <- pars[3] * exp(pars[1] * x_1[i] + pars[2] * 0)
  } else {
    H <- pars[3] * exp(pars[1] * x_1[i] + pars[2] * 1)
  }

  # Survival function
  S <- exp(-t * H)
  return(S)
}

survs_trt1 <- c(1, 50, 100, 250, 500, 1000) %>%
  purrr::map(surv_func, trt = 0) %>%
  purrr::map_dfc(enframe) %>%
  select_if(str_detect(colnames(.), "value")) %>%
  set_names(paste0(
    "x_1 = ", round(x_1[c(1, 50, 100, 250, 500, 1000)], 1)
  )) %>%
  tidyr::gather(value = "surv") %>%
  dplyr::mutate(time = rep(t, 6))

survs_trt2 <- c(1, 50, 100, 250, 500, 1000) %>%
  purrr::map(surv_func, trt = 1) %>%
  purrr::map_dfc(enframe) %>%
  select_if(str_detect(colnames(.), "value")) %>%
  set_names(paste0(
    "x_1 = ", round(x_1[c(1, 50, 100, 250, 500, 1000)], 1)
  )) %>%
  tidyr::gather(value = "surv") %>%
  dplyr::mutate(time = rep(t, 6))

p1 <- survs_trt1 %>%
  ggplot(aes(y = surv, x = time, group = key)) +
  geom_line(aes(colour = key)) +
  labs(
    y = "Survival Function", x = "Time",
    colour = "Covariables at",
    title = "Treatment 1"
  ) +
  xlim(0, 35) +
  theme_bw()

p2 <- survs_trt2 %>%
  ggplot(aes(y = surv, x = time, group = key)) +
  geom_line(aes(colour = key)) +
  labs(
    y = "Survival Function", x = "Time",
    colour = "Covariables at",
    title = "Treatment 2"
  ) +
  xlim(0, 35) +
  theme_bw()

p1 + p2 + plot_layout(ncol = 1)

# Real example ------------------------------------------------------------

df <- survival::veteran %>%
  select(time, age, trt)

# The data is a randomised trial of two treatment regimens
# for lung cancer. We consider the covariables age and treatment.


# Set up the data
model_data <- list(T = T, t = df$time, x_1 = df$age, x_2 = df$trt)

# Choose the parameters to watch
model_parameters <- c("beta_1", "beta_2", "lambda_0")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)


# Plotting the results -------------------
pars <- model_run$BUGSoutput$summary[c(1, 2, 4)]
surv_func <- function(i, trt) {
  if (trt == 1) {
    H <- pars[3] * exp(pars[1] * df$age[i] + pars[2] * 0)
  } else {
    H <- pars[3] * exp(pars[1] * df$age[i] + pars[2] * 1)
  }
  # Survival function
  S <- exp(-df$time * H)
  return(S)
}


survs_trt1 <- c(1, 10, 20, 50, 100, 137) %>%
  purrr::map(surv_func, trt = 1) %>%
  purrr::map_dfc(enframe) %>%
  select_if(str_detect(colnames(.), "value")) %>%
  set_names(paste0(
    "Age = ", round(df$age[c(1, 10, 20, 50, 100, 137)], 1)
  )) %>%
  tidyr::gather(value = "surv") %>%
  dplyr::mutate(time = rep(df$time, 6))

survs_trt2 <- c(1, 10, 20, 50, 100, 137) %>%
  purrr::map(surv_func, trt = 2) %>%
  purrr::map_dfc(enframe) %>%
  select_if(str_detect(colnames(.), "value")) %>%
  set_names(paste0(
    "Age = ", round(df$age[c(1, 10, 20, 50, 100, 137)], 1)
  )) %>%
  tidyr::gather(value = "surv") %>%
  dplyr::mutate(time = rep(df$time, 6))

p1 <- survs_trt1 %>%
  ggplot(aes(y = surv, x = time, group = key)) +
  geom_line(aes(colour = key)) +
  labs(
    y = "Survival Function", x = "Time",
    colour = "Covariables at", title = "Treatment 1"
  ) +
  xlim(0, 200) +
  theme_bw()


p2 <- survs_trt2 %>%
  ggplot(aes(y = surv, x = time, group = key)) +
  geom_line(aes(colour = key)) +
  labs(
    y = "Survival Function", x = "Time",
    colour = "Covariables at", title = "Treatment 2"
  ) +
  xlim(0, 200) +
  theme_bw()

p1 + p2 + plot_layout(ncol = 1)
