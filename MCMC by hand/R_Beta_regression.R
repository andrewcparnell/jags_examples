# ---------------------------------------------------------------- #
# Description: Metropolis-Hastings for Bayesian Beta regression    #
#              with a multivariate Normal as prior for betas and   #
#              an uniform for phi                                  #
#  Author: Alan Inglis                                             #
# ---------------------------------------------------------------- #

library(MASS)
library(ggplot2)
library(gridExtra)
library(progress)

## Defining simulation size, burn-in, etc
## --------------------------------------
set.seed(123)
Niter <- 10000
BurnIn <- 5000
TotIter <- Niter+BurnIn
N <- 1000
j_phi <- 0
j_beta <- 0
alpha <- -1
beta <- 0.2
phi <- 5
P <- c(alpha, beta, phi)
AuxBurnIn <- 1

## Simulation scheme 
## -----------------
x0 <- rep(1,N)
x1 <- runif(N, 0, 10)
logit_mu <- alpha + beta * x1
mu <- exp(logit_mu)/(1+exp(logit_mu)) # inverse logit
a <- mu * phi
b <- (1 - mu) * phi
X <- cbind(x0,x1)
y <- rbeta(N, a, b)

# Prior distributions and the values of the hyperparameters
#------------------------------------------------------------
# Beta ~ N(m, V)
m <- rbind(0,0)
V   <- diag(2)*100.0
V_1 <- solve(V)

# Sigma ~ U(a, b)
a = 0
b = 100

## Data frame that will store MCMC values 
## ------------------------------------------------
SaveResults <- as.data.frame(matrix(data = NA, nrow = Niter, ncol = length(P)+1))
colnames(SaveResults) <- c('Iter', 'Alpha', 'Beta', 'Phi')

# Starting points
#------------------------------------------------------------------------------
phi = 10
MCMCBetasI <- c(0,0)
phi_I <- 2
Vphi <- 0.04 # variance of the proposal distribution for phi
mprop <- diag(2)*0.001

## Algorithm 
## -----------------------------------
pb <- progress_bar$new(
 format = "  iterations [:bar] :percent done",
 total = TotIter, clear = FALSE, width= 60) # Displays a progress bar for iterations

func_mu = function(beta){
  return(exp(X%*%beta) / (1 + exp(X%*%beta)))
}

# Since the proposal distribution for phi is a Gamma, the ideia is as following:
# mean = alpha/beta
# var = alpha/beta^2; Thus
# alpha = mean^2 / var;
# beta = mean/var.    
# Consider that phi_I = mean and var = var, where var is set up below.

var = 0.5 # Controlling the variance of the proposal distribution for phi_I. Thus, it is possible to control the acceptance rate of phi_I.

for(i in 1:TotIter){
  pb$tick()

  MCMCBetasC <- mvrnorm(1, MCMCBetasI, mprop)
  mu_I <- func_mu(MCMCBetasI)
  muC  <- func_mu(MCMCBetasC)
  
  # Computing the ratio stated in the Metropolis-Hastings algorithm
  ratio_beta = ((- sum(log(gamma(muC * phi_I))) - sum(log(gamma((1 - muC) * phi_I))) + sum((muC * phi_I - 1) * log(y)) 
                 + sum(((1 - muC) * phi_I - 1) *  log(1 - y)) - 0.5 * (t(MCMCBetasC - m) %*% V_1 %*%(MCMCBetasC - m))) - 
                (- sum(log(gamma(mu_I * phi_I))) - sum(log(gamma((1 - mu_I) * phi_I))) + sum((mu_I * phi_I - 1) * log(y))
                 + sum(((1 - mu_I) * phi_I - 1) * log(1 - y)) - 0.5 * (t(MCMCBetasI - m) %*% V_1 %*%(MCMCBetasI - m))))         
  
   # Accept/Reject step for beta
  if(runif(1) < min(1, exp(ratio_beta)))
  {MCMCBetasI <- MCMCBetasC
  j_beta <- j_beta +1}
  
  # Proposal distribution for phi
  phiC = rgamma(1, (phi_I^2)/var, scale=1/(phi_I / var))
 
  # Computing the ratio stated in the Metropolis-Hastings algorithm
  ratio_phi = ((N * log(gamma(phiC)) - sum(log(gamma(mu_I * phiC))) - sum(log(gamma((1 - mu_I) * phiC))) + sum((mu_I* phiC - 1) * log(y)) +
                sum(((1 - mu_I) * phiC - 1) * log(1 - y)) - log(b - a) + dgamma(phi_I, (phiC^2)/var,  scale = 1/(phiC / var))) -
               (N * log(gamma(phi_I)) -sum(log(gamma(mu_I * phi_I))) -sum(log(gamma((1 - mu_I) * phi_I))) +sum((mu_I* phi_I - 1) * log(y)) +
               sum(((1 - mu_I) * phi_I - 1)* log(1 - y)) - log(b - a) + dgamma(phiC,  (phi_I^2)/var, scale = 1/(phi_I / var))))
  
  #Accept/Reject step for phi
  if(runif(1) < min(1, exp(ratio_phi)))
  {phi_I <- phiC
  j_phi <- j_phi +1}
  
  if (i > BurnIn){
    SaveResults[AuxBurnIn,] <- c(AuxBurnIn, MCMCBetasI, phi_I)
    AuxBurnIn <- AuxBurnIn + 1
  }
}

# Acceptance rate -------------------------------------------------------------
Acc_rate_beta = j_beta/TotIter
Acc_rate_beta
Acc_rate_phi = j_phi/TotIter
Acc_rate_phi

## Plots
## -----
Chainalpha <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Alpha)) +
  labs(title=expression(paste('Chain of ', alpha)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[1], linetype = 'dotted', color='coral') +
  theme_bw()

Chainbeta <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Beta)) +
  labs(title=expression(paste('Chain of ', beta)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[2], linetype = 'dotted', color='coral') +
  theme_bw()

Chainphi <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Phi)) +
  labs(title=expression(paste('Chain of ', phi)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[3], linetype = 'dotted', color='coral') +
  theme_bw()

grid.arrange(Chainalpha, Chainbeta, Chainphi, ncol=2)

# Comparison:
#---------------------------------------------------------------------
# Checking the posterior estimates -----------------
alpha_est <- mean(SaveResults$Alpha)
beta_est <- mean(SaveResults$Beta)
phi_est <- mean(SaveResults$Phi)

means_MCMC = c(alpha_est, beta_est, phi_est)
means_JAGS = c(-1.12, 0.22, 4.97)
sd_JAGS = c(0.05117907, 0.00901529, 0.2067916)

# load needed libraries----------------------------
library(reshape2)

# Data Frame---------------------------------------
Name1 <- c('Alpha', 'Beta', 'Phi')
myData <- data.frame( Name1, means_JAGS, means_MCMC)
myData

# reshape data into long format-------------------
myDATAlong <- melt(myData)

# make the plot-----------------------------------
ggplot(myDATAlong) +
  geom_bar(aes(x = Name1, y = value, fill = variable), 
           stat="identity", position = "dodge", width = 0.7) +
  scale_fill_manual("Beta Regression \nMeans\n", values = c("red","blue"), 
                    labels = c("Jags", "MCMC")) +
  labs(x="\nVars",y="Mean\n") +
  theme_bw(base_size = 14)
