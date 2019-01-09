# ------------------------------------------------------------------ #
# Description: Metropolis-Hastings for Bayesian logistic regression  #
#              with a Multivariate Normal as prior for betas.        #
# Author: Alan Inglis                                                #
# Last modification: 18/12/2018                                      #
# ------------------------------------------------------------------ #

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
k <- 1
j <- 1
T <- 10000
alpha <- 1
b1 <- 0.5
b2 <- -0.5
P <- c(alpha,b1,b2)
AuxBurnIn <- 1

## Simulate data
## -----------------
x0 <- rep(1,T)
x1 <- runif(T, 0, 10)
x2 <- runif(T, 0, 10)

X <- cbind(x0, x1, x2)
logit_p = alpha + b1 * x1 + b2 * x2
p =  exp(logit_p)/(1+exp(logit_p)) # Inverse Logit
y <- rbinom(n = T, size = 1, prob = p)

## Defining prior distributions and the values of the hyperparameters
## ----------------------------
## Beta ~ N(mu, sigma)
## ----------------------------
mu = c(0,0,0)
Sigma <- diag(3)*100
Sigma_inv <- solve(Sigma)

## Data frame that will store MCMC values for betas
## ------------------------------------------------
SaveResults <- as.data.frame(matrix(data = NA, nrow = Niter, ncol = length(P)+1))
colnames(SaveResults) <- c('Iter', 'Alpha', 'Beta1', 'Beta2')

## Initial values for Beta
## ------------------------
MCMCBetasI <- c(0,0,0)
V = diag(3)*0.0005 # Constructs a diagonal matrix

## Metropolis-Hastings
## -----------------------------------
pb <- progress_bar$new(
  format = "  iterations [:bar] :percent done",
  total = TotIter, clear = FALSE, width= 60) # Displays progress bar for iterations

for(i in 1:TotIter){
  pb$tick()
  # Proposal distribution for beta
  MCMCBetasC <- mvrnorm(1, MCMCBetasI, V)
  
  # Metropolis ratio
  ratio <- ((-k * sum(log(1 + exp(X%*%MCMCBetasC)))  + sum(y * X%*%MCMCBetasC) - (1/2) * t(MCMCBetasC - mu)%*%Sigma_inv%*%(MCMCBetasC - mu)) -
              ((-k * sum(log(1 + exp(X%*%MCMCBetasI)))) + sum(y * X%*%MCMCBetasI) - (1/2) * t(MCMCBetasI - mu)%*%Sigma_inv%*%(MCMCBetasI - mu)))
  
  # Accept/reject step
  if(runif(1) < min(1, exp(ratio)))
  {MCMCBetasI <- MCMCBetasC
  j <- j+1}
  
  # Store values
  if (i > BurnIn){
    SaveResults[AuxBurnIn,] <- c(AuxBurnIn, MCMCBetasI)
    AuxBurnIn <- AuxBurnIn + 1
  }
}

## Acceptance rate
acceptance <- j/TotIter
acceptance

## Plots
## -----
Chainalpha <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Alpha)) +
  labs(title=expression(paste('Chain of ', alpha)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[1], linetype = 'dotted', color='coral') +
  theme_bw()

ChainB1 <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Beta1)) +
  labs(title=expression(paste('Chain of ', beta[1])),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[2], linetype = 'dotted', color='coral') +
  theme_bw()

ChainB2 <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Beta2)) +
  labs(title=expression(paste('Chain of ', beta[2])),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=P[3], linetype = 'dotted', color='coral') +
  theme_bw()

grid.arrange(Chainalpha, ChainB1, ChainB2, ncol = 2)

# Comparison:
#---------------------------------------------------------------------
# Checking the posterior estimates ----------------------
alpha_est <- mean(SaveResults$Alpha)
beta1_est <- mean(SaveResults$Beta1)
beta2_est <- mean(SaveResults$Beta2)

means_MCMC = c(alpha_est, beta1_est, beta2_est)
means_JAGS = c(1.05, 0.57, -0.55)
sd_JAGS = c(0.2277933, 0.04452061, 0.04181527)

# load needed libraries---------------------------------
library(reshape2)

# Data Frame--------------------------------------------
Name1 <- c('Alpha', 'Beta1', "Beta2")
myData <- data.frame( Name1, means_JAGS, means_MCMC)

# reshape data into long format------------------------
myDATAlong <- melt(myData)

# make the plot----------------------------------------
ggplot(myDATAlong) +
  geom_bar(aes(x = Name1, y = value, fill = variable), 
           stat="identity", position = "dodge", width = 0.7) +
  scale_fill_manual("Logistic Regression \nMeans\n", values = c("red","blue"), 
                    labels = c("Jags", "MCMC")) +
  labs(x="\nVars",y="Mean\n") +
  theme_bw(base_size = 14)
