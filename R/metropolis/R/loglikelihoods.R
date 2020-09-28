#### log likelihoods ####


.errcheck <- function(par){
  if(any(is.na(par))) warning("At least one paramter value is NA. Check the model and consider stronger priors or modify tuning parmameters via metropolis.control.")
  if(any(is.nan(par))) warning("At least one paramter value is NaN. Check the model and consider stronger priors or modify tuning parmameters via metropolis.control.")
}

logistic_ll <- function(y, X, par){
  #' @title logistic log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of model coefficients 
  #' @return a scalar quantity proportional to a binomial likelihood
  #' with logistic parameterization, given y,X,and par
  .errcheck(par)
  #sum(dbinom(y, 1, expit(X %*% par), log=TRUE))
  #xp = X %*% par
  xp = tcrossprod(X,t(par))
  sum(dbinom(y, 1, expit(xp), log=TRUE))
}

.normal_ll <- function(y, X, sig, beta){
  if(sig<0) return(-Inf)
  #sum(dnorm(y, X %*% beta, sig, log=TRUE))
  #xp = X %*% beta
  xp = tcrossprod(X,t(beta))
  sum(dnorm(y, xp, sig, log=TRUE))
}

normal_ll <- function(y, X, par){
  #' @title Gaussian log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of gaussian scale parameter followed by model coefficients 
  #' @return a scalar quantity proportional to a normal likelihood
  #' with linear parameterization, given y, X, and par
  .errcheck(par)
   beta = par[-1]
   sig = par[1]
  .normal_ll(y,X,sig,beta)
}

