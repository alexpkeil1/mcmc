

library(ggplot2)
library(grid)
library(ggExtra)

#leading to 3 exposed cases and 5 exposed controls
# 36 leukemia cases and 198 controls

# data
y = c(rep(1, 36), rep(0, 198)) # leukemia cases
x = c(rep(1, 3), rep(0, 33), rep(1, 5), rep(0, 193)) # exposure
X = cbind(rep(1,length(x)), x) # design matrix


expit <- function(mu) 1/(1+exp(-mu))
metrop.hastings <- function(iter=100, burnin=round(iter/2), y, X=X, pm=c(0,0), pv=c(100,0.5), chain=1, prop.sigma.start=1, seed=NULL, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=TRUE, incl.likelihood=TRUE) {
  #metropolis hastings algorithm to fit a logistic model with one intercept and single exposure
    set.seed(seed)
  # some modifiable settings
    adapt.start = 25 # start adapting after this many iterations; set to iter+1 to turn off adaptation
    adapt.window = 100 # base acceptance rate on maximum of this many iterations
    adapt.phase = burnin # stop adapting after this many iterations (should be <= burnin)
    adapt.update = 10 # frequency of adaptation
  # collect some info on data
    p = dim(X)[2] # number of parameters
  # make copies of design matrix for risk difference calculation
    X1 = X0 = X
    X1[,2] = 1
    X0[,2] = 0
  # create empty matrixes/vectors for posterior estimates
    accept <- beta <- matrix(nrow=iter+burnin, ncol=p)
    rd <- numeric(iter+burnin)
  # generate initial values
    if(is.null(inits)) inits = runif(p)*4-2
    beta[1,] = inits
    cov =  rep(prop.sigma.start, p)# starting value for covariance parameter of proposal distribution
    accept[1,] = rep(1,p)
    scale = 1 # scale for adaptive estimation
    if(!adaptive) adapt.start = iter+1+burnin
    pn = rep(1, p) # direction of guiding
  # run metropolis algorithm
  for(i in 2:(iter+burnin)){
   #if adaptive, then adapt while in the adaptation phase
    if((i>=adapt.start  & !(i%%adapt.update) & i<=adapt.phase)){
      ap = pmax(0.01, apply(accept[max(1,i-adapt.window):i,], 2, mean, na.rm=TRUE))
      scale = scale*(ap/0.44)
      cov <- scale*apply(beta, 2, sd, na.rm=TRUE)
    }
    b.prv = beta[i-1,]
    b.cand = b.prv
    # loop over parameters
    for(q in 1:p){
      # non-normalized log-probability at previous beta (prior, likelihood, or posterior)
        llp = ifelse(incl.likelihood,sum(dbinom(y,1,expit(X %*% b.cand), log = TRUE)), 0) + # log likelihood
              ifelse(incl.priors, sum(dnorm(b.cand, mean = pm, sd = sqrt(pv), log = TRUE)), 0) # 
      # draw candidate value from proposal distribution
        z = rep(0,p)
        z[q] = rnorm(1,0, cov[q]) # set to zero for variable that is not being updated
        if(guided) z = abs(z) * pn
        b.cand = b.cand + z
      # non-normalized log-probability at candidate value (prior, likelihood, or posterior)
        lp = ifelse(incl.likelihood,sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)), 0) + # log likelihood
              ifelse(incl.priors, sum(dnorm(b.cand, mean = pm, sd = sqrt(pv), log = TRUE)), 0) # 
      # accept/reject candidate value
        accept.prob = min(1, exp(lp-llp)) # acceptance probability
        a = rbinom(1, 1, accept.prob)
        if(!a){
          b.cand = b.cand - z
          pn[q] = -pn[q]
        }
        accept[i,q] = a
    }# loop over p
    beta[i,] = b.cand
    poprisk = 4.8/100000
    popodds = poprisk/(1-poprisk)
    studyodds = mean(y)/(1-mean(y))
    rd[i] = mean(expit(log(popodds/studyodds) + X1%*%b.cand) - expit(log(popodds/studyodds) + X0%*%b.cand))
  } #loop over i
  bt = data.frame(beta)
  names(bt) <- paste0('b[', 1:p, ']')
  bt$chain = rep(chain, dim(bt)[1])
  bt$iter = 1:dim(bt)[1]
  bt$burn = 1.0*(bt$iter<=burnin)
  bt$rd = rd
  bt <- bt[,c('chain', 'iter', 'burn', grep('b[', names(bt), value = TRUE, fixed=TRUE), 'rd')]
  # output an R list with posterior samples, acceptance rate, and covariance of proposal distribution
  list(beta = bt, accept=accept, cov=cov)
}

probdist.summary <- function(posterior.samples){
  # function to compute summary of probability distribution sampled via
  # metropolis-hastings algorithm
  b2samples = posterior.samples$beta[posterior.samples$beta$burn==0,'b[2]']
  rdsamples = posterior.samples$beta[posterior.samples$beta$burn==0,'rd']
  
  #effective sample size
  ar.beta = ar(b2samples)
  ar.rd = ar(rdsamples)
  sv.beta = ar.beta$var.pred/(1 - sum(ar.beta$ar))^2
  sv.rd = ar.rd$var.pred/(1 - sum(ar.rd$ar))^2
  beta.ess = min(length(b2samples), length(b2samples)*var(b2samples)/sv.beta)
  rd.ess = min(length(rdsamples), length(rdsamples)*var(rdsamples)/sv.rd)

  #estimates based on posterior mean, normal approx probability intervals
  (beta.mean  <- mean(b2samples))
  (rd.mean  <- mean(rdsamples))
  or.mean <- exp(beta.mean)
  beta.se  <- sd(b2samples)
  rd.se  <- sd(rdsamples) # not recommended, but done anyway
  beta.lci.normal <- beta.mean-1.96*beta.se
  beta.uci.normal <- beta.mean+1.96*beta.se
  rd.lci.normal <- rd.mean-1.96*rd.se
  rd.uci.normal <- rd.mean+1.96*rd.se
  or.lci.normal <- exp(beta.mean-1.96*beta.se)
  or.uci.normal <- exp(beta.mean+1.96*beta.se)

  #estimates based on posterior median, quantile based probability intervals
  beta.median = quantile(b2samples, c(0.5))
  rd.median = quantile(rdsamples, c(0.5))
  or.median <- exp(beta.median)
  beta.lci.quantile <- quantile(b2samples, c(0.025))
  beta.uci.quantile <- quantile(b2samples, c(0.975))
  rd.lci.quantile <- quantile(rdsamples, c(0.025))
  rd.uci.quantile <- quantile(rdsamples, c(0.975))
  or.lci.quantile <- exp(beta.lci.quantile)
  or.uci.quantile <- exp(beta.uci.quantile)

  #estimates based on posterior mode (via kernal density), highest probability density based intervals
  beta.dens <- density(b2samples)
  beta.mode <- beta.dens$x[which.max(beta.dens$y)][1]
  rd.dens <- density(rdsamples)
  rd.mode <- rd.dens$x[which.max(rd.dens$y)][1] # could also evaluate at mode of beta
  or.mode <- exp(beta.mode)
  hpdfun <- function(x, p=0.95){
   x = sort(x)
   xl = length(x)
   g = round(xl*p)
   idx = which.min(x[1:(xl-g)])
   c(x[idx], x[idx+g])
  }
  hpd.beta = hpdfun(b2samples, c(0.95))
  hpd.rd = hpdfun(rdsamples, c(0.95))
  beta.lci.hpd <- hpd.beta[1]
  beta.uci.hpd <- hpd.beta[2]
  rd.lci.hpd <- hpd.rd[1]
  rd.uci.hpd <- hpd.rd[2]
  or.lci.hpd <- exp(hpd.beta[1])
  or.uci.hpd <- exp(hpd.beta[2])
  
  list(
    nsamples = length(b2samples),
    sd_beta = sd(b2samples),
    se_beta = sd(b2samples)/sqrt(beta.ess),
    ESS_beta = beta.ess,
    ESS_rd = rd.ess,
    summary = rbind(
     beta.mean = c(est=beta.mean, lci=beta.lci.normal, uci=beta.uci.normal),
     beta.median = c(beta.median, beta.lci.quantile, beta.uci.quantile),
     beta.mode = c(beta.mode, beta.lci.hpd, beta.uci.hpd),
     rd.mean = 1000*c(est=rd.mean, lci=rd.lci.normal, uci=rd.uci.normal),
     rd.median = 1000*c(rd.median, rd.lci.quantile, rd.uci.quantile),
     rd.mode = 1000*c(rd.mode, rd.lci.hpd, rd.uci.hpd),
     or.mean = c(or.mean, or.lci.normal, or.uci.normal),
     or.median = c(or.median, or.lci.quantile, or.uci.quantile),
     or.mode = c(or.mode, or.lci.hpd, or.uci.hpd)))
}

# total samples after burnin
M = 100000
# burnin/adaptation length
B = 1000

# getting non-Bayesian estimates for four different algorithms
samples.rw = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=FALSE, incl.likelihood=TRUE)
samples.guided = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=TRUE, incl.priors=FALSE, incl.likelihood=TRUE)
samples.adaptive = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=TRUE, guided=FALSE, incl.priors=FALSE, incl.likelihood=TRUE)
samples.guided.adaptive = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=TRUE, guided=TRUE, incl.priors=FALSE, incl.likelihood=TRUE)

probdist.summary(samples.rw)
probdist.summary(samples.guided)
probdist.summary(samples.adaptive)
probdist.summary(samples.guided.adaptive)

# increasing number of draws from random walk to demonstrate convergence to same answer
samples.rw2 = metrop.hastings(iter=M*40, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=FALSE, incl.likelihood=TRUE)
probdist.summary(samples.rw2)


# getting Bayesian estimates for four different algorithms
samples.rw.bayes = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=TRUE, incl.likelihood=TRUE)
samples.guided.bayes = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=TRUE, incl.priors=TRUE, incl.likelihood=TRUE)
samples.adaptive.bayes = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=TRUE, guided=FALSE, incl.priors=TRUE, incl.likelihood=TRUE)
samples.guided.adaptive.bayes = metrop.hastings(iter=M, burnin=B, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=TRUE, guided=TRUE, incl.priors=TRUE, incl.likelihood=TRUE)

probdist.summary(samples.rw.bayes)
probdist.summary(samples.guided.bayes)
probdist.summary(samples.adaptive.bayes)
probdist.summary(samples.guided.adaptive.bayes)

b2samples = (rnorm(100))

Rprof("/Users/akeil/temp/prof.txt")
 system.time(metrop.hastings(iter=100000, burnin=1000, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=TRUE, incl.likelihood=TRUE))
Rprof(NULL)
