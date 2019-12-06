expit <- function(mu){
  #' @title Inverse logit transform
  #'
  #' @param mu log-odds 
  #' @export
  #' @examples
  #' logodds = rnorm(10)
  #' expit(logodds)
  1/(1+exp(-mu))
}

print.metropolis.samples <- function(x, ...){
  fam = x$family$family
  mod = as.character(x$f)
  fun = paste(mod[2], mod[1], mod[3])
  cat(paste0("\n Metropolis: ", x$iter, " iterations with ", x$burnin, " burn-in", "\n"))
  cat(paste0(" Model: ", fun, "    (", fam, ")\n"))
  cat(paste0(" Guided: ", x$guided,  "\n"))
  cat(paste0(" Adaptive: ", x$adaptive,  "\n"))
  #cat(paste0("\n Initial values: ", x$inits,  "\n"))
  invisible(x)    
}


summary.metropolis.samples <- function(object, keepburn=FALSE, ...){
  #' @title Summarize a probability distribution from a Markov Chain
  #'
  #' @description This function allows you to summarize output from the metropolis function.
  #' @details TBA
  #' @param object an object from the function "metropolis"
  #' @param keepburn keep the burnin iterations in calculations (if adapt=TRUE, keepburn=TRUE 
  #' will yield potentially invalid summaries)
  #' @param ... not used
  #' @export
  #' @examples
  #' runif(1)

  # function to compute summary of probability distribution sampled via
  # metropolis-hastings algorithm
  samples = object$parms[object$parms$burn<ifelse(keepburn, 2, 1),]
  sims = samples[, -grep("burn|iter", names(samples))]
  nms = object$dimnames
  if(object$family$family=="gaussian") nms = c("logsigma", nms)
  names(sims) <- nms
  nsims = dim(sims)[1]
  
  #effective sample size
  ar.beta = apply(sims, 2, ar)
  sv.beta = sapply(ar.beta, function(x) x$var.pred/(1-sum(x$ar))^2)
  var.beta = apply(sims, 2, var)
  ess.beta = pmin(nsims, nsims*var.beta/sv.beta)

  #estimates based on posterior mean, normal approx probability intervals
  (mean.beta  <- apply(sims, 2, mean))
  (sd.beta  <- sqrt(var.beta))
  lci.normal.beta <- mean.beta-1.96*sd.beta
  uci.normal.beta <- mean.beta+1.96*sd.beta

  #estimates based on posterior median, quantile based probability intervals
  median.beta = apply(sims, 2, function(x) quantile(x, c(0.5)))
  lci.quantile.beta <- apply(sims, 2, function(x) quantile(x, c(0.025)))
  uci.quantile.beta <- apply(sims, 2, function(x) quantile(x, c(0.975)))

  #estimates based on posterior mode (via kernal density), highest probability density based intervals
  dens.beta <- apply(sims, 2, density) 
  mode.beta = sapply(dens.beta, function(x) x$x[which.max(x$y)][1])
  hpdfun <- function(x, p=0.95){
   x = sort(x)
   xl = length(x)
   g = round(xl*p)
   idx = which.min(x[1:(xl-g)])
   c(x[idx], x[idx+g])
  }
  hpd.beta = apply(sims, 2, function(x) hpdfun(x, c(0.95)))
  lci.hpd.beta <- hpd.beta[1,]
  uci.hpd.beta <- hpd.beta[2,]

  list(
    nsamples = nsims,
    sd = sd.beta,
    se = sd.beta/sqrt(ess.beta),
    ESS_beta = ess.beta,
     postmean = cbind(mean=mean.beta, normal_lci=lci.normal.beta, normal_uci=uci.normal.beta),
     postmedian = cbind(median = median.beta, pctl_lci = lci.quantile.beta, pctl_uci = uci.quantile.beta),
     postmode = cbind(mode=mode.beta, hpd_lci = lci.hpd.beta, hpd_uci = uci.hpd.beta)
     )
}

plot.metropolis.samples <- function(x, keepburn=FALSE, parms=NULL, ...){
  #' @title Plot the output from the metropolis function
  #'
  #' @description This function allows you to summarize output from the metropolis function.
  #' @details TBA
  #' @param x the outputted object from the "metropolis_glm" function
  #' @param keepburn keep the burnin iterations in calculations (if adapt=TRUE, keepburn=TRUE 
  #' @param parms names of parameters to plot (plots the first by default, if TRUE, plots all)
  #' @param ... other arguments to plot
  #' @import stats graphics
  #' @importFrom grDevices dev.hold
  #' @export
  #' @examples
  #' runif(1)
  samples = x$parms[x$parms$burn<ifelse(keepburn, 2, 1),]
  sims = samples[, -grep("burn|iter", names(samples))]
  nms = x$dimnames
  if(x$family$family=="gaussian") nms = c("logsigma", nms)
  names(sims) <- nms
  nsims = dim(sims)[1]
  if(isTRUE(parms[1])) parms = names(sims)
  if(is.null(parms[1])) parms = 1
  
  plsims = sims[,parms, drop=FALSE]
  oldmar = par()$mar
  if(!keepburn) idx=(x$burn+1):(x$burn+x$iter)
  if(keepburn) idx=1:(x$burn+x$iter)
  par(mfcol=c(2,1), mar=c(3,4,1,1))
  for(k in 1:dim(plsims)[2]){
    #scatter plot
    hist(plsims[,k], main="", breaks = 50)
    #dev.hold()
    #trace plot
    plot(idx, plsims[,k], type="l", ylab=names(plsims)[k], ...)
  }
  par(mfcol=c(1,1), mar=oldmar)
}


logistic_ll <- function(y, X, par){
  #' @title logistic log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of model coefficients 
  sum(dbinom(y, 1, expit(X %*% par), log=TRUE))
}

normal_ll <- function(y, X, par){
  #' @title Gaussian log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of gaussian scale parameter followed by model coefficients 
  beta = par[-1]
  sig = par[1]
  sum(dnorm(y, X %*% beta, sig, log=TRUE))
}

metropolis.control <- function(
  adapt.start = 25,
  adapt.window = 200, 
  adapt.update = 25,
  min.sigma = 0.001,
  prop.sigma.start = 1,
  scale = 2.4 
  ){
  #' @title metropolis.control 
  #'
  #' @param adapt.start start adapting after this many iterations; set to iter+1 to turn off adaptation 
  #' @param adapt.window base acceptance rate on maximum of this many iterations
  #' @param adapt.update frequency of adaptation
  #' @param min.sigma minimum of the proposal distribution standard deviation (if set to zero, 
  #' posterior may get stuck)
  #' @param prop.sigma.start starting value, or fixed value for proposal distribution s
  #' standard deviation
  #' @param scale scale value for adaptation (how much should the posterior
  #'  variance estimate be scaled by?). Scale/sqrt(p) is used in metropolis_glm function, and 
  #'  Gelman et al. (Bayesian Data Analysis v3, p 296) recommend a scale of 2.4
  #' @export
  list(
    adapt.start = adapt.start, 
    adapt.window = adapt.window,
    adapt.update = adapt.update,
    min.sigma = min.sigma,
    prop.sigma.start = prop.sigma.start,
    scale = scale 
  )
}


calcpost <- function(y,X,par,family, pm, pv){
  if(family$family == "binomial") llp = logistic_ll(y, X, par)
  if(family$family == "gaussian"){
    modpar = par
    modpar[1] = exp(par[1])
    llp = normal_ll(y, X, modpar)
  }
  # add in priors
  if(!is.null(pm)){
    llp = llp + sum(dnorm(par, mean = pm ,sd = sqrt(pv), log=TRUE))
  }
  # add in Jacobian of transform for linear models
  if(family$family == "gaussian") llp = llp + exp(par[1])
  llp
}


metropolis_glm <- function(
                       f, 
                       data,
                       family=binomial(),
                       iter=100, 
                       burnin=round(iter/2), 
                       pm=NULL, 
                       pv=NULL, 
                       chain=1, 
                       prop.sigma.start=.1, 
                       inits=NULL, 
                       adaptive=TRUE, 
                       guided=FALSE, 
                       block=TRUE,
                       saveproposal=FALSE,
                       control = metropolis.control()
                       ) {
  #' @title Use the Metropolis Hastings algorithm to estimate Bayesian glm parameters
  #'
  #' @description This function carries out the metropolis algorithm.
  #' @details TBA
  #' @param f an R style formula (e.g. y ~ x1 + x2)
  #' @param data an R data frame containing the variables in f
  #' @param family R glm style family that determines model form: normal() or binomial()
  #' @param iter number of iterations after burnin to keep
  #' @param burnin number of iterations at the beginning to throw out (also used for adaptive phase)
  #' @param pm vector of prior means for normal prior on log(scale) (if applicable) and 
  #' regression coefficients (set to NULL to use uniform priors) 
  #' @param pv vector of prior variances for normal prior on log(scale) (if applicable) and 
  #' regression coefficients (set to NULL to use uniform priors) 
  #' @param chain chain id [plan to deprecate] 
  #' @param prop.sigma.start proposal distribution standard deviation (starting point if adapt=TRUE) 
  #' @param inits NULL, a vector with length equal to number of parameters (intercept + x + scale 
  #' [gaussian() family only model only]), or "glm" to set priors based on an MLE fit
  #' @param adaptive logical, should proposal distribution be adaptive? (TRUE usually gives better answers)
  #' @param guided logical, should Gustafson's "guided" algorithm be used (TRUE usually gives better answers)
  #' @param block logical or a vector that sums to total number of parameters (e.g. if there are 4 
  #' random variables in the model, including intercept, then block=c(1,3) will update the 
  #' intercept separately from the other three parameters.) If TRUE, then updates each parameter 
  #' 1 by 1. Using "guide=TRUE" with blocking=<vector> is not advised
  #' @param saveproposal (logical, default=FALSE) save the rejected proposals (block=TRUE only)?
  #' @param control parameters that control fitting algorithm. See metropolis.control()
  #' @import stats
  #' @export
  #' @examples
  #' dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
  #' res = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=10000, burnin=3000, 
  #' adapt=TRUE, guided=TRUE, block=FALSE)
  #' apply(res$parms, 2, mean)
  #' glm(y ~ x1 + x2, family=binomial(), data=dat)
  #' dat = data.frame(y = rnorm(100, 1, 0.5), x1=runif(100), x2 = runif(100), x3 = rpois(100, .2))
  #' res = metropolis_glm(y ~ x1 + x2 + factor(x3), data=dat, family=gaussian(), inits="glm", 
  #' iter=10000, burnin=3000, adapt=TRUE, guide=TRUE, block=FALSE)
  #' apply(res$parms, 2, mean)
  #' glm(y ~ x1 + x2+ factor(x3), family=gaussian(), data=dat)
  # error catching 
  if (is.character(family)) 
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family)) 
    family <- family()
  if (is.null(family$family)) {
    print(family)
    stop("'family' not recognized")
  }
  if (!(family$family %in% c("gaussian", "binomial"))) {
    print(family)
    stop("only 'gaussian' and 'binomial' families are currently implemented")
  }
  if(length(block)>1 & guided){
    warning("Block sampling with 'guided=TRUE' is not advised, unless block=TRUE")
  }
  
  # collect some info on data
    X = model.matrix(f, data = data)
    mterm = terms(f)
    outcol = as.character(attr(mterm, "variables")[[2]])
    y = data[,outcol]
    
    p = dim(X)[2] # number of parameters
    if(family$family == "gaussian"){
      p = p+1 # scale parameter
    }

    # create empty matrixes/vectors for posterior estimates
    accept <- parms <- matrix(nrow=iter+burnin, ncol=p)
    proposals = NULL
  # generate initial values
    if(is.null(inits)){
      inits = c(beta=runif(p)*4-2)
    } 
    if(length(inits)==1 && inits=="glm"){
      mlefit = glm(f, family=family, data)
      inits = as.numeric(mlefit$coefficients)
      if(family$family == "gaussian") inits = c(logscale=log(sd(mlefit$residuals)), beta=inits)
    }
    parms[1,] <- inits
    if(saveproposal & length(block)==1 && block) proposals = parms
    if(length(control$prop.sigma.start)==1){
      cov = rep(control$prop.sigma.start, p)
    }else{
      cov = control$prop.sigma.start
    }

    accept[1,] = rep(1, p)
    if(!adaptive) control$adapt.start = iter+1+burnin
    # guiding direction
    if(length(block)==1 && block) dir = 1
    if(length(block)>1) dir = rep(1, length(block))
    if(length(block)==1 && !block) dir = rep(1, p)
    # run metropolis algorithmm
  for(i in 2:(iter+burnin)){
   #if adaptive, then adapt while in the adaptation phase
    if((i>=control$adapt.start  & !(i%%control$adapt.update) & i<=burnin)){
      cov <- control$scale/sqrt(p)*apply(parms, 2, sd, na.rm=TRUE)
    }
    b.prv = parms[i-1,]
    b.cand = b.prv
    if(length(block)==1 && block){
      # block update all parameters
      # non-normalized log-probability at previous beta
      llp = calcpost(y,X,b.prv,family, pm, pv)        
      # draw candidate value from proposal distribution
      z = rnorm(p,0, cov) 
      if(guided) z = abs(z) * dir
      b.cand = b.cand + z
      if(saveproposal) proposals[i,] = b.cand
      
      # non-normalized log-probability at candidate beta
      lp = calcpost(y,X,b.cand,family, pm, pv)        
      # accept/reject candidate value
      accept.prob = exp(lp-llp) # acceptance probability
      a = (accept.prob > runif(1))
      if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
      if(!a){
        b.cand = b.cand - z
        dir = dir
      }
      accept[i,] = a
    } else{ # partial blocks, or update 1 by 1
      # loop over parameters
      if(length(block)>1) blocks=block else blocks = rep(1,p)
      st = cumsum(c(1, blocks))
      end = cumsum(c(blocks, 1))
      blocklist = list()
      for(idx in 1:length(st[-1])){
        blocklist[[idx]] = st[idx]:end[idx]
      }
      for(q in blocklist){ # loop over blocks
        # non-normalized log-probability at previous beta
        llp = calcpost(y,X,b.prv,family, pm, pv)        
        # draw candidate value from proposal distribution
        z = rep(0, p)
        z[q] = rnorm(length(q),0, cov[q]) 
        if(guided) z = abs(z) * dir
        b.cand = b.cand + z
        # non-normalized log-probability at candidate beta
        lp = calcpost(y,X,b.cand,family, pm, pv)        
        # accept/reject candidate value
        accept.prob = exp(lp-llp) # acceptance probability
        a = (accept.prob > runif(1))
        if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
        if(!a){
          b.cand = b.cand - z
          dir[q] = -dir[q]
        }
        b.prv[q] = b.cand[q]
        accept[i,q] = a
      }# end loop over blocks
    } # end partial blocks, or update 1 by 1
    parms[i,] = b.cand
  } #loop over i
  bt = data.frame(parms)
  if(family$family=="binomial") nm = paste0('b_', 0:(p-1), '')
  if(family$family=="gaussian") nm = c("logsigma", paste0('b_', 0:(p-2), ''))
  names(bt) <- nm
  bt$iter = 1:dim(bt)[1]
  bt$burn = 1.0*(bt$iter<=burnin)
  bt <- bt[,c('iter', 'burn', grep('logsigma|b_', names(bt), value = TRUE, fixed=FALSE))]
  # output an R list with posterior samples, acceptance rate, and covariance of proposal distribution
  res = list(parms = bt, 
             accept=accept, 
             cov=cov, 
             iter=iter, 
             burnin=burnin, 
             f=f, 
             family=family, 
             dimnames = colnames(X), 
             guided=guided,
             adaptive = adaptive,
             priors = ifelse(is.null(pm), "uniform", "normal"),
             inits = inits,
             proposals = proposals
             )
  class(res) <- "metropolis.samples"
  res
}


# coda object
as.mcmc.metropolis.samples <- function(object, ...){
  #' @title Convert glm_metropolis output to `mcmc` object from package coda
  #'
  #' @description Allows use of useful functions from `coda` package
  #' @details TBA
  #' @param object an object from the function "metropolis"
  #' @param ... not used
  #' @importFrom coda mcmc
  #' @importFrom coda as.mcmc
  #' @export
  #' @examples
  #' runif(1)
  samples = object$parms[object$parms$burn==0,grep('logsigma|b_', names(object$parms), value = TRUE, fixed=FALSE)]
  mcmc(data= samples, 
       start =  object$burnin, 
       end = object$iter,
       thin = 1)
  
}
