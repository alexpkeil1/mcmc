library(ggplot2)
library(tibble)
library(coda)

# from gustafson, finding the mean of a standard normal

mh.standard <- function(iter,prop.sigma=1,seed=NULL){
  set.seed(seed)
  x <- accept <- prop.cov <- numeric(iter) # x= thing you are sampling
  x[1] <- -6
  for(i in 2:iter){
    z <- rnorm(1, 0, prop.sigma)
    xm1 = x[i-1]
    y <- xm1+z
    lp <- dnorm(y)
    plp <- dnorm(xm1)
    a <- rbinom(1, 1, min(1, lp/plp))
    x[i] <- ifelse(a, y, xm1)
    accept[i] = a
    prop.cov[i] = prop.sigma
  }
  tibble(iter=1:iter, x=x, accept=accept, prop.cov=prop.cov)
}

mh.adaptive <- function(iter, prop.sigma.start=1,seed=NULL){
  # adaptive MH algorithm from Bayesian Data Analysis Chap. 9. (Gelman)
  set.seed(seed)
  # adapdation parameters
  d = 1 # number of parameters in posterior
  cov = prop.sigma.start # starting value for covariance parameter of proposal distribution
  ai = 30 # adapt every ai iterations
  ari = 100 # base acceptance rate on this many iterations
  adapt.phase = Inf # stop adapting after this many iterations
  #####
  x <- accept <- prop.cov <- numeric(iter) # x= thing you are sampling
  x[1] <- -6
  prop.cov[1] = cov
  scale = 1
  for(i in 2:iter){
    # adapt
    if((i%%ai)==0 & i>1 & i<=adapt.phase){
      # ideal acceptance rate is 0.44 (if too high, then steps are too small, so need to scale)
      ap = max(0.01, mean(accept[max(1,i-ari):i], na.rm=TRUE))
      #ap = max(0.01, mean(accept[1:i], na.rm=TRUE))
      scale = scale*(ap/.44)
      cov = sd(x[1:i])*scale
    }
    ######
    z <- rnorm(1, 0, cov)
    xm1 = x[i-1]
    y <- xm1+z
    lp <- dnorm(y)
    plp <- dnorm(xm1)
    a <- rbinom(1, 1, min(1, lp/plp))
    x[i] <- ifelse(a, y, xm1)
    accept[i] = a
    prop.cov[i] = cov
  }
  tibble(iter=1:iter, x=x, accept=accept, prop.cov=prop.cov)
}

mh.guided <- function(iter,prop.sigma=1,seed=NULL){
  # guided walk MH from Gustafson 1998
  set.seed(seed)
  x <- accept <- prop.cov <- numeric(iter) # x= thing you are sampling
  x[1] <- -6
  pn = 1
  for(i in 2:iter){
    z <- abs(rnorm(1, 0, prop.sigma))
    xm1 = x[i-1]
    y <- xm1+pn*z
    lp <- dnorm(y)
    plp <- dnorm(xm1)
    a <- rbinom(1, 1, min(1, lp/plp))
    x[i] <- ifelse(a, y, xm1)
    if(!a) pn = -pn
    accept[i] = a
    prop.cov[i] = prop.sigma
  }
  tibble(iter=1:iter, x=x, accept=accept, prop.cov=prop.cov)
}


mh.adaptive.guided <- function(iter, prop.sigma.start=1,seed=NULL){
  # adaptive MH algorithm from Bayesian Data Analysis Chap. 9. (Gelman)
  set.seed(seed)
  # adapdation parameters
  d = 1 # number of parameters in posterior
  cov = prop.sigma.start # starting value for covariance parameter of proposal distribution
  ai = 1 # adapt every ai iterations
  ari = 100 # base acceptance rate on this many iterations
  adapt.phase = Inf # stop adapting after this many iterations
  #####
  x <- accept <- prop.cov <- numeric(iter) # x= thing you are sampling
  x[1] <- -6
  prop.cov[1] = cov
  scale = 1
  pn = 1
  for(i in 2:iter){
    # adapt
    if((i%%ai)==0 & i>1 & i<=adapt.phase){
      # ideal acceptance rate is 0.44 (if too high, then steps are too small, so need to scale)
      ap = max(0.01, mean(accept[max(1,i-ari):i], na.rm=TRUE))
      #ap = max(0.01, mean(accept[1:i], na.rm=TRUE))
      scale = scale*(ap/.44)
      cov = sd(x[1:i])*scale
    }
    ######
    z <- abs(rnorm(1, 0, cov))
    xm1 = x[i-1]
    y <- xm1+pn*z
    lp <- dnorm(y)
    plp <- dnorm(xm1)
    a <- rbinom(1, 1, min(1, lp/plp))
    x[i] <- ifelse(a, y, xm1)
    accept[i] = a
    if(!a) pn = -pn
    prop.cov[i] = cov
  }
  tibble(iter=1:iter, x=x, accept=accept, prop.cov=prop.cov)
}


mc.iter = 50000
prop.sigma = 0.1
burnin = 1000
if(burnin>0){
  keep = (burnin-1):mc.iter
  }else keep = 1:mc.iter

SEED= 1232
samples = mh.standard(mc.iter, prop.sigma=prop.sigma, seed=SEED)
samples.adapt = mh.adaptive(mc.iter, prop.sigma.start=prop.sigma, seed=SEED)
samples.guided = mh.guided(mc.iter, prop.sigma=prop.sigma, seed=SEED)
samples.adaptive.guided = mh.adaptive.guided(mc.iter, prop.sigma=prop.sigma, seed=SEED)

# autocorrelation
autocorr(as.mcmc(samples$x))
autocorr(as.mcmc(samples.adapt$x))
autocorr(as.mcmc(samples.guided$x))


# convergence dx
# Z-scores for a test of equality of means between the first and last parts of the chain
geweke.diag(as.mcmc(samples$x))
geweke.diag(as.mcmc(samples.adapt$x))
geweke.diag(as.mcmc(samples.guided$x))

effectiveSize(as.mcmc(samples$x))
effectiveSize(as.mcmc(samples.adapt$x))
effectiveSize(as.mcmc(samples.guided$x))
effectiveSize(as.mcmc(samples.adaptive.guided$x))



# acceptance probabilities
mean(samples$accept[keep])
mean(samples.adapt$accept[keep])
mean(samples.guided$accept[keep])
mean(samples.adaptive.guided$accept[keep])

# means
fn <- function(x) c(mean(x), sd(x))
fn(samples$x[keep])
fn(samples.adapt$x[keep])
fn(samples.guided$x[keep])
fn(samples.adaptive.guided$x[keep])

#  
mean(samples.adapt$prop.cov[keep])
mean(samples.adaptive.guided$prop.cov[keep])
hist(samples.adaptive.guided$prop.cov[keep])


# data for plotting
samples$aprob = cumsum(samples$accept)/samples$iter
samples$adapt.aprob = cumsum(samples.adapt$accept)/samples$iter
samples$guided.aprob = cumsum(samples.guided$accept)/samples$iter



samples$x.adapt = samples.adapt$x
samples$x.guided = samples.guided$x
samples$x.adaptive.guided = samples.adaptive.guided$x

# proposal distribution std.deviation
ggplot(data=samples.adapt) + 
  geom_line(aes(x=iter,y=prop.cov)) +
  scale_y_continuous(name="Proposal distribution std. dev.") +
  theme_classic()


# acceptance probabilities over time
ggplot(data=samples) + 
  geom_line(aes(x=iter,y=adapt.aprob, colour='Adaptive MH')) + 
  geom_line(aes(x=iter,y=aprob, colour='RW MH')) +
  geom_line(aes(x=iter,y=guided.aprob, colour='Guided MH')) +
  scale_y_continuous(name="Cum. acceptance probability") +
  scale_color_discrete(name="") +
  theme_classic() + theme(legend.position = c(1,0), legend.justification = c(1,0))

# parameter traceplots
ggplot(data=samples) + 
  geom_line(aes(x=iter,y=x.adaptive.guided, colour='Adaptive, guided MH')) + 
  geom_line(aes(x=iter,y=x.adapt, colour='Adaptive MH')) + 
  geom_line(aes(x=iter,y=x.guided, colour='Guided MH')) + 
  geom_line(aes(x=iter,y=x, colour='RW MH')) +
  scale_color_discrete(name="") +
  scale_y_continuous(name="X") +
  theme_classic() + theme(legend.position = c(1,0), legend.justification = c(1,0))



