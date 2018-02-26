library(ggplot2)
library(rjags)
library(rstan)
library(grid)
library(ggExtra)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
#leading to 3 exposed cases and 5 exposed controls
# 36 leukemia cases and 198 controls

y = c(rep(1, 36), rep(0, 198))
x = c(rep(1, 3), rep(0, 33), rep(1, 5), rep(0, 193))

table(x,y)

# maximum likelihood
# odds ratio
summary(glm(y~x, family=binomial()))$coefficients
exp(glm(y~x, family=binomial())$coefficients)
exp(confint(glm(y~x, family=binomial())))
# exact odds ratio (OR derived from Breslow and Day 1980, p 165)
fisher.test(x,y) # uses 'central' method for CI

# risk difference
baserisk = 4.8/100000
baseodds = baserisk/(1-baserisk)
caseodds = mean(y)/(1-mean(y))
os = log(baseodds/caseodds) 
mdat = data.frame(y=y,x=x, os=os)
mdat1 = data.frame(y=y,x=rep(1, length(y)), os=os)
mdat0 = data.frame(y=y,x=rep(0, length(y)), os=os)

(mod <- glm(y~x + offset(-os), family=binomial(), data=mdat))
rd = 1000*(mean(expit(cbind(rep(1, length(x)), rep(1, length(x)))%*%mod$coefficients)) - 
  mean(expit(cbind(rep(1, length(x)), rep(0, length(x)))%*%mod$coefficients)))


#bootstrap ci
boots = numeric(1000)
set.seed(3227)
for(i in 1:length(boots)){
  idx = sample(1:(36+198), size=36+198, replace=TRUE)
  mod = glm(y ~x+ offset(-os),family=binomial(), data=mdat[idx,])
  caseodds = mean(y[idx])/(1-mean(y[idx]))
  os = log(baseodds/caseodds) 
  boots[i] = 1000*(mean(expit(cbind(rep(1, length(x)), rep(1, length(x)))%*%mod$coefficients)) - 
    mean(expit(cbind(rep(1, length(x)), rep(0, length(x)))%*%mod$coefficients)))
}
sum(boots>5) # failures
se.rd = sd(boots)
rd; rd-1.96*se.rd; rd+1.96*se.rd;



expit <- function(mu) 1/(1+exp(-mu))
mh.adaptive.guided.logistic <- function(iter, y, X=X, pm=c(0,0), pv=c(100,0.5), chain=1, prop.sigma.start=1, seed=NULL, inits=NULL, adaptive=FALSE, guided=FALSE, incl.priors=TRUE, incl.likelihood=TRUE) {
  set.seed(seed)
  #adaptive metropolis hastings
  X1 = X0 = X
  X1[,2] = 1
  X0[,2] = 0
  p = dim(X)[2]
  accept <- beta <- matrix(nrow=iter, ncol=p)
  rd <- numeric(iter)
  if(is.null(inits)) inits = runif(p)*4-2
  beta[1,] = inits
  cov =  rep(prop.sigma.start, p)# starting value for covariance parameter of proposal distribution
  ai = 30 # adapt every ai iterations; set to iter+1 to turn off adaptation
  if(!adaptive) ai = iter+1
  ari = 100 # base acceptance rate on this many iterations
  adapt.phase = 1000 # stop adapting after this many iterations
  accept[1,] = rep(1,p)
  scale = 1
  j = 2
  pn = rep(1, p) # direction of guiding
  for(i in 2:iter){
   #adaptive
    if((i%%ai)==0 & i>1 & i<=adapt.phase){
      ap = pmax(0.01, apply(accept[max(1,i-ari):i,], 2, mean, na.rm=TRUE))
      scale = scale*(ap/0.44)
      cov <- scale*apply(beta, 2, sd, na.rm=TRUE)
    }
    b.prv = beta[i-1,]
    b.cand = b.prv
    for(q in 1:p){
     # update one at a time
     z = rep(0,p)
     z[q] = rnorm(1,0, cov[q])
     if(guided) z = abs(z) * pn
     # non-normalized log-probability at previous beta
     #dbinom(y,1,expit(X%*%b.cand), log = TRUE)
     llp = ifelse(incl.likelihood,sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)), 0) + # log likelihood
            ifelse(incl.priors, (dnorm(b.cand[1], mean = pm[1], sd = sqrt(pv[1]), log = TRUE) + # 
                     dnorm(b.cand[2], mean = pm[2], sd = sqrt(pv[2]), log = TRUE)), 0) # 
     #include draw from proposal dist'n
     b.cand = b.cand + z
     # non-normalized log-probability at new beta
     lp = ifelse(incl.likelihood,sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)), 0) + # log likelihood
            ifelse(incl.priors, (dnorm(b.cand[1], mean = pm[1], sd = sqrt(pv[1]), log = TRUE) + # 
                     dnorm(b.cand[2], mean = pm[2], sd = sqrt(pv[2]), log = TRUE)), 0) # 
     l.rat = exp(lp-llp)
     a = rbinom(1, 1, min(1, l.rat))
     if(!a){
       b.cand = b.cand - z
       pn[q] = -pn[q]
     }
     accept[i,q] = a
    }# loop over p
    beta[i,] = b.cand
    baserisk = 4.8/100000
    baseodds = baserisk/(1-baserisk)
    caseodds = mean(y)/(1-mean(y))
    rd[i] = mean(expit(log(baseodds/caseodds) + X1%*%b.cand) - expit(log(baserisk/caseodds) + X0%*%b.cand))
  } #loop over i
  bt = data.frame(beta)
  names(bt) <- c('b[1]', 'b[2]')
  bt$iter = 1:dim(bt)[1]
  bt$chain = rep(chain, dim(bt)[1])
  bt$rd = rd
  bt$rd1000 = rd*10000
  list(beta = bt, accept=accept, cov=cov)
}

X = cbind(rep(1,length(x)), x) 

# trace plots for different algorithms with same initial values, seed
iter=10000
samples = mh.adaptive.guided.logistic(iter=iter, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=32217, inits=c(2, -3))
samples.adapt = mh.adaptive.guided.logistic(iter=iter, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=32217, inits=c(2, -3), adaptive=TRUE)
samples.guide = mh.adaptive.guided.logistic(iter=iter, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=32217, inits=c(2, -3), guided=TRUE)
samples.guide.adapt = mh.adaptive.guided.logistic(iter=iter, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE)



# fig 1 (prior, no data; likelihood, no prior;  both)
set.seed(32217)
rngburn = sample(1000:10000, 250)
priors = mh.adaptive.guided.logistic(iter=max(rngburn), y=y, X=X, chain=1, prop.sigma.start=.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE, incl.priors=TRUE, incl.likelihood=FALSE)
priors2 = mh.adaptive.guided.logistic(iter=max(rngburn), y=y, X=X, pm=c(0,1.5), pv=c(2,0.25), chain=1, prop.sigma.start=.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE, incl.priors=TRUE, incl.likelihood=FALSE)
likelihood = mh.adaptive.guided.logistic(iter=max(rngburn), y=y, X=X, chain=1, prop.sigma.start=.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE, incl.priors=FALSE, incl.likelihood=TRUE)
posterior = mh.adaptive.guided.logistic(iter=max(rngburn), y=y, X=X, chain=1, prop.sigma.start=.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE, incl.priors=TRUE, incl.likelihood=TRUE)
posterior2 = mh.adaptive.guided.logistic(iter=max(rngburn), y=y, X=X, pm=c(0,1.5), pv=c(2,0.25), chain=1, prop.sigma.start=.1, seed=32217, inits=c(2, -3), adaptive=TRUE, guided=TRUE, incl.priors=TRUE, incl.likelihood=TRUE)

panel11 <- ggplot() + 
  geom_point(aes(x=priors$beta[rngburn,'b[2]'],y=priors$beta[rngburn,'b[1]']), alpha=0.15) +
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
panel.blank <- ggplot() + geom_point(aes(x=-100,y=-100))+
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
panel22 <- ggplot() + 
  geom_point(aes(x=likelihood$beta[rngburn,'b[2]'],y=likelihood$beta[rngburn,'b[1]']), alpha=0.15) +
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
panel33 <- ggplot() + 
  geom_point(aes(x=posterior$beta[rngburn,'b[2]'],y=posterior$beta[rngburn,'b[1]']), alpha=0.15) +
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
panel41 <- ggplot() + 
  geom_point(aes(x=priors2$beta[rngburn,'b[2]'],y=priors2$beta[rngburn,'b[1]']), alpha=0.15) +
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
panel43 <- ggplot() + 
  geom_point(aes(x=posterior2$beta[rngburn,'b[2]'],y=posterior2$beta[rngburn,'b[1]']), alpha=0.15) +
  scale_x_continuous(name=expression(beta[2])) +
  scale_y_continuous(name=expression(beta[1])) + 
  theme_classic() + 
  coord_cartesian(xlim=c(-3,4),ylim=c(-4, -.1))
#panel11 <- ggMarginal(panel11, type = "histogram")
#panel22 <- ggMarginal(panel22, type = "histogram")
#panel33 <- ggMarginal(panel33, type = "histogram")
#panel41 <- ggMarginal(panel41, type = "histogram")
#panel43 <- ggMarginal(panel43, type = "histogram")


vp0a <- viewport(width=0.33, height=.25, x=0.00, y=1.00, just=c("left", "top"))
vp1a <- viewport(width=0.33, height=.25, x=0.33, y=1.00, just=c("left", "top"))
vp2a <- viewport(width=0.33, height=.25, x=0.66, y=1.00, just=c("left", "top"))
vp0b <- viewport(width=0.33, height=.25, x=0.00, y=0.75, just=c("left", "top"))
vp1b <- viewport(width=0.33, height=.25, x=0.33, y=0.75, just=c("left", "top"))
vp2b <- viewport(width=0.33, height=.25, x=0.66, y=0.75, just=c("left", "top"))
vp0c <- viewport(width=0.33, height=.25, x=0.00, y=0.50, just=c("left", "top"))
vp1c <- viewport(width=0.33, height=.25, x=0.33, y=0.50, just=c("left", "top"))
vp2c <- viewport(width=0.33, height=.25, x=0.66, y=0.50, just=c("left", "top"))
vp0d <- viewport(width=0.33, height=.25, x=0.00, y=0.25, just=c("left", "top"))
vp1d <- viewport(width=0.33, height=.25, x=0.33, y=0.25, just=c("left", "top"))
vp2d <- viewport(width=0.33, height=.25, x=0.66, y=0.25, just=c("left", "top"))

pdf("~/Desktop/fig1.pdf", width=6, height=7)
#quartz()
 print(panel11, vp=vp0a)
 print(panel.blank, vp=vp1a)
 print(panel.blank, vp=vp2a)
 print(panel.blank, vp=vp0b)
 print(panel22, vp=vp1b)
 print(panel.blank, vp=vp2b)
 print(panel11, vp=vp0c)
 print(panel22, vp=vp1c)
 print(panel33, vp=vp2c)
 print(panel41, vp=vp0d)
 print(panel22, vp=vp1d)
 print(panel43, vp=vp2d)
dev.off()






rng0 = 1:10000
rng1 = 1:100
rng2 = 9900:10000
p0 = 
  ggplot() + 
  geom_hline(aes(yintercept=c(.527-1.96*.546,.527+1.96*.546)), color='black', linetype=3) +
  geom_line(aes(y=samples.guide.adapt$beta[rng0,'b[2]'],x=rng0, color='d')) +
  geom_line(aes(y=samples.guide$beta[rng0,'b[2]'],x=rng0, color='c')) +
  geom_line(aes(y=samples$beta[rng0,'b[2]'],x=rng0, color='a')) +
  scale_y_continuous(name=expression(beta[2])) +
  scale_x_continuous(name='Iteration') + 
  scale_color_grey(name='',labels=c("Random walk", "Guided",  "Guided, adaptive")) + 
  theme_classic() + theme(legend.position =  c(1,.01), legend.justification = c(1,0), legend.background = element_blank()) +
  coord_cartesian(ylim=c(-3,2))

p1 = 
  ggplot() + 
  geom_hline(aes(yintercept=c(.527-1.96*.546,.527+1.96*.546)), color='black', linetype=3) +
  geom_line(aes(y=samples.guide.adapt$beta[rng1,'b[2]'],x=rng1, color='d')) +
  geom_line(aes(y=samples.guide$beta[rng1,'b[2]'],x=rng1, color='c')) +
  geom_line(aes(y=samples$beta[rng1,'b[2]'],x=rng1, color='a')) +
  scale_y_continuous(name='') +
  scale_x_continuous(name='', breaks=c(0,100)) + 
  scale_color_grey(name='',labels=c("Random walk" "Guided",  "Guided, adaptive"), guide=FALSE) + 
  theme_classic() + theme(axis.title.y = element_blank(), axis.line.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  theme(legend.position =  c(1,.01), legend.justification = c(1,0), legend.background = element_blank()) +
  coord_cartesian(ylim=c(-3,2))

p2 = ggplot() + 
  geom_hline(aes(yintercept=c(.527-1.96*.546,.527+1.96*.546)), color='black', linetype=3) +
  geom_line(aes(y=samples.guide.adapt$beta[rng2,'b[2]'],x=rng2, color='d')) +
  geom_line(aes(y=samples.guide$beta[rng2,'b[2]'],x=rng2, color='c')) +
  geom_line(aes(y=samples$beta[rng2,'b[2]'],x=rng2, color='a')) +
  scale_y_continuous(name='') +
  scale_x_continuous(name=' ', breaks=c(9900,10000)) + 
  scale_color_grey(name='',labels=c("Random walk", "Guided",  "Guided, adaptive"), guide=FALSE) + 
  theme_classic() + theme(axis.title.y = element_blank(), axis.line.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  theme(legend.position =  c(1,.01), legend.justification = c(1,0), legend.background = element_blank()) +
  coord_cartesian(ylim=c(-3,2))

# unused trace of risk difference
p3 = ggplot() + 
  geom_line(aes(y=samples.guide.adapt$beta$rd,x=samples.guide.adapt$beta$iter, color='d')) +
  geom_line(aes(y=samples$beta$rd,x=samples$beta$iter, color='a')) +
  geom_line(aes(y=samples.guide$beta$rd,x=samples.guide$beta$iter, color='c')) +
  scale_y_continuous(name='') +
  scale_x_continuous(name=' ') + 
  scale_color_grey(name='',labels=c("Random walk", "Guided",  "Guided, adaptive")) + 
  theme_classic() + theme(axis.title.y = element_blank(), axis.line.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  theme(legend.position =  c(1,.01), legend.justification = c(1,0), legend.background = element_blank())+
  coord_cartesian(ylim=c(-3,2))

vp0 <- viewport(width=0.5, height=1, x=0, y=1, just=c("left", "top"))
vp1 <- viewport(width=0.25, height=1, x=.5, y=1, just=c("left", "top"))
vp2 <- viewport(width=0.2, height=1, x=0.75, y=1, just=c("left", "top"))
pdf("~/Desktop/fig2.pdf", width=7, height=3.5)
 print(p0, vp=vp0)
 print(p1, vp=vp1)
 print(p2, vp=vp2)
dev.off()





  
burn = 1000
samples = mh.adaptive.guided.logistic(iter=101000, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=FALSE)
samples.guide = mh.adaptive.guided.logistic(iter=101000, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=FALSE, guided=TRUE)
samples.guide.adapt = mh.adaptive.guided.logistic(iter=101000, y=y, X=X, chain=1, prop.sigma.start=0.1, seed=19238, inits=NULL, adaptive=TRUE, guided=TRUE)

# random walk
ests = samples$beta[samples$beta$iter>burn,'b[2]']
rd = samples$beta[samples$beta$iter>burn,'rd']
#or, SE
(beta.hat  <- mean(ests))
exp(beta.hat)
(se  <- sd(ests))
# normality based CI
exp(beta.hat);exp(beta.hat-1.96*se);exp(beta.hat+1.96*se)
#percentile based CI
exp(quantile(ests, c(0.5, 0.025, 0.975)))
# mode
dd <- density(ests)
plot(dd)
exp(dd$x[which.max(dd$y)])

#RD, SE
(rd.hat  <- mean(rd))
(se.rd  <- sd(rd))
# normality based CI
rd.hat;rd.hat-1.96*se.rd;rd.hat+1.96*se.rd
#percentile based CI
(quantile(rd, c(0.5, 0.025, 0.975)))
# mode
dd <- density(rd)
(dd$x[which.max(dd$y)])



# guided
ests = samples.guide$beta[samples.guide$beta$iter>burn,'b[2]']
rd = samples.guide$beta[samples.guide$beta$iter>burn,'rd']
#or, SE
(beta.hat  <- mean(ests))
(se  <- sd(ests))
median(ests)
# normality based CI
exp(beta.hat);exp(beta.hat-1.96*se);exp(beta.hat+1.96*se)
#percentile based CI
exp(quantile(ests, c(0.5, 0.025, 0.975)))
dd <- density(ests)
exp(dd$x[which.max(dd$y)])


#RD, SE
(rd.hat  <- mean(rd))
(se.rd  <- sd(rd))
median(rd)
# normality based CI
rd.hat;rd.hat-1.96*se.rd;rd.hat+1.96*se.rd
#percentile based CI
(quantile(rd, c(0.5, 0.025, 0.975)))
# mode
dd <- density(rd)
(dd$x[which.max(dd$y)])



# guided, adaptive
ests = samples.guide.adapt$beta[samples.guide.adapt$beta$iter>burn,'b[2]']
rd = samples.guide.adapt$beta[samples.guide.adapt$beta$iter>burn,'rd']
#or, SE
(beta.hat  <- mean(ests))
(se  <- sd(ests))
median(ests)
# normality based CI
exp(beta.hat);exp(beta.hat-1.96*se);exp(beta.hat+1.96*se)
#percentile based CI
exp(quantile(ests, c(0.5, 0.025, 0.975)))
# mode
dd <- density(ests)
exp(dd$x[which.max(dd$y)])

#RD, SE
(rd.hat  <- mean(rd))
(se.rd  <- sd(rd))
median(rd)
# normality based CI
rd.hat;rd.hat-1.96*se.rd;rd.hat+1.96*se.rd
#percentile based CI
(quantile(rd, c(0.5, 0.025, 0.975)))
# mode
dd <- density(rd)
(dd$x[which.max(dd$y)])



#### doing in off-the-shelf MCMC software
# repeat in jags
jagsmod <- '
model{
  mu = X%*%b
 for(i in 1:N){
  y[i] ~ dbern(1/(1+exp(-mu[i])))
 } #N
 # priors
  b[1] ~ dnorm(0, 1/100)
  b[2] ~ dnorm(0, 1/(0.5))
} #model
'
tmpf2=tempfile()
tmpf2 = '/Users/akeil/temp/_____jmod.txt'
tmps=file(tmpf2,"w")
cat(jagsmod,file=tmps)
close(tmps)

burn = 1000
jags.data <- list(X = cbind(rep(1,length(x)), x),y=as.numeric(y),N=length(x))
jags.check <- jags.model(file=tmpf2, data=jags.data, n.chains=4, n.adapt = 1000)
update(jags.check, n.iter=burn)
jags.fit <- coda.samples(jags.check, variable.names=c('b[1]', 'b[2]'), n.iter=10000)

#or, SE
(beta.hat  <- jags.sum$statistics[c('b[2]'),1])
(se  <- jags.sum$statistics[c('b[2]'),2])
# normality based CI
exp(beta.hat);exp(beta.hat-1.96*se);exp(beta.hat+1.96*se)
#percentile based CI
exp(jags.sum$quantiles[2,c(3,1,5)])




# repeat in stan
stanmod <- '
data{
 int N;
 real offset;
 int<lower=0,upper=1> y[N];
 matrix[N,2] X;
}
parameters{
 vector[2] b;
}
model{
 //b[1] ~ normal(0, 10);
 //b[2] ~ normal(0, sqrt(0.5));
 y ~ bernoulli_logit(X*b);
}
generated quantities{
 real rd;
 real rd1000;
 rd = 0;
 for(i in 1:N){
   rd = rd + (inv_logit(log(offset)+b[1]+b[2]) - inv_logit(log(offset)+b[1]))/N;
 }
rd1000 = rd*1000;
}
'
baserisk = 4.8/100000
baseodds = baserisk/(1-baserisk)
caseodds = mean(y)/(1-mean(y))


stan.data <- list(X = cbind(rep(1,length(x)), x),y=as.numeric(y),N=length(x), offset=baseodds/caseodds)
stanmod.check <- stan(model_code = stanmod, data=stan.data, chains=1, iter=1, refresh=-1)
stanmod.fit <- stan(fit = stanmod.check, data=stan.data, chains=4,warmup = 1000, iter=101000, refresh=-1)
print(stanmod.fit, digits_summary = 4)

#or, SE
(beta.hat <- summary(stanmod.fit)$summary[2,1])
(se <- summary(stanmod.fit)$summary[2,3])
# normality based CI
exp(beta.hat);exp(beta.hat-1.96*se);exp(beta.hat+1.96*se)
#percentile based CI
exp(summary(stanmod.fit)$summary[2,c(6,4,8)])
