library("metropolis")
# logistic model
set.seed(12321)
dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
reslog = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=50000, burnin=3000, 
           adapt=TRUE, guided=TRUE, block=FALSE)
print(reslog)
summary(reslog)
bayescoef = apply(reslog$parms[-c(1:3000),], 2, mean)
logmod = glm(y ~ x1 + x2, family=binomial(), data=dat)

stopifnot(all.equal(as.numeric(logmod$coefficients), as.numeric(bayescoef[3:5]), tolerance = 3e-1))

# linear model
datlin = data.frame(y = rnorm(100, 1, 0.5), x1=runif(100), x2 = runif(100), x3 = rpois(100, .2))
reslin = metropolis_glm(y ~ x1 + x2 + factor(x3), data=datlin, family=gaussian(), inits="glm", 
        iter=50000, burnin=3000, adapt=TRUE, guide=TRUE, block=FALSE)
bayescoef2 = apply(reslin$parms[-c(1:3000),], 2, mean)
linmod = glm(y ~ x1 + x2+ factor(x3), family=gaussian(), data=datlin)

stopifnot(all.equal(as.numeric(linmod$coefficients), as.numeric(bayescoef2[4:9]), tolerance = 3e-1))
