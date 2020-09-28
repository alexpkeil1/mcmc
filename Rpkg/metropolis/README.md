`metropolis` v0.1.8

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) [![](http://www.r-pkg.org/badges/version/metropolis)](http://www.r-pkg.org/pkg/metropolis)
[![CRAN RStudio mirror downloads](http://cranlogs.r-pkg.org/badges/metropolis)](http://www.r-pkg.org/pkg/metropolis) [![Build Status](https://travis-ci.com/alexpkeil1/mcmc.svg?branch=master)](https://travis-ci.com/alexpkeil1/mcmc) [![codecov](https://codecov.io/gh/alexpkeil1/mcmc/branch/master/graph/badge.svg)](https://codecov.io/gh/alexpkeil1/mcmc)

    devtools::install_github("alexpkeil1/mcmc", subdir="Rpkg/metropolis", build_vignettes = TRUE)
    library(metropolis)
    data("magfields", package="metropolis")
    # see lots of examples here
    vignette("metropolis-vignette", package="metropolis")

    table(magfields)
    >          x
    > y      0   1
    >    0 193   5
    >    1  33   3

    summary(glm(y~x, family=binomial(), data=magfields))$coefficients
    >              Estimate Std. Error   z value     Pr(>|z|)
    > (Intercept) -1.766183   0.188373 -9.375988 6.853094e-21
    > x            1.255357   0.754200  1.664488 9.601492e-02
  
## Example: fitting a logistic regression model with uniform priors under two different Metropolis algorithms

### Random walk Metropolis
    res.rw = metropolis_glm(y ~ x, data=magfields, family=binomial(), iter=20000, 
                        burnin=3000, adapt=FALSE, guided=FALSE, block=TRUE, inits=c(2,-3), 
                        control = metropolis.control(prop.sigma.start = c(0.05, .1)))
    
    summary(res.rw, keepburn = FALSE)
    
    > $nsamples
    > [1] 20000
    > 
    > $sd
    > (Intercept)           x 
    >   0.1932110   0.7472765 
    > 
    > $se
    > (Intercept)           x 
    >  0.01153963  0.08424655 
    > 
    > $ESS_beta
    > [1] 280.33684  78.67893
    > 
    > $postmean
    >                  mean normal_lci normal_uci
    > (Intercept) -1.770576 -2.1492695  -1.391882
    > x            1.086751 -0.3779107   2.551413
    > 
    > $postmedian
    >                median   pctl_lci  pctl_uci
    > (Intercept) -1.764117 -2.1664080 -1.415535
    > x            1.092517 -0.3861917  2.494210
    > 
    > $postmode
    >                  mode   hpd_lci   hpd_uci
    > (Intercept) -1.760881 -2.581002 -1.470022
    > x            1.050640 -1.863551  2.282397
    

    
    plot(res.rw, par = 1:2, keepburn=TRUE)

![RW1](fig/random_walkb0.png)

![RW2](fig/random_walkb1.png)

### Guided, adaptive random walk Metropolis
    res.ga = metropolis_glm(y ~ x, data=magfields, family=binomial(), iter=20000, 
                                        burnin=3000, adapt=TRUE, guided=TRUE, block=FALSE, inits=c(2,-3))
    
    summary(res.ga, keepburn = FALSE)
    
    > $nsamples
    > [1] 20000
    > 
    > $sd
    > (Intercept)           x 
    >   0.1911599   0.7936077 
    > 
    > $se
    > (Intercept)           x 
    > 0.002199134 0.009393553 
    > 
    > $ESS_beta
    > [1] 7555.971 7137.597
    > 
    > $postmean
    >                  mean normal_lci normal_uci
    > (Intercept) -1.779409 -2.1540819  -1.404735
    > x            1.197478 -0.3579937   2.752949
    > 
    > $postmedian
    >                median   pctl_lci  pctl_uci
    > (Intercept) -1.776448 -2.1625510 -1.418052
    > x            1.217966 -0.4181273  2.659286
    > 
    > $postmode
    >                  mode   hpd_lci   hpd_uci
    > (Intercept) -1.773507 -2.578257 -1.472365
    > x            1.260855 -3.812090  2.458669
    plot(res.ga, par = 1:2, keepburn=TRUE)

![AG1](fig/adaptive_guidedb0.png)

![AG2](fig/adaptive_guidedb1.png)
