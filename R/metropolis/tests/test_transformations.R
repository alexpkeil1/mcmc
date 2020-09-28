# transfourmations

mm = matrix(runif(200), nrow=20, ncol=10)
apply(mm, 2, mean)
mm2 = metropolis:::.zstd(mm)


stopifnot(all.equal(as.numeric(apply(mm2, 2, mean)), as.numeric(rep(0, 10)), tolerance = 3e-1))
stopifnot(all.equal(as.numeric(apply(mm2, 2, sd)), as.numeric(rep(1, 10)), tolerance = 3e-1))
