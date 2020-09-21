#### transformations ####

.zstd <- function(MAT, bincols=NULL, factcols=NULL){
  # mm = matrix(runif(200), nrow=20, ncol=10)
  # apply(mm, 2, mean)
  # mm2 = .zstd(mm)
  # apply(mm2, 2, mean)
  # apply(mm2, 2, sd)

  #mf = apply(MAT, 2, mean)
  mf = colMeans(MAT)
  sf = apply(MAT, 2, sd)
  mf[c(bincols, factcols)] = 0
  sf[c(bincols, factcols)] = 1
  MAT = sweep(MAT, 2, mf, FUN="-")
  MAT = sweep(MAT, 2, sf, FUN="/")
  attr(MAT, "mf") = mf
  attr(MAT, "sf") = sf
  MAT
}

expit <- function(mu){
  #' @title Inverse logit transform
  #'
  #' @param mu log-odds
  #' @return returns a scalar or vector the same length 
  #' as mu with values that are the inverse logit transform 
  #' of mu 
  #' @export
  #' @examples
  #' logodds = rnorm(10)
  #' expit(logodds)
  #' logodds = log(1.0)
  #' expit(logodds)
  1/(1+exp(-mu))
}
