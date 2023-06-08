# to do: unify hyperprior terminology
# to do: move Xint to optional


# proposed hyperprior: _parammeter_priorterm0
#  e.g. _beta_mu is the prior mean of the _beta parameters

# gibbs sampler + derivations
using Distributions, Random, DataFrames, StatsBase, LinearAlgebra
# utilities
#using FreqTables
#using InvertedIndices

#function calcmd!(munc, muint, ymean, yscale, Xint, _beta)
#  mul!(munc, Xint[1], _beta[2:end])
#  mul!(muint, Xint[2], _beta[2:end])
#  cm1 = ymean + yscale*(_beta[1] + mean(munc))
#  cm0 = ymean + yscale*(_beta[1] + mean(muint))
#  md = cm1-cm0
#  cm1, cm0, md
#end
#
#function gibbslm(y,X, Xint, iter, burnin, rng; ymean=0.0, yscale=1.0)
#  # independently derived, but matches R code at https://www4.stat.ncsu.edu/~reich/ST740/Computing2.pdf
#  (N,p) = size(X)
#  X = hcat(ones(N), X)
#  Xt = transpose(X)
#  xtx = Xt * X
#  #Xint = [hcat(ones(N), Xint[1]), hcat(ones(N), Xint[2])]
#  (N,p) = size(X)
#  Nf = Float64(N)
#  Ni = size(Xint[1], 1)
#  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
#  # constants/hyperpriors
#  _mu, _tau = zeros(p), ones(p)*1000
#  _alpha, _lambda = 0.0, 0.0 #  jeffrey's prior
#  # initial values
#  _sigma = rand()*2
#  _beta = rand(p)
#  bb = rand(p)
#  _beta_store = zeros(iter, p)
#  _sigma_store = zeros(iter)
#  _py_store = zeros(iter, 3)
#  xtx = transpose(X) * X
#  for i in 1:iter
#    # update sigma
#    se = (y .- X * _beta).^2.
#    a = _alpha + N/2.
#    b = _lambda + sum(se)/2.
#    _sigma = sqrt(rand(rng, InverseGamma(a, b)))
#    # update beta
#    for j in 1:p
#      bb[:] = _beta
#      bb[j] = 0
#      V = 1/(xtx[j,j] / _sigma^2 + 1 / _tau[j]^2)
#      M = (transpose(y .- X * bb ) * X[:,j] / _sigma^2 + _mu[j]/_tau[j]^2)*V
#      _beta[j] = rand(rng, Normal(M, sqrt(V)))
#    end
#    #
#    cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
#    #
#    _py_store[i,:] = vcat(cm1, cm0, md)
#    _sigma_store[i] = _sigma
#    _beta_store[i,:] = _beta
#  end
#  df = convert(DataFrame, hcat(_py_store, _beta_store,_sigma_store))
#  rename!(df, vcat(:m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:(p-1)],:sigma))
#  df[(burnin+1):iter,:]
#end
#
#
#function gibbs2lm(y,X, Xint, iter, burnin, rng; ymean=0.0, yscale=1.0)
#  # block sampler
#  (N,p) = size(X)
#  X = hcat(ones(N), X)
#  Xt = transpose(X)
#  xtx = Xt * X
#  #Xint = [hcat(ones(N), Xint[1]), hcat(ones(N), Xint[2])]
#  p = size(X)[2]
#  (N,p) = size(X)
#  Nf = Float64(N)
#  Ni = size(Xint[1], 1)
#  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
#  # constants/hyperpriors
#  _mu, _tau = zeros(p), ones(p)*1000
#  _alpha, _lambda = 0.0, 0.0 #  jeffrey's prior
#  # initial values
#  _sigma = rand()*2
#  _beta = rand(p)
#  bb = rand(p)
#  _beta_store = zeros(iter, p)
#  _sigma_store = zeros(iter)
#  _py_store = zeros(iter, 3)
#  Xt = transpose(X)
#  xtx = Xt * X
#  for i in 1:iter
#    # update sigma
#    se = (y .- X * _beta).^2.
#    a = _alpha + N/2.
#    b = _lambda + sum(se)/2.
#    _sigma = sqrt(rand(rng, InverseGamma(a, b)))
#    # update beta
#    Lam = Diagonal(_tau.^2)
#    iLam = inv(Lam)
#    V = Symmetric(inv(Xt * X ./_sigma^2  + iLam))
#    M = V * (Xt * y ./_sigma^2 + iLam * _mu)
#    #
#    cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
#    #
#    _py_store[i,:] = vcat(cm1, cm0, md)
#    _beta = rand(rng, MvNormal(M, V))
#    _sigma_store[i] = _sigma
#    _beta_store[i,:] = _beta
#  end
#  df = convert(DataFrame, hcat(_py_store, _beta_store, _sigma_store))
#  rename!(df, vcat(:m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:(p-1)],:sigma))
#  df[(burnin+1):iter,:]
#end


# now better version that follows other conventions
function inputchecks(y,X,Xint,p, pl)
  if !any(isnan.(Xint[1]))
    for xx in Xint
      @assert size(xx,2)==size(X,2) "X and Xint have different numbers of columns"
    end
  end
  @assert size(y,1) == size(X,1) "y and X have different numbers of rows"
  @assert sum(pl) == p "sum(pl) should equal number of columns of X (without intercept)"
end



function initialize!(_beta, binits, p)
  if isnan(binits[1])
    _beta .= randn(p)*2
  else
    _beta .= copy(binits)
  end
end

function inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0, binits, pl)
  # constants
  (N,p) = size(X)
  Nf = Float64(N)
  if isnan(pl[1])
    plidx, pls, j = NaN, NaN, NaN
  else
    j = size(pl, 1)
    inputchecks(y,X,Xint,p, pl)
    pls = vcat(2, (cumsum(pl) .+ 2)[1:(end-1)])
    plidx = [pls[g]:cumsum(pl)[g]+1 for g in 1:j]
  end
  Ni = size(Xint[1], 1)
  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
  X = hcat(ones(N), X)
  _beta = Array{Float64,1}(undef, p+1)
  initialize!(_beta, binits, p+1)
  mu = X * _beta
  res = y .- mu
  #Xint = Xint
  Xt = transpose(X)
  xtx = Xt * X
  xty = Xt * y
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _A_ms = Array{Float64,1}(undef, p+1)
  if size(_tau_beta0^(-2),1) == 1
    _iLams = Diagonal(vcat(_tau_eta0^(-2), fill(_tau_beta0^(-2), p))) # precision prior on beta coefficients
    _muvec = vcat(_mu_eta0, fill(_mu_beta0, p))      # mean prior on beta coefficients
  else
    @assert size(_mu_beta0) == p-1 && size(_tau_beta0) == p-1
    _iLams = Diagonal(vcat(_tau_eta0^(-2), _tau_beta0.^(-2))) # precision prior on beta coefficients
    _muvec = vcat(_mu_eta0, _mu_beta0)      # mean prior on beta coefficients  
  end
  a = _sigmaa0 + Nf/2.
  _beta_store = Array{Float64,2}(undef, iter, p+1)
  _py_store = zeros(iter, 3)
  _sig_store = zeros(iter)
  return(
    N, Nf, p, _beta, mu, plidx, pls, j,
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec,
    a, res,
    munc, muint,
    _beta_store, _py_store,
    _sig_store
  )
end
inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0, binits=NaN) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0, binits, NaN)
inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0, NaN, NaN)
inits(y,X,Xint,iter) = inits(y,X,Xint,iter,0.0,0.0,0.0,0.0, 0.0, NaN, NaN)

################################################################################################
#  Better version that follows other conventions
################################################################################################


function updatebeta!(rng, _beta, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xtx, xty)
  # block sampling of all model coefficients
  _A.data .= xtx .* _invsigma2 + _iLams 
  _A_ms .= xty .* _invsigma2 + _iLams * _muvec
  _beta .= rand(rng, MvNormalCanon(_A_ms,_A)) # parameters mu*Sigma^-1, Sigma^-1
end

function sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta)
  # sample precision from a gamma prior
  res .= y .- X * _beta 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #a = _sigmaa0 + Nf/2.
  b = _sigmab0 + se/2.
  #sigma2 = rand(rng, InverseGamma(a, b))
  rand(rng, Gamma(a, 1. / b))
end

function sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, nu, _invsigma2, Nf)
  # sample precision from a half cauchy prior (via representation as a scale mixture of gammas)
  va = 0.5 + nu/2.0
  vb = nu * _invsigma2 + _sigmab0^(-2) # _sigmab0 = A
  V = rand(rng, InverseGamma( va, vb))
  res .= y .- X * _beta 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #
  sa = (Nf + nu)/2.0
  sb = nu/V + se/2.
  rand(rng, Gamma(sa, 1. / sb))
end

function calcmd!(munc, muint, ymean, yscale, Xint, _beta)
  mul!(munc, Xint[1], _beta[2:end])
  mul!(muint, Xint[2], _beta[2:end])
  cm1 = ymean + yscale*(_beta[1] + mean(munc))
  cm0 = ymean + yscale*(_beta[1] + mean(muint))
  md = cm1-cm0
  cm1, cm0, md
end

function gibbs_lm(y,X,Xint,iter,burnin,rng; chain=1, binits=NaN,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 1000.,                              # prior mean, sd of intercept
               _sigmaa0 = 0., _sigmab0 = 0.0,                                # precision priors
                ymean = 0.0, # rescaling coefficients
                yscale = 1.0 # rescaling coefficients
)

   (
    N, Nf, p, _beta, mu, plidx, pls, j,
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec,
    a, res,
    munc, muint,
    _beta_store, _py_store,
    _sig_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0, binits)
  
  # hierarchy not yet implemented
  #j = size(pl, 1)
  #pls = vcat(1, (cumsum(pl) .+ 1)[1:(end-1)])
  #plidx = [pls[g]:cumsum(pl)[g] for g in 1:j]
  #
  # priors
  # initials
  _invsigma2 = rand()*4
  _invtausq = _iLams.diag; # - creates a view!
  for i in 1:iter
    ############
    # sample beta (with normal mixture prior, where mixing distribution)
    ############
    updatebeta!(rng, _beta, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xtx, xty)
    #
    ############
    # sample sigma from InverseGamma
    ############
    _invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta) # inverse gamma
    #_invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, 1.0, _invsigma2, Nf) # half-cauchy
    ####################
    # update mean difference
    ####################
    cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)

    # store
    _py_store[i,:] = vcat(cm1, cm0, md)
    _beta_store[i,:] = _beta
    _sig_store[i] = _invsigma2
  end
    _sig_store = sqrt.(inv.(_sig_store))
  # this should be new template
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], 
             _py_store, _beta_store, _sig_store)
  nms = vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta_$i") for i in 0:p],
                  :sigma)
  if !isapprox(yscale, 1.0)
    _beta_ustd = _beta_store .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("betau_$i") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #exnm = names(df)
  #renamer = Dict([exnm[i] => nms[i] for i in 1:length(exnm)])
  #rename!(df, nms)
  df[(burnin+1):iter,:]
end

gibbs_lm(y,X,Xint,iter,burnin;thin=1,chain=1,ymean = 0.0, yscale=1.0, _mu_eta0 = 0., _tau_eta0 = 1000., _mu_beta0 = 0., _tau_beta0 = 1000., _sigmaa0 = 0., _sigmab0 = 0.0 ) = gibbs_lm(y,X,Xint, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, ymean = ymean, yscale=yscale, _mu_eta0 = _mu_eta0, _tau_eta0 = _tau_eta0, _mu_beta0 = _mu_beta0, _tau_beta0 = _tau_beta0, _sigmaa0 = _sigmaa0, _sigmab0 = _sigmab0 )
gibbs_lm(y,X,Xint,iter;thin=1,chain=1,ymean = 0.0, yscale=1.0, _mu_eta0 = 0., _tau_eta0 = 1000., _mu_beta0 = 0., _tau_beta0 = 1000., _sigmaa0 = 0., _sigmab0 = 0.0 ) = gibbs_lm(y,X,Xint, iter, 0;thin=thin,chain=chain, ymean = ymean, yscale=yscale, _mu_eta0 = _mu_eta0, _tau_eta0 = _tau_eta0, _mu_beta0 = _mu_beta0, _tau_beta0 = _tau_beta0, _sigmaa0 = _sigmaa0, _sigmab0 = _sigmab0 )
;

################################################################################################
#  Hierarchical version (Gibbs)
################################################################################################

function updatemu!(rng, _mu, _muvec, _beta, j, pl, plidx, _tausq, _mu_tau0, _mu_mu0)    
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    iV = pl[q] / _tausq[q] +  _mu_tau0[q] .^(-2)
    MoV = (sum(bl) / _tausq[q] .+ _mu_mu0[q] * _mu_tau0[q] .^(-2))
    _mu[q] = rand(rng, NormalCanon(MoV, iV))
    _muvec[plidx[q]] .= _mu[q]
  end
end

function updatetausq!(rng, _tausq, _iLams, _beta, j, pl, plidx, _mu, _tau_mu0, _tau_tau0)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres = bl .- _mu[q]
    bse = transpose(bres) * bres
    a_tau = _tau_mu0[q] + pl[q]/2.
    b_tau = _tau_tau0[q] + bse/2.
    _tausq[q] = rand(rng, Gamma(a_tau, 1. / b_tau))
    _iLams.diag[plidx[q]] .= inv(_tausq[q])
  end
end

 ### begin unimplemented ###
function updatebres!(bres, _beta, _mu, plidx, j)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres[plidx[q]] .= bl .- _mu[q]
  end
end

function updatev!(rng, _V, _tausq, _tau_tau0, nu, j)
  @inbounds for q in 1:j # loop over hierarchical groups
    _V[q] = rand(rng, InverseGamma((nu+1.)/2.0, _tau_tau0[q]^(-2.0) + nu / _tausq[q]))
  end
end

function updatetausq_robust!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    #bres[plidx[q]] .= bl .- _mu[q]*_alpha
    bres[plidx[q]] .= bl .- _mu[q]*_alpha[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    _tausq[q] = rand(rng, InverseGamma(a_tau, b_tau))
  end
end

function updatetausq_robust_or!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha; K=9)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  # including ordered over-relaxation
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    #bres[plidx[q]] .= bl .- _mu[q]*_alpha
    bres[plidx[q]] .= bl .- _mu[q]*_alpha[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    orcand = rand(rng, InverseGamma(a_tau, b_tau), K)
    sp = sortperm(vcat(_tausq[q], orcand))
    selidx = K - sp[1]  +2 
    _tausq[q] = orcand[sp[2:end] .== selidx][1]
  end
end

 ### end unimplemented ###

function gibbs_hlm(y,X,Xint,iter,burnin,rng; chain = 1, binits=NaN,
               pl = NaN,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _sigmaa0 = 0.0, _sigmab0 = 0.0,                                # precision priors
               _mu_mu0= 0.0, _mu_tau0 = 10.0,                                 # mu
               _tau_mu0 = 0., _tau_tau0 = 1.0,                                
                ymean = 0.0, # rescaling coefficients
                yscale = 1.0 # rescaling coefficients
)

   (
    N, Nf, p, _beta, mu, plidx, pls, j,
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec,
    a, res,
    munc, muint,
    _beta_store, _py_store,
    _sig_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,0.0,rand(rng)*2.0, _sigmaa0, binits, pl)
  println(j)
  # hierarchy not yet implemented
  #j = size(pl, 1)
  #pls = vcat(1, (cumsum(pl) .+ 1)[1:(end-1)])
  #plidx = [pls[g]:cumsum(pl)[g] for g in 1:j]
  #
  # priors
  # initials
  _invsigma2 = rand()*4
  _invtausq = _iLams.diag; # - creates a view!
  _mu = randn(rng, j)
  _tausq = rand(rng, j)
  _mu_store = Array{Float64, 2}(undef, iter, j)
  _tausq_store = Array{Float64, 2}(undef, iter, j)
  for i in 1:iter
    ############
    # sample beta (with normal mixture prior, where mixing distribution)
    ############
    updatebeta!(rng, _beta, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xtx, xty)
    #
    ############
    # sample sigma from InverseGamma
    ############
    _invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta) # inverse gamma
    #_invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, 1.0, _invsigma2, Nf) # half-cauchy
    updatemu!(rng, _mu, _muvec, _beta, j, pl, plidx, _tausq, _mu_tau0, _mu_mu0)    
    updatetausq!(rng, _tausq, _iLams, _beta, j, pl, plidx, _mu, _tau_mu0, _tau_tau0)
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
      # store
      _py_store[i,:] = vcat(cm1, cm0, md)
      _beta_store[i,:] = _beta
      _sig_store[i] = _invsigma2
      _mu_store[i,:] = _mu
      _tausq_store[i,:] = _tausq
    end
  end
    _sig_store = sqrt.(inv.(_sig_store))
  # this should be new template
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store, _mu_store, _tausq_store, _sig_store)
  nms = vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta_$i") for i in 0:p], [Symbol("mu_$i") for i in 1:j], [Symbol("tausq_$i") for i in 1:j],
                  :sigma)
  if !isapprox(yscale, 1.0)
    _beta_ustd = _beta_store .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("betau_$i") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #exnm = names(df)
  #renamer = Dict([exnm[i] => nms[i] for i in 1:length(exnm)])
  #rename!(df, nms)
  df[(burnin+1):iter,:]
end

gibbs_hlm(y,X,Xint,iter,burnin;pl = NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0, _mu_eta0 = 0., _tau_eta0 = 1000., _mu_beta0 = 0., _tau_beta0 = 1000., _sigmaa0 = 0., _sigmab0 = 0.0 ) = gibbs_hlm(y,X,Xint, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));pl = pl,thin=thin,chain=chain, ymean = ymean, yscale=yscale, _mu_eta0 = _mu_eta0, _tau_eta0 = _tau_eta0, _mu_beta0 = _mu_beta0, _tau_beta0 = _tau_beta0, _sigmaa0 = _sigmaa0, _sigmab0 = _sigmab0 )
gibbs_hlm(y,X,Xint,iter;pl = NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0, _mu_eta0 = 0., _tau_eta0 = 1000., _mu_beta0 = 0., _tau_beta0 = 1000., _sigmaa0 = 0., _sigmab0 = 0.0 ) = gibbs_hlm(y,X,Xint, iter, 0;pl = pl,thin=thin,chain=chain, ymean = ymean, yscale=yscale, _mu_eta0 = _mu_eta0, _tau_eta0 = _tau_eta0, _mu_beta0 = _mu_beta0, _tau_beta0 = _tau_beta0, _sigmaa0 = _sigmaa0, _sigmab0 = _sigmab0 )
;