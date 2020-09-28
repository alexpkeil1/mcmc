#!/usr/bin/env julia
# a series of gibbs variable selection functions

#REPL.TerminalMenus.config(charset=:ascii)
using LinearAlgebra, Distributions, DataFrames, StatsBase
using PolyaGammaDistribution
import wellwisejl.summarygibbs, wellwisejl.expit

# data
function expiti(mu::T) where T
  inv(1.0 + exp(-mu))
end

function expit!(mu::Array{T, 1}) where T
  mu .= expiti.(mu)
end

function meanexpit(mu::Array{T, 1}) where T
  mean(expit!(mu))
end


function inputchecks(y,X,Xint,p, pl)
  if !any(isnan.(Xint[1]))
    for xx in Xint
      @assert size(xx,2)==size(X,2) "X and Xint have different numbers of columns"
    end
  end
  @assert size(y,1) == size(X,1) "y and X have different numbers of rows"
  @assert sum(pl) == p "sum(pl) should equal number of columns of X (without intercept)"
end

function inits(rng,iter,binit, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, pa, pb, pl)
  kappa = y .- 0.5
  # indexes
  (N,p) = size(X)
  if isnan(pl[1])
    plidx, pls, j = NaN, NaN, NaN
  else
    j = size(pl, 1)
    inputchecks(y,X,Xint,p, pl)
    pls = vcat(2, (cumsum(pl) .+ 2)[1:(end-1)])
    plidx = [pls[g]:cumsum(pl)[g]+1 for g in 1:j]
  end
  # parameters
  _pi = rand.(rng, Beta.(pa,pb))
  _invsigma2 = rand()
  if size(binit, 1)==1 && isnan(binit)
    _beta = randn(rng, p+1)*2
  else
    _beta::Array{Float64, 1} = copy(binit) # if drawn from coef(glmfit), this can actually change the coefficients without copy()
  end
  # todo check length of mu/tau
  # draw initial selection from prior
  _gamma = vcat(1.0, vcat([Array{Float64,1}(rand(pp) .< _pi[g]) for (g, pp) in enumerate(pl)]...))
  _bg = _gamma .* _beta
  # hyper parameters
  om = Array{Float64,1}(undef, N)
  Om = Diagonal(om)
  _A_ms = Array{Float64, 1}(undef, p+1) 
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _iLams = Diagonal(vcat( _tau_eta0[1]^(-2), [ _mu_tau0[1]^(-2) for j in 1:p]))
  _muvec = vcat(_mu_eta0[1], [_mu_mu0[1] for j in 1:p])
  # storage
  _beta_store = zeros(iter, p+1)
  _gamma_store = zeros(iter, p)
  _bg_store = zeros(iter, p)
  _pi_store = zeros(iter, j)
  _py_store = zeros(iter, 3)
  # pre-allocations
  Nu = _gamma .* _beta
  Nuj = _gamma .* _beta
  X = hcat(ones(N), X)
  Xs = copy(X)
  makex!(Xs,_gamma)
  Xst = transpose(Xs)
  xstx = Xst * Xs
  munc  =  Xint[1] * _bg[2:end]
  muint = Xint[2] * _bg[2:end]
  yr = copy(y)
  bres = Array{Float64, 1}(undef, p+1)
  res = Array{Float64, 1}(undef, N) 
  mu = X * _beta
  # conveniences
  Nf = Float64(N)

  return (
    Nf, p, pl, pls, plidx, j,
    _pi, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, yr, kappa, om, Om, mu
  )
end

function makex!(Xs,_gamma; shift=0)
  @inbounds for (i,g) in enumerate(_gamma)
    Xs[:,i+shift] = Xs[:,i+shift]*g
  end
end

function makex!(X, Xs,_gamma;shift=0)
  copyto!(Xs,X)
  makex!(Xs,_gamma,shift=shift)
end

function fixp!(pj)
  if pj > one(Float64)
    println("pj above 0.0 - check for possible errors if this is frequent")
    #pj = one(Float64)
  elseif pj < zero(Float64)
    println("pj below 0.0 - check for possible errors if this is frequent")
    #pj = zero(Float64)
  elseif isnan(pj)
    println("pj is NaN - check for possible errors if this is frequent")
    #pj = zero(Float64)
  end
  pj
end


function updategamma!(rng, _gamma, _pi, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
  # variable selection indicators
  # algorithm adapted from Kuo and Mallick.
  # contrasts weighted normal likelihood under including vs. excluding a single coefficient to derive
  #  a probability of inclusion (weights equal to prior inclusion probabilities)
  # odds of inclusion are converted to a probability, which is used to take a Bernoulli draw
  rp = randperm(j)         #random order
  #rp = [pp for pp in 1:j] # fixed order
  @inbounds for q in rp                   # loop over hierarchical groups in random order
    vp = sample(plidx[q], size(plidx[q]), replace=false)  # random order (always excludes intercept)
    #vp = [pp for pp in plidx[q]]                         # fixed order
    piq = _pi[q]
    lpiq = log(piq)
    lnpiq = log(1-piq)
    for k in vp                               # loop over variables within group in random order
      Nuj .= _gamma .* _beta
      Nuj[k] = _beta[k]                       # first added in beta
      res .= logpdf.(Bernoulli.(expit(X * Nuj)), y)
      #res .= -log.(1.0 .+ exp.(-X * Nuj)) .* y + -log.(1.0 .+ exp.(X * Nuj)) .* (1.0 .- y) # log(p(y|xnuj))
      rr =  sum(res)
      cj = lpiq + rr            # proportional to likelihood*prior with inclusion
      #res .+=  _beta[k] .* X[:,k]             # zero out the kth index (add in because of residual)
      #rr =  transpose(res) * res
      #dj = lnpiq + halfinvsigma2 * rr           # proportional to likelihood*prior with exclusion
      Nuj[k] = 0.0
      #res .= -log.(1.0 .+ exp.(-X * Nuj)) .* y + -log.(1.0 .+ exp.(X * Nuj)) .* (1.0 .- y)
      res .= logpdf.(Bernoulli.(expit(X * Nuj)), y)
      rr =  sum(res)
      dj = lnpiq + rr           # proportional to likelihood*prior with exclusion
      pj =  1. /(1. + exp( dj - cj))       # C*(lik*prior inclusion)/[C*(lik*prior exclusion) + C*(lik*prior exclusion) ] = posterior probability of inclusion
      _gamma[k] = Float64(rand(rng, Bernoulli(pj)))
    end
  end
end

function updatepi!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
  # update model inclusion priors via beta-bernoulli conjugacy
  @inbounds for q in 1:j
    g1 = sum(_gamma[plidx[q]])
    g0 = pl[q] - g1
    _pi[q] = rand(rng, Beta(pa[q] + g1, pb[q] + g0))
  end
end


function updatepi_or!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb;K=3)
  # update model inclusion priors via beta-bernoulli conjugacy
  @inbounds for q in 1:j
    g1 = sum(_gamma[plidx[q]])
    g0 = pl[q] - g1
    orcand = rand(rng, Beta(pa[q] + g1, pb[q] + g0), K)
    sp = sortperm(vcat(_pi[q], orcand))
    selidx = K - sp[1]  +2 
    _pi[q]= orcand[sp[2:end] .== selidx][1]
  end
end


function updateom!(rng, mu, om, Om, Xs, _bg)
  mu .= Xs * _bg
  om .= rand.(PolyaGamma.(1, mu))
  Om .= Diagonal(om)
end

function updatebeta!(rng, _beta, _bg, _gamma, om, Om, Xs, Xst, kappa, _iLams, _muvec, _A, _A_ms)
   # block sampler for beta coefficients
   _A.data .= Xst * Om * Xs + _iLams
   _A_ms .= Xst * kappa .+ _iLams * _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  _beta .= rand(rng, MvNormalCanon(_A_ms, _A))
  _bg .= _gamma .* _beta
end

function updatebeta_or!(rng, _beta, _bg, _gamma, om, Om, Xs, Xst, kappa, _iLams, _muvec, _A, _A_ms; K=11, p=p)
   # block sampler for beta coefficients
   # ordered over-relaxation by a random member of b
   # does not seem terribly effective
  _A.data .= Xst * Om * Xs + _iLams
  _A_ms .= Xst * kappa .+ _iLams * _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  orcand = rand(rng, MvNormalCanon(_A_ms, _A), K)
  sp = sortperm(vcat(_beta[bidx], orcand[bidx,:]))
  selidx = K - sp[1]  +2 
  _beta .= orcand[:,sp[2:end] .== selidx][:]
  _bg .= _gamma .* _beta
end


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
function updatemu!(rng, _mu, _muvec, _beta, _tausq, j, pl, plidx, _mu_tau0, _mu_mu0)    
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

function updatetausq_robust!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres[plidx[q]] .= bl .- _mu[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    _tausq[q] = rand(rng, InverseGamma(a_tau, b_tau))
  end
end

function updatetausq_robust_or!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu; K=9)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  # including ordered over-relaxation
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres[plidx[q]] .= bl .- _mu[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    orcand = rand(rng, InverseGamma(a_tau, b_tau), K)
    sp = sortperm(vcat(_tausq[q], orcand))
    selidx = K - sp[1]  +2 
    _tausq[q] = orcand[sp[2:end] .== selidx][1]
  end
end

function update_iLams!(rng, _iLams, _tausq, plidx, j)
   #this is the appropriate covariance entry
   @inbounds for q in 1:j # loop over hierarchical groups
     _iLams.diag[plidx[q]] .= inv.(_tausq[q])   
   end
end

function updategprior!(_iLams, xtx, G)
  _iLams[2:end,2:end] .= xtx[2:end,2:end] ./ G
end

function calcmd!(munc, muint, Xint, _bg; offset=0.0)
  mul!(munc,  Xint[1], _bg[2:end])
  mul!(muint, Xint[2], _bg[2:end])
  munc .+= _bg[1] + offset
  muint .+= _bg[1] + offset
  cm1 = meanexpit(muint)
  cm0 = meanexpit(munc)
  md = cm1-cm0
  cm1, cm0, md
end

# gibbs variable selection (Kuo and Mallick)
function gvs_logit(y, X, Xint, pl, iter, burnin, rng; thin=1, chain=1,
               _mu_eta0 = 0., _tau_eta0 = 10.,                              # prior mean, sd of intercept
               _mu_mu0 = 0., _mu_tau0 = 1.0,                                # prior mean, sd of mu_l (beta means) (assumed constant over all beta)
               pa = [1.0], pb=[1.0],
               binit = NaN,
               offset = 0.0,
               ymean = 0, # rescaling coefficients
               yscale = 1.0 # rescaling coefficients
               )
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, yr, kappa, om, Om, mu
  ) =
  inits(rng,iter,binit, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, pa, pb, pl)
  nu = 1.0
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update gamma, pi for each group
    ####################
    updategamma!(rng, _gamma, _pi, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    updatepi!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    ####################
    # update beta
    ####################
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updateom!(rng, mu, om, Om, Xs, _bg)
    updatebeta!(rng, _beta, _bg, _gamma, om, Om, Xs, Xst, kappa, _iLams, _muvec, _A, _A_ms)
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, Xint, _bg; offset=offset)
      ####################
      # store sweep values
      ####################  
      _pi_store[i,:] = _pi
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store)
  nms = vcat(
       :chain, :iter,
       :m1, :m0, :md,
     [Symbol("beta" * "[$i]") for i in 0:p],
     [Symbol("gamma" * "[$i]") for i in 1:p],
     [Symbol("b*g" * "[$i]") for i in 1:p],
     [Symbol("pi" * "[$i]") for i in 1:j]
     )
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = convert(DataFrame, rr)
  rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]
end
gvs_logit(y,X,Xint,pl, iter, burnin;thin=1,chain=1, offset=0.0) = gvs_logit(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, offset=offset)
gvs_logit(y,X,Xint,pl, iter;thin=1,chain=1, offset=0.0)         = gvs_logit(y,X,Xint,pl, iter, 0;thin=thin,chain=chain, offset=offset)



# gibbs variable selection (Kuo and Mallick with hierarchical priors on model coefficeints)
function gvs_hlogit(y, X, Xint, pl, iter, burnin, rng; thin=1, chain=1,
               _mu_eta0 = 0.0, _tau_eta0 = 100.0,                              # prior mean, sd of intercept
               _mu_mu0 = [0.0], _mu_tau0 = [25.0],                # prior mean, scale of mu_l (beta group means)
               _tau_mu0 = [0.0], _tau_tau0 = [25.0],          # prior center, scale of half-t/half-cauchy (_tau_mu0 is unused)
               pa = [1.0], pb=[1.0],                              # prior beta(a,b) parameters for pi (group selection priors)
               binit = NaN,
               nu = 1.0,
               offset = 0.0,
               ymean = 0.0, # rescaling coefficients
               yscale = 1.0 # rescaling coefficients
               )
  # block sampler
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, yr, kappa, om, Om, mu
  ) =
  inits(rng,iter,binit, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, pa, pb, pl)
  # hierarchical specific
  _alpha = rand(rng)*2
  _v_store = zeros(iter, j)
  _mu_store = zeros(iter, j)
  _tau_store = zeros(iter, j)
  _alpha_store = zeros(iter)
  _mu = randn(rng,j)*2
  _tausq = rand(rng,j)*2
  _V = 1.0 ./ _tausq
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update gamma, pi for each group
    ####################
    updategamma!(rng, _gamma, _pi, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    #updatepi!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    updatepi_or!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb; K=21)
    ####################
    # update tau and mu from t/Cauchy distribution
    ####################
    updatebres!(bres, _beta, _mu, plidx, j)
    updatev!(rng, _V, _tausq, _tau_tau0, nu, j)
    updatemu!(rng, _mu, _muvec, _beta, _tausq, j, pl, plidx, _mu_tau0, _mu_mu0)
    #updatetausq_robust!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu)
    updatetausq_robust_or!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu; K=21)    
    update_iLams!(rng, _iLams, _tausq, plidx, j)
    
    ####################
    # update beta
    ####################
    # expand Lam, mu to pXp matrix
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updateom!(rng, mu, om, Om, Xs, _bg)
    updatebeta!(rng, _beta, _bg, _gamma, om, Om, Xs, Xst, kappa, _iLams, _muvec, _A, _A_ms)

    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, Xint, _bg; offset=offset)
      ####################
      # store sweep values
      ####################  
      _v_store[i,:] = _V
      _pi_store[i,:] = _pi
      _mu_store[i,:] = _mu .* _alpha
      _tau_store[i,:] = sqrt.(_tausq)
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr= hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store, 
                                _mu_store, _tau_store, _v_store)
  nms = vcat(
       :chain, :iter,
       :m1, :m0, :md,
     [Symbol("beta" * "[$i]") for i in 0:p],
     [Symbol("gamma" * "[$i]") for i in 1:p],
     [Symbol("b*g" * "[$i]") for i in 1:p],
     [Symbol("pi" * "[$i]") for i in 1:j],
     [Symbol("mu" * "[$i]") for i in 1:j],
     [Symbol("tau" * "[$i]") for i in 1:j],
     [Symbol("V" * "[$i]") for i in 1:j]
     )                              
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
   rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = convert(DataFrame, rr)
  rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]

end
gvs_hlogit(y,X,Xint,pl, iter, burnin;thin=1,chain=1,nu=1.0, offset=0.0) = gvs_hlogit(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, nu=nu, offset=offset)
gvs_hlogit(y,X,Xint,pl, iter;thin=1,chain=1,nu=1.0, offset=0.0)         = gvs_hlogit(y,X,Xint,pl, iter, 0;thin=thin,chain=chain, nu=nu, offset=offset)


# gibbs variable selection with Zellner's g-prior
# this suffers from getting "stuck" when pi is close to zero for one group
# could benefit from parameter expansion?
function gvs_zlogit(y, X, Xint, pl, iter, burnin, rng; thin=1, chain=1,
               _mu_eta0 = 0., _tau_eta0 = 10.,                              # prior mean, sd of intercept
               _mu_mu0 = 0., _mu_tau0 = 1.0,                            # not actually used here
               pa = [1.0, 1.0], pb=[1.0, 1.0],
               G=Float64(size(y,1)),                                                     # the g prior
               binit = NaN,
               offset = 0.0,
               ymean = 0, # rescaling coefficients
               yscale = 1.0 # rescaling coefficients
               )
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, yr, kappa, om, Om, mu
  ) =
  inits(rng,iter,binit, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, pa, pb, pl)
  # zellner specific 
  _iLams = diagm(vcat( _tau_eta0[1]^(-2), [ _mu_tau0[1]^(-2) for j in 1:p])) # this gets overwritten except row/col 1
  xtx = transpose(X)*X
  nu = 1.0
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update gamma, pi for each group
    ####################
    updategamma!(rng, _gamma, _pi, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    updatepi!(rng, _pi, _gamma, _beta, Nuj, res, y, X, j, plidx, pl, pa, pb)
    ####################
    # update g prior
    ####################
    updategprior!(_iLams, xtx, G)
    ####################
    # update beta
    ####################
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updateom!(rng, mu, om, Om, Xs, _bg)
    updatebeta!(rng, _beta, _bg, _gamma, om, Om, Xs, Xst, kappa, _iLams, _muvec, _A, _A_ms)
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, Xint, _bg; offset=offset)
      ####################
      # store sweep values
      ####################  
      _pi_store[i,:] = _pi
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store)
  nms = vcat(
       :chain, :iter, :m1, :m0, :md,
     [Symbol("beta" * "[$i]") for i in 0:p], [Symbol("gamma" * "[$i]") for i in 1:p],
     [Symbol("b*g" * "[$i]") for i in 1:p], [Symbol("pi" * "[$i]") for i in 1:j]
     )                              
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = convert(DataFrame, rr)
  rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]
end
gvs_zlogit(y,X,Xint,pl, iter, burnin;thin=1,chain=1,G=Float64(size(y,1)), offset=0.0) = gvs_zlogit(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain,G=G, offset=offset)
gvs_zlogit(y,X,Xint,pl, iter;thin=1,chain=1,G=Float64(size(y,1)), offset=0.0)         = gvs_zlogit(y,X,Xint,pl, iter, 0;thin=thin,chain=chain,G=G, offset=offset)

;