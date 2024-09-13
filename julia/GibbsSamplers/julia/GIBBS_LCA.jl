# GIBBS sampling for latent class analysis based on a series of binary indicators
# Julia translation of model here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364555/

using Distributions, Random, Combinatorics, LinearAlgebra

function pi_i(classvec, u)
    classvec
end

"""
# posterior (marginal) probability of being in a class
p(C)

classvecs = [
    1 0 0;
    0 1 0;
    0 1 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    0 0 1 
]
pi = ones(3)
u = ones(3)
sample_pi!(pi, classvecs, u)
"""
function sample_pi!(pi, classvecs, u)
    #pli = [classvecs[i,:] for i in 1:size(classvecs,1)]
    pl = sum(classvecs, dims=1)[:] + u
    pi .= rand(Dirichlet(pl))
    nothing
end


"""
Posterior probability vector for measured variables, given class
P(Y | C)

Each Y follows a beta distribution within each class
```julia
classvecs = [
    1 0 0;
    0 1 0;
    0 1 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    0 0 1 
]
n,Q = size(classvecs, 1), 5 # dimensions of Y
K = size(classvecs, 2) # class index
y = Int.(rand(n,Q) .> 0.5)
yi = y[1,:]
ajk = ones(K,Q)
bjk = ones(K,Q)
pjk = ones(K,Q)
sample_pjk!(pjk, K, y, classvecs, ajk, bjk)
pjk
```
"""
function sample_pjk!(rng, pjk, K, y, classvecs, ajk, bjk)
    for c in 1:K
        a = classvecs[:,c:c]'*y .+ ajk[c:c,:]# .- 1
        b = classvecs[:,c:c]'*(1.0 .- y) .+ bjk[c:c,:]# .- 1
        pjk[c,:] .= rand.(rng, Beta.(a[:],b[:]))
    end
    nothing
end

sample_pjk!(pjk, J, y, classvecs, ajk, bjk) = sample_pjk!(MersenneTwister(), pjk, J, y, classvecs, ajk, bjk)


function classvecs_i(K, pi_, yi, pjk)
    # p(c|y) = p(c) * p(y|c) / p(y)
    cv = zeros(K)
    for c in 1:K
      lpj = yi .* log.(pjk[c,:]) .+  (1.0 .- yi) .* log.(1.0 .- pjk[c,:])
      cv[c] += exp(log(pi_[c])+sum(lpj)) # cv[j] = p(j) * p(y|j)
    end
    # pi_' * pjk
    cv ./ sum(cv)
end

"""
Posterior draws of classes, given the outcome

P(C|Y) = (P(C)*P(Y|C))/(SUM P(Y|C))


```julia
pi_ = [0.2792204, 0.1283047, 0.07102063, 0.5214543]
pjk=[
0.3072238 0.5514973 0.3462326 0.03370501 0.5089951 0.3452769;
0.5733021 0.6741166 0.5370192 0.60961181 0.4667233 0.3464321;
0.9712629 0.2293985 0.5746601 0.77493576 0.4271499 0.7361765;
0.8057957 0.1349706 0.1792205 0.65392391 0.6252457 0.4477678
]

classvecs_i(C, pi_, y[1,:], pjk)
```
"""
function sample_classvecs!(rng, classvecs, pi_, y, pjk, K)
    # future: optionally output the probabilities
    for i in 1:size(y,1)
        classvecs[i,:] .= rand(rng, Multinomial(1, classvecs_i(K, pi_, y[i,:], pjk)))
    end
    nothing
end
sample_classvecs!(classvecs, pi, y, pjk, K) = sample_classvecs!(MersenneTwister(), classvecs, pi, y, pjk, K)

function labswitch_classmove(cvi, cvcur, perms)
    matchmat = cvi' * cvcur #.= classvecs[:,sp]
    whichp = argmax([sum(diag(matchmat[:,p])) for p in perms])
    perms[whichp]
end



"""
Convenience function to relabel a vector

```julia
labs = post["labels"]
parm = post["pi"]
mean(parm, dims=1)
relabelpost!(parm, labs, 1)
mean(post["pi"], dims=1)
mean(parm, dims=1)
```
"""
function relabelpost!(parm, labs)
  for row in 1:size(parm,1)
    tmp = parm[row,labs[row,:]]
    parm[row,:] .= tmp
  end
  nothing
end


"""
# fully worked example
#using Add Health data linked in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364555/
```julia
using CSV, DataFrames
nmorig = [:AID,:H1DS3,:H1DS15,:H1DS2,:H1DS4,:H1DS13,:H1DS14]
nmnew =  [:AID,:lied,:publicly,:damaged,:shoplift,:stole50,:grpfight]
addhealth = CSV.read(expanduser("~/Documents/DataSets/PublicData/AddHealth/ICPSR_21600/DS0001/21600-0001-Data.tsv"), DataFrame, delim="\t", select=nmorig)[:,nmorig]
rename!(addhealth, nmnew)

describe(addhealth)


y = Matrix(Int.(addhealth[:,2:end] .> 0))
```
Note: this example recodes everything as binary (this does not seem correct, but it fits with the paper example)

Based on the questionnaire for the H1DS2/"damaged" variable, the original values are

  H1DS2: "In the past 12 months, how often did you deliberately damage property that didn’t belong to you?"
  
    0 = never
    1 = 1 or 2 times
    2 = 3 or 4 times
    3 = 5 or more times
    6 = refused
    8 = don’t know
    9 = not applicable

There is no justification to group "not applicable" with "3 or 4 times," for example. However, for comparability with the paper, we are keeping this apparent recoding. 

```julia
test = gibbs_lca(y,4;iters=10, burnin=1)

post = gibbs_lca(y,4;iters=12000, burnin=2500)

extrema(post["labels"], dims=1) # check for label switching
# example of re-labeling to address label switching
#relabelpost!(post["pi"], labs, 1)
# mean(post["pi"], dims=1)
# no need to address label switching

mean(post["pi"], dims=1)
# stability of classes 
# pjk describes probability of being in a class, given Y_k; here we estimate that probability given the vector Y
classmode = [mode(post["class"][:,i]) for i in 1:size(y,1)] # modal class for each id
classstable = reduce(hcat, [post["class"][iter,:] .== classmode for iter in 1:size(post["class"],1)])
probclass = mean(classstable, dims=2)[:] # P(C|Y) = pjk[c,k]*pi[c] / (sum(pjk*pi))


describe(probclass)

addhealth.modalclass = classmode
addhealth[findall(probclass .< 0.5),:]




# re-creating posterior mean table
DataFrame(hcat(vcat(:pi_j,nmnew[2:end]), round.(vcat(mean(post["pi"], dims=1),
     reshape(mean(post["pjk"], dims=1), (4,6))'), digits=3)), [:_, :c1, :c2, :c3, :c4])

# Trace plot example
using Plots
plot(post["pi"][100:end,1])

plot(post["pjk"][100:end,7])
```
"""
function gibbs_lca(rng, y,K;
    # MCMC parameters
    iters=100, burnin=0, 
    # priors
    u = ones(K),     # Dirichlet concentration parameters for pi
    ajk = ones(K,size(y,2)), # beta priors for pjk (parameter 1)
    bjk = ones(K,size(y,2))  # beta priors for pjk (parameter 2)
    )
    # pre-reqs
    N,Q = size(y)
    perms = [p for p in permutations(collect(1:K))] # label switching
    nperms = factorial(K)
    # initialize parameters
    classlabs = sample(rng, 1:K, N)
    classvecs = zeros(Int64,N,K)
    classvecs_init = zeros(Int64,N,K)
    for i in 1:N
      classvecs[i,classlabs[i]] = 1
    end
    pi_ = rand(rng, K)
    pjk = rand(rng, K,Q)
    # storage containers across the chain
    pi_post = Array{Float64,2}(undef, iters-burnin, K)
    pjk_post = Array{Float64,2}(undef, iters-burnin, K*Q)
    class_post = Array{Float64,2}(undef, iters-burnin, N)
    reord_post = Array{Int64,2}(undef, iters-burnin, K)
    if burnin == 0
      pi_post[1,:] = pi_
      pjk_post[1,:] = pjk[:] # down in columns: p[1,1], p[2,1], ... etc.
      class_post[1,:] = [c[2] for c in argmax(classvecs, dims=2)] # down in columns: classvec[1,1], classvec[2,1], ... etc.
    end
    # iterate
    for sweep in 2:iters
        # sampling parameters
        sample_classvecs!(MersenneTwister(), classvecs, pi_, y, pjk, K)
        sample_pi!(pi_, classvecs, u)
        sample_pjk!(MersenneTwister(), pjk, K, y, classvecs, ajk, bjk)
        # end sampling parameters
        if (sweep == 2 && burnin==0) || (sweep == burnin)
            classvecs_init .= deepcopy(classvecs)
        end
        if sweep > burnin
          # label switching (which permutation of the confusion matrix columns gives the largest diagonal?)
          ord = labswitch_classmove(classvecs_init, classvecs, perms)
          pi_post[sweep-burnin,:] .= pi_
          pjk_post[sweep-burnin,:] .= pjk[:] # down in columns: p[1,1], p[2,1], ... etc.
          class_post[sweep-burnin,:] .= [c[2] for c in argmax(classvecs, dims=2)] # down in columns: classvec[1,1], classvec[2,1], ... etc.    
          reord_post[sweep-burnin,:] .= ord # down in columns: classvec[1,1], classvec[2,1], ... etc.    
        end
    end
    Dict(
        "pi" => pi_post, 
        "pjk" => pjk_post, 
        "class" => class_post,
        "labels" => reord_post
    )
end
gibbs_lca(y,K;kwargs...) = gibbs_lca(MersenneTwister(), y,K;kwargs...)

