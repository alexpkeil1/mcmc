using DataFrames
using Statistics
using StatsBase
using MCMCDiagnosticTools




#struct GibbsSampler{I <: Int64, B <: Int64, D <: DataFrame} <: DataFrame
#    iter::I
#    burnin::B
#    dat::D
#    @doc """
#    $(SIGNATURES)
#
#    """
#    function GibbsSampler(iter::I, burnin::B, dat::D) where {I <: Int64, B <: Int64, D <: DataFrame}
#    end
#end

function flat(arr::Array)
  res = mapreduce(x -> isa(x, Array) ? flat(x) : x, append!, arr,init=[])
  convert(Array{Float64}, res)
end


function summarygibbs(results::T; parms=nothing) where {T<:DataFrame}
 sets, means, medians, pl, pu, stds, ac1, ac5, ess, lens = Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[]
 nm = names(results)
 if length(parms)>0
  keepidx = findall([n ∈ parms for n in nm])
 else
  keepidx = 1:size(results, 2)
 end

 for i in keepidx
   col = results[:,i]
   means = vcat(means, mean(col))
   medians = vcat(medians, median(col))
   pl = vcat(pl, quantile(col, 0.025)[1])
   pu = vcat(pu, quantile(col,  0.975)[1])
   stds = vcat(stds, std(col))
   ac = autocor(col, [1,5])
   ac1 = vcat(ac1, ac[1])
   ac5 = vcat(ac5, ac[2])
   ess = vcat(ess, MCMCDiagnosticTools.ess(col))
   lens = vcat(lens, length(col))
 end
 nms =  [:nm, :mean, :std, :median, :lower2_5, :upper97_5, :ess, :autocor_1, :autocor_5, :length]
 res = DataFrame(hcat(nm[keepidx], means, stds, medians, pl, pu, ess, ac1, ac5, lens), nms)
 return res
end

function summarygibbs(results::Dict{T,W}; parms = []) where {T<:Any, W<:DataFrame}
 sets, means, medians, pl, pu, stds, ac1, ac5, ess, rhat, lens = Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[]
 # may err with 
 dkeys = collect(keys(results))
 nm = names(results[dkeys[1]])
 if length(parms)>0
    keepidx = findall([n ∈ parms for n in nm])
 else
    keepidx = 1:size(results[dkeys[1]], 2)
 end
 for i in keepidx # number of parameters
   col = flat([vcat(r[2][:,i]) for r in results])
   means = vcat(means, mean(col))
   medians = vcat(medians, median(col))
   pl = vcat(pl, quantile(col, 0.025)[1])
   pu = vcat(pu, quantile(col,  0.975)[1])
   stds = vcat(stds, std(col))
   ac = autocor(col, [1,5])
   ac1 = vcat(ac1, ac[1])
   ac5 = vcat(ac5, ac[2])
   ess = vcat(ess, MCMCDiagnosticTools.ess(col))
   #rhat = vcat(rhat,  potential_scale_reduction([vcat(r[2][:,i]) for r in results]...))
   rht = try
    #gelmandiag(hcat([vcat(r[2][:,i]) for r in results]...)[:,:,:]).psrf[1]
    MCMCDiagnosticTools.rhat(hcat([vcat(r[2][:,i]) for r in results]...)[:,:,:])
   catch e
    NaN
   end
   rhat = vcat(rhat,  rht) # draws, chains, parameters dimension
   lens = vcat(lens, length(col))
 end
 nms =  [:nm, :mean, :std, :median, :lower2_5, :upper97_5, :ess, :rhat, :autocor_1, :autocor_5, :length]
 res = DataFrame(hcat(nm[keepidx], means, stds, medians, pl, pu, ess, rhat, ac1, ac5, lens), nms)
 return res
end


function summarygibbs(results::Array{<:T};kwargs...) where {T<:DataFrame}
  rdict = Dict{String, DataFrame}()
  for (nm, r) in enumerate(results)
    push!(rdict, "$nm" => r)
  end
  summarygibbs(rdict;kwargs...)
 end
 ;