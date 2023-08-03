# this script contains the grid studies and tables created in response to the reviews, which can be found on the last page of the final paper. 

using GPDiffEq

using LinearAlgebra

using DataStructures

using Interpolations
using OnlineStats

# fixing random seed for reproducibility
using Random
Random.seed!(1234)

# defining the problem

ts = range(-4.25, 4.252; length=100)
f(x) = x * cos(x)

X = range(-3.0, 3.0; length=10)
σ_n = 0.1
y = f.(X) .+ σ_n * randn(length(X))

ker = SqExponentialKernel()
gp = GP(ker)
fx = gp(X, σ_n^2)

fp = posterior(fx, y)

# u0 = 1.0
u0 = Normal(1.0, 0.07)
tspan = (0.0, 3.0)
gpff = GPODEFunction(fp)
h = 0.0011

prob = GPODEProblem(gpff, u0, tspan)

xrange = range(X[1], X[end]; length=13)

sprob = SampledGPODEProblem(gpff, RegularGridSampling(xrange), u0, tspan)

# precompile
solve(prob, PULLEuler(); dt=0.2)
solve(prob, PULLEuler(ConstantBuffer(1)); dt=0.2)


######## end setup 

# high accuracy reference
xrange = range(X[1], X[end]; length=13)
sprob = SampledGPODEProblem(gpff, RegularGridSampling(xrange), u0, tspan)
oensprob = GPODESampledEnsembleProblem(sprob; nGPSamples=5000, nInValSamples=150)
saveat = tspan[1]:h:(tspan[2]+h)
oenssol = solve(oensprob, Tsit5(); saveat)

# step size study

hv = [0.01, 0.025, 0.05, 0.1, 0.2]
stepsize_grid_sols = OrderedDict()
stepsize_grid_times = OrderedDict()
for h in hv
    println("h: $h")
    stime = @elapsed pesol = solve(prob, PULLEuler(); dt=h)
    stepsize_grid_sols[h] = pesol
    stepsize_grid_times[h] = stime
end
println(values(stepsize_grid_times))
no_buffer_sols = OrderedDict()
for h in hv
    println("h: $h")
    stime = @elapsed pesol = solve(prob, PULLEuler(ConstantBuffer(1)); dt=h)
    no_buffer_sols[h] = pesol
    stepsize_grid_times[h] = stime
end
println(values(stepsize_grid_times))

# # plot
# begin
#     x = oenssol.t
#     my = mean.(oenssol.u)
#     vy = std.(oenssol.u)
#     mitp = linear_interpolation(x, my)
#     vitp = linear_interpolation(x, vy)

#     # p = plot(; layout=(2, 1), size=(700, 800))
#     # plot!(p; inset=(1, bbox(0.4, 0.5, 0.55, 0.4)), subplot=3)
#     # plot!(p; inset=(2, bbox(0.4, 0.5, 0.55, 0.4)), subplot=4)
#     merv = []
#     nbmerv = []
#     verrv = []
#     nbverrv = []
#     for (key, _) in stepsize_grid_sols
#         # mean
#         t = stepsize_grid_sols[key].t[1:end-1]
#         m = mean.(stepsize_grid_sols[key].u)[1:end-1]
#         # plot!(p, t, m; subplot=1, label="$key")
#         merr = abs.(m .- mitp(t))
#         # plot!(p, t, merr; subplot=3, label="")
#         push!(merv, mean(merr))
#         nbm = mean.(no_buffer_sols[key].u)[1:end-1]
#         nbmerr = abs.(nbm .- mitp(t))
#         push!(nbmerv, mean(nbmerr))
#         #var
#         v = std.(stepsize_grid_sols[key].u)[1:end-1]
#         # plot!(p, t, v; subplot=2, label="")
#         verr = abs.(v .- vitp(t))
#         mverr = mean(verr)
#         push!(verrv, mverr)
#         println("h: $key, verr: $mverr")
#         nbv = std.(no_buffer_sols[key].u)[1:end-1]
#         nbverr = abs.(nbv .- vitp(t))
#         push!(nbverrv, mean(nbverr))
#         # plot!(p, t, verr; subplot=4, label="")
#     end
#     println(hv)
#     println("mean")
#     println(merv)
#     println(nbmerv)
#     println("var")
#     println(verrv)
#     println(nbverrv)
#     # p
# end

# [0.01, 0.025, 0.05, 0.1, 0.2]
# mean
# Any[0.007756130868938289, 0.008798296938200452, 0.010532449568038038, 0.013989747720691866, 0.020855866789928835]
# Any[0.007756130868938289, 0.008798296938200452, 0.010532449568038038, 0.013989747720691866, 0.020855866789928835]
# var
# Any[0.0006267054464432552, 0.000645993546758104, 0.0007237559818879855, 0.0009929113220711124, 0.001788747891401575]
# Any[0.0252028764150362, 0.02239500821042304, 0.01894121747825646, 0.013978532498996039, 0.00802929744188212]


# #
xrange = range(X[1], X[end]; length=13)

sprob = SampledGPODEProblem(gpff, RegularGridSampling(xrange), u0, tspan)

h = 0.0011

ssol = solve(sprob, Euler(); dt=h)


#########################
# no input noise
u0 = 1.0
println("no input noise")

# let's run some experiments
function gridsolve_comp!(multisample_mean, multisample_cov, elapsed_times, ntraj, ngridsample; nbtch=1000, nrepeats=5)
    println("ntraj: $ntraj, ngridsample: $ngridsample")
    multisample_mean[ntraj, ngridsample] = [FitNormal() for _ in 1:length(ssol.u)]
    multisample_cov[ntraj, ngridsample] = [FitNormal() for _ in 1:length(ssol.u)]
    elapsed_times[ntraj, ngridsample] = FitNormal()

    xr = range(xrange[1], xrange[end]; length=ngridsample)

    sprob = SampledGPODEProblem(gpff, RegularGridSampling(xr), u0, tspan)

    for _ in 1:nrepeats
        oensprob1 = GPODESampledEnsembleProblem(sprob; nGPSamples=ntraj)
        t = @elapsed oenssol1 = solve(oensprob1, Euler(); dt=h)

        fit!(elapsed_times[ntraj, ngridsample], t)

        sm = mean.(oenssol1.u)
        sc = std.(oenssol1.u)
        fit!.(multisample_mean[ntraj, ngridsample], sm)
        fit!.(multisample_cov[ntraj, ngridsample], sc)
    end
end

multisample_mean3 = Dict()
multisample_cov3 = Dict()
elapsed_times3 = Dict()

compsol = solve(prob, PULLEuler(); dt=h);

using Measurements

nsamplv = [4, 6, 8, 10]
ntrajv = 100 * [5, 10, 20]
for nsampl in nsamplv, ntraj in ntrajv
    gridsolve_comp!(multisample_mean3, multisample_cov3, elapsed_times3, ntraj, nsampl)
end

begin
    meanerrmatrix = zeros(Measurement, length(nsamplv) + 1, length(ntrajv) + 1)
    meanerrmatrix[:, 1] = vcat(0, nsamplv)
    meanerrmatrix[1, :] = vcat(0, ntrajv)
    coverrmatrix = deepcopy(meanerrmatrix)
    timematrix = deepcopy(meanerrmatrix)
    for (i, nsampl) in enumerate(nsamplv), (j, ntraj) in enumerate(ntrajv)
        key = (ntraj, nsampl)
        mtmp = [mean(o) .± sqrt.(var(o)) for o in multisample_mean3[key]]
        meanerrmatrix[i+1, j+1] = mean(norm.(mtmp - mean.(compsol.u)))
        scov = mean.(multisample_cov3[key])
        scovvar = std.(multisample_cov3[key])
        vtmp = [scov[i] .± scovvar[i] for i in eachindex(scov)]
        coverrmatrix[i+1, j+1] = mean(norm.(vtmp - std.(compsol.u)))
        timematrix[i+1, j+1] = mean(elapsed_times3[key]) .± sqrt(var(elapsed_times3[key]))
    end
    display("mean error")
    display(meanerrmatrix)
    display("coverr")
    display(coverrmatrix)
    display("time")
    display(timematrix)

end

# "mean error"
# 5×4 Matrix{Measurement}:
#   0.0±0.0     500.0±0.0       1000.0±0.0       2000.0±0.0
#   4.0±0.0  0.013979±3.9e-5  0.014129±2.9e-5   0.01407±1.6e-5
#   6.0±0.0  0.112884±3.5e-5  0.113081±1.6e-5  0.112745±2.9e-5
#   8.0±0.0  0.013206±2.1e-5  0.012172±3.8e-5  0.012526±1.0e-5
#  10.0±0.0  0.004932±3.4e-5  0.004506±2.3e-5   0.00447±2.3e-5
# "coverr"
# 5×4 Matrix{Measurement}:
#   0.0±0.0     500.0±0.0       1000.0±0.0        2000.0±0.0
#   4.0±0.0  0.014495±2.9e-5  0.014841±1.7e-5  0.0150765±2.7e-6
#   6.0±0.0   0.02044±3.2e-5  0.020946±3.2e-5   0.021042±1.2e-5
#   8.0±0.0  0.015914±2.9e-5  0.016014±1.9e-5  0.0157493±5.8e-6
#  10.0±0.0  0.012436±1.2e-5  0.012071±1.3e-5  0.0122497±8.5e-6
# "time"
# 5×4 Matrix{Measurement}:
#   0.0±0.0    500.0±0.0      1000.0±0.0    2000.0±0.0
#   4.0±0.0     0.15±0.21      0.093±0.042   0.186±0.059
#   6.0±0.0    0.065±0.05      0.113±0.058   0.196±0.058
#   8.0±0.0    0.061±0.046     0.093±0.047   0.212±0.05
#  10.0±0.0  0.04023±0.00069   0.094±0.048   0.175±0.053


#########################################
# with input noise

u0 = Normal(1.0, 0.07)

function in_gridsolve_comp!(multisample_mean, multisample_cov, elapsed_times, nGPSamples, ngridsample, nInValSamples=100, nrepeats=5)
    println("ntraj: $nGPSamples, ngridsample: $ngridsample, nInValSamples: $nInValSamples")
    multisample_mean[nGPSamples, ngridsample, nInValSamples] = [FitNormal() for _ in 1:length(ssol.u)]
    multisample_cov[nGPSamples, ngridsample, nInValSamples] = [FitNormal() for _ in 1:length(ssol.u)]
    elapsed_times[nGPSamples, ngridsample, nInValSamples] = FitNormal()

    xr = range(xrange[1], xrange[end]; length=ngridsample)

    sprob = SampledGPODEProblem(gpff, RegularGridSampling(xr), u0, tspan)

    for _ in 1:nrepeats
        # oensprob1 = OnlineEnsembleProblem(sprob, TrajOnNormal(); nInValSamples, nGPSamples)
        oensprob1 = GPODESampledEnsembleProblem(sprob; nGPSamples, nInValSamples)
        t = @elapsed oenssol1 = solve(oensprob1, Euler(); dt=h)

        fit!(elapsed_times[nGPSamples, ngridsample, nInValSamples], t)

        sm = mean.(oenssol1.u)
        sc = std.(oenssol1.u)
        fit!.(multisample_mean[nGPSamples, ngridsample, nInValSamples], sm)
        fit!.(multisample_cov[nGPSamples, ngridsample, nInValSamples], sc)
    end
end


multisample_mean3 = Dict()
multisample_cov3 = Dict()
elapsed_times3 = Dict()

compsol = solve(prob, PULLEuler(); dt=h);

nInSamples = 10 * [5, 10, 15]
ntrajv = 100 * [5, 10, 20]
nGridS = 11
for nInSamples in nInSamples, ntraj in ntrajv
    in_gridsolve_comp!(multisample_mean3, multisample_cov3, elapsed_times3, ntraj, nGridS, nInSamples)
end

begin
    meanerrmatrix = zeros(Measurement, length(nInSamples), length(ntrajv))
    meanerrmatrix = zeros(Measurement, length(nInSamples) + 1, length(ntrajv) + 1)
    meanerrmatrix[:, 1] = vcat(0, nInSamples)
    meanerrmatrix[1, :] = vcat(0, ntrajv)
    coverrmatrix = deepcopy(meanerrmatrix)
    timematrix = deepcopy(meanerrmatrix)
    for (i, nInSample) in enumerate(nInSamples), (j, ntraj) in enumerate(ntrajv)
        key = (ntraj, nGridS, nInSample)
        mtmp = [mean(o) .± sqrt.(var(o)) for o in multisample_mean3[key]]
        meanerrmatrix[i+1, j+1] = mean(norm.(mtmp - mean.(compsol.u)))
        scov = mean.(multisample_cov3[key])
        scovvar = var.(multisample_cov3[key])
        vtmp = [scov[i] .± sqrt.(scovvar[i]) for i in eachindex(scov)]
        coverrmatrix[i+1, j+1] = mean(norm.(vtmp - std.(compsol.u)))
        timematrix[i+1, j+1] = mean(elapsed_times3[key]) .± sqrt(var(elapsed_times3[key]))
    end
    display(meanerrmatrix)
    display(coverrmatrix)
    display(timematrix)
end

# 4×4 Matrix{Measurement}:
#    0.0±0.0     500.0±0.0       1000.0±0.0        2000.0±0.0
#   50.0±0.0  0.004766±4.7e-5   0.00542±0.00011  0.006069±9.6e-5
#  100.0±0.0    0.0081±7.5e-5  0.006247±4.8e-5   0.005118±6.0e-5
#  150.0±0.0  0.006058±9.0e-5  0.008084±5.4e-5   0.005809±7.8e-5
# 4×4 Matrix{Measurement}:
#    0.0±0.0     500.0±0.0       1000.0±0.0       2000.0±0.0
#   50.0±0.0  0.002053±4.5e-5    0.0014±5.7e-5  0.000441±2.8e-5
#  100.0±0.0  0.000548±5.5e-5  0.001238±3.0e-5   0.00051±4.7e-5
#  150.0±0.0  0.000677±2.8e-5  0.001446±3.8e-5  0.000387±3.4e-5
# 4×4 Matrix{Measurement}:
#    0.0±0.0  500.0±0.0    1000.0±0.0    2000.0±0.0
#   50.0±0.0   1.83±0.14    3.603±0.03    7.174±0.091
#  100.0±0.0  3.616±0.056   7.174±0.087   14.44±0.11
#  150.0±0.0  5.324±0.091   10.82±0.13   21.652±0.083
