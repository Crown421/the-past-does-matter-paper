using GPDiffEq
using LinearAlgebra

using Plots
pgfplotsx()

using Random
Random.seed!(1234)

# defining the problem

ts = range(-4.25, 4.252; length=100)
f(x) = x * cos(x)

X = range(-3.0, 3.0; length=10)
σ_n = 0.2
y = f.(X) .+ σ_n * randn(length(X))

ker = SqExponentialKernel()
gp = GP(ker)
fx = gp(X, σ_n^2)

fp = posterior(fx, y)

u0 = 1.0
tspan = (0.0, 3.0)
gpff = GPODEFunction(fp)
h = 0.0011

prob = GPODEProblem(gpff, u0, tspan)


pxrange = range(X[1], X[end]; length=30)

# first plot in the poster to show distribution over ODE models
begin
    Random.seed!(12345)

    p = plot(size=(800, 520) .* 0.45)
    plot!(p[1], pxrange, mean(gpff.gp, pxrange); ribbons=3 * sqrt.(var(gpff.gp, pxrange)),
        label="GP", linewidth=3, color=:mediumseagreen,
        ylabel="f(x)", xlabel="x", ylim=[-3.6, 4.2], legend=:topright)
    scatter!(p[1], X, y; label="data", color=:black, markersize=3)
    p
end

savefig(p, "~/tmp/posterplots/intro_plot.pdf")

# extended alternative first plot, that also contains trajectories
begin
    Random.seed!(12345)

    p = plot(layout=(2, 1), size=(800, 750) .* 0.45)
    plot!(p[1], pxrange, mean(gpff.gp, pxrange); ribbons=3 * sqrt.(var(gpff.gp, pxrange)),
        label="GP mean ± 2σ ", linewidth=3, color=:mediumseagreen,
        ylabel="f(x)", xlabel="x", ylim=[-3.6, 4.5])

    greys = Plots.Colors.colormap("Grays", 9)[3:end-1]
    lbls = fill("", 6)
    lbls[3] = "GP sample trajectories"
    for i in 2:6
        sprob = SampledGPODEProblem(gpff, RegularGridSampling(xrange), u0, tspan)
        sy = sprob.prob.f.(pxrange, Ref(sprob.prob.p), 0.0)
        plot!(p[1], pxrange, sy, color=greys[i], linewidth=1.5, label="")

        sol = solve(sprob, Tsit5())
        plot!(p[2], sol, color=greys[i], linewidth=1.5, label=lbls[i], legend=:bottomright)
    end
    plot!(p[2], xlabel="t", ylabel="x(t)")

    scatter!(p[1], X, y; label="observations", color=:black, markersize=3)

    p
end

savefig(p, "~/tmp/posterplots/intro_plot.pdf")

