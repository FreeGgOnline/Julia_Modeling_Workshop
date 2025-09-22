### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 885f5685-5239-42aa-942c-2b9bdf7df221
# SciML Tools
import OrdinaryDiffEq as ODE

# ╔═╡ f5c9ea83-6dba-4848-8340-1a8c772bd942
import ModelingToolkit as MTK

# ╔═╡ e13c7d4f-4cdb-43d5-9f77-33401cb501e3
import DataDrivenDiffEq

# ╔═╡ c7193526-d32f-4f2a-95aa-09e98b88b30c
import SciMLSensitivity as SMS

# ╔═╡ 36985121-9fe5-4961-9ec2-3f5ee57cfaa2
import DataDrivenSparse

# ╔═╡ 45dc0462-d63c-11ef-0f62-e5d029618cc2
import Optimization as OPT

# ╔═╡ 6d29e44b-3d70-4700-ad53-9e75df7ced0d
import OptimizationOptimisers

# ╔═╡ 51c67a4d-f49a-4bb5-a1c2-361ea0e8f20f
import OptimizationOptimJL

# ╔═╡ ca453691-227c-44f1-8b48-8fb42c42d482
import LineSearches

# ╔═╡ 34efa390-0d6d-48dd-b08b-0c1bc391ffc3
# Standard Libraries
import LinearAlgebra

# ╔═╡ 448dd6ac-0d7f-4c1f-a9ce-3f9a586951c2
import Statistics

# ╔═╡ 760b16ea-410f-4f46-ae44-f859de9b75e4
# External Libraries
import ComponentArrays

# ╔═╡ abc12345-6789-def0-1234-567890abcdef
import Lux

# ╔═╡ bcd23456-789a-ef01-2345-6789abcdef01
import Zygote

# ╔═╡ cde34567-89ab-f012-3456-789abcdef012
import Plots

# ╔═╡ def45678-9abc-0123-4567-89abcdef0123
import StableRNGs

# ╔═╡ ef456789-abcd-1234-5678-9abcdef01234
Plots.gr()

# ╔═╡ a1b2c3d4-e5f6-7890-abcd-ef1234567890
md"""
# Automatically Discover Missing Physics by Embedding Machine Learning into Differential Equations

In this notebook, we'll demonstrate how Universal Differential Equations (UDE) can help discover missing physics in mathematical models using the Lotka-Volterra predator-prey equations.

## The Lotka-Volterra System

The Lotka-Volterra equations describe predator-prey dynamics:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta xy \\
\frac{dy}{dt} &= \gamma xy - \delta y
\end{align}
```

where:
- `x` represents prey population
- `y` represents predator population
- `α` is prey growth rate
- `β` is predation rate affecting prey
- `γ` is predator efficiency in converting prey
- `δ` is predator death rate
"""

# ╔═╡ b2c3d4e5-f6a7-8901-bcde-f12345678901
md"""
## Step 1: Generate Training Data

First, we'll generate synthetic data from the true Lotka-Volterra system with added noise.
"""

# ╔═╡ c3d4e5f6-a7b8-9012-cdef-123456789012
# Set a random seed for reproducible behaviour
rng = StableRNGs.StableRNG(1111)

# ╔═╡ f5678901-bcde-2345-6789-abcdef012345
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# ╔═╡ a6789012-cdef-3456-789a-bcdef0123456
# Define the experimental parameter
tspan = (0.0, 5.0)

# ╔═╡ b789a123-def0-4567-89ab-cdef01234567
u0 = 5.0f0 * rand(rng, 2)

# ╔═╡ c89ab234-ef01-5678-9abc-def012345678
p_ = [1.3, 0.9, 0.8, 1.8]

# ╔═╡ d9abc345-f012-6789-abcd-ef0123456789
prob = ODE.ODEProblem(lotka!, u0, tspan, p_)

# ╔═╡ eabcd456-0123-789a-bcde-f01234567890
solution = ODE.solve(prob, ODE.Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# ╔═╡ d4e5f6a7-b890-1234-def0-234567890123
# Add noise in terms of the mean
X = Array(solution)

# ╔═╡ fabcd567-1234-89ab-cdef-012345678901
t = solution.t

# ╔═╡ aabcd678-2345-9abc-def0-123456789012
x̄ = Statistics.mean(X, dims = 2)

# ╔═╡ bbcde789-3456-abcd-ef01-234567890123
noise_magnitude = 5e-3

# ╔═╡ ccdef890-4567-bcde-f012-345678901234
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

# ╔═╡ ddef0901-5678-cdef-0123-456789012345
plot_data = Plots.plot(solution, alpha = 0.75, color = :black, label = ["True Prey" "True Predator"])

# ╔═╡ eef01a12-6789-def0-1234-567890123456
Plots.scatter!(plot_data, t, Xₙ', color = :red, label = ["Noisy Prey" "Noisy Predator"])

# ╔═╡ e5f6a7b8-9012-3456-ef01-345678901234
md"""
## Step 2: Define the Hybrid Model (UDE)

Now we'll create a hybrid model that assumes we only know part of the dynamics. We'll use a neural network to learn the missing interaction terms.

We assume we know:
- Prey grows exponentially: `αx`
- Predators die exponentially: `-δy`

But we don't know the interaction terms.
"""

# ╔═╡ f6a7b890-1234-5678-f012-456789012345
# Define the network
rbf(x) = exp.(-(x .^ 2))

# ╔═╡ fbcde567-1234-89ab-cdef-012345678901
# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(2, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2)
)

# ╔═╡ acdef678-2345-9abc-def0-123456789012
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)

# ╔═╡ bdef0789-3456-abcd-ef01-234567890123
# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# ╔═╡ cef01890-4567-bcde-f012-345678901234
# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# ╔═╡ df012901-5678-cdef-0123-456789012345
# Define the UDE problem
prob_nn = ODE.ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# ╔═╡ b8901234-5678-9abc-1234-678901234567
md"""
## Step 3: Define Loss Function and Training

We'll train the neural network to minimize the difference between predicted and observed trajectories.
"""

# ╔═╡ c9012345-6789-abcd-2345-789012345678
function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = ODE.remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(ODE.solve(_prob, ODE.Vern7(), saveat = T,
                     abstol = 1e-6, reltol = 1e-6,
                     sensealg = SMS.QuadratureAdjoint(autojacvec = SMS.ReverseDiffVJP(true))))
end

# ╔═╡ b3456d45-9abc-0123-4567-890123456789
function loss(θ)
    X̂ = predict(θ)
    sum(abs2, Xₙ .- X̂)
end

# ╔═╡ c4567e56-abcd-1234-5678-901234567890
losses = Float64[]

# ╔═╡ d5678f67-bcde-2345-6789-012345678901
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# ╔═╡ d0123456-789a-bcde-3456-890123456789
md"""
## Step 4: Train the UDE Model

We'll use a two-stage optimization approach:
1. ADAM for initial rough optimization
2. LBFGS for fine-tuning
"""

# ╔═╡ e1234567-89ab-cdef-4567-901234567890
adtype = OPT.AutoZygote()

# ╔═╡ e6789078-cdef-3456-789a-123456789012
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)

# ╔═╡ f7890189-def0-4567-89ab-234567890123
optprob = OPT.OptimizationProblem(optf, ComponentArrays.ComponentVector(p))

# ╔═╡ a890129a-ef01-5678-9abc-345678901234
res1 = OPT.solve(optprob, OptimizationOptimisers.ADAM(), callback = callback, maxiters = 5000)

# ╔═╡ b90123ab-f012-6789-abcd-456789012345
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# ╔═╡ f2345678-9abc-def0-5678-012345678901
# Second stage with LBFGS
optprob2 = OPT.OptimizationProblem(optf, res1.u)

# ╔═╡ ca0123bc-0123-789a-bcde-567890123456
res2 = OPT.solve(optprob2, OptimizationOptimJL.Optim.LBFGS(linesearch = LineSearches.BackTracking()),
                callback = callback, maxiters = 1000)

# ╔═╡ db1234cd-1234-89ab-cdef-678901234567
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# ╔═╡ ec2345de-2345-9abc-def0-789012345678
# Rename the best candidate
p_trained = res2.u

# ╔═╡ a3456789-bcde-f012-6789-123456789012
# Plot training loss
plot_loss = Plots.plot(losses, xlabel = "Iterations", ylabel = "Loss",
                       yscale = :log10, title = "Training Loss", legend = false)

# ╔═╡ b4567890-cdef-0123-789a-234567890123
# Plot the trained UDE trajectories
tsample = 0.0:0.1:5.0

# ╔═╡ ae456ff0-4567-bcde-f012-901234567890
X̂ = predict(p_trained, Xₙ[:, 1], tsample)

# ╔═╡ bf567001-5678-cdef-0123-012345678901
# Trained on noisy data vs real solution
plot_trajectory = Plots.plot(tsample, X̂[1, :], label = "Learned Prey", color = :blue)

# ╔═╡ c0678112-6789-def0-1234-123456789012
Plots.plot!(plot_trajectory, tsample, X̂[2, :], label = "Learned Predator", color = :orange)

# ╔═╡ d1789223-789a-ef01-2345-234567890123
Plots.scatter!(plot_trajectory, t, Xₙ[1, :], color = :blue, label = "Prey Data", alpha = 0.5)

# ╔═╡ e289a334-89ab-f012-3456-345678901234
Plots.scatter!(plot_trajectory, t, Xₙ[2, :], color = :orange, label = "Predator Data", alpha = 0.5)

# ╔═╡ f39ab445-9abc-0123-4567-456789012345
Plots.plot!(plot_trajectory, solution, color = :black, linestyle = :dash, alpha = 0.75,
           label = ["True Prey" "True Predator"], title = "Learned Dynamics vs True Solution")

# ╔═╡ c5678901-def0-1234-89ab-345678901234
md"""
## Step 5: Symbolic Regression - Discovering the Missing Physics

Now we use symbolic regression to find interpretable equations from the trained neural network.
"""

# ╔═╡ d6789012-ef01-2345-9abc-456789012345
# Generate dense data from trained model for symbolic regression
t_long = 0.0:0.001:5.0

# ╔═╡ a4abc556-abcd-1234-5678-567890123456
X_pred_long = predict(p_trained, Xₙ[:, 1], t_long)

# ╔═╡ b5bcd667-bcde-2345-6789-678901234567
# Calculate the neural network contributions at each point
nn_contributions = zeros(2, length(t_long))

# ╔═╡ c6cde778-cdef-3456-789a-789012345678
for i in 1:length(t_long)
    nn_contributions[:, i] = U(X_pred_long[:, i], p_trained, st)[1]
end

# ╔═╡ e8ef099a-ef01-5678-9abc-901234567890
# True missing terms for comparison
true_missing_terms = zeros(2, length(t_long))

# ╔═╡ f9f01aab-f012-6789-abcd-012345678901
for i in 1:length(t_long)
    true_missing_terms[1, i] = -p_[2] * X_pred_long[1, i] * X_pred_long[2, i]
    true_missing_terms[2, i] = p_[3] * X_pred_long[1, i] * X_pred_long[2, i]
end

# ╔═╡ e7890123-f012-3456-abcd-567890123456
# Symbolic regression using DataDrivenDiffEq
@MTK.variables u[1:2]

# ╔═╡ bb123ccd-1234-89ab-cdef-234567890123
# Polynomial basis for symbolic regression
basis = DataDrivenDiffEq.polynomial_basis(u, 2)

# ╔═╡ cc234dde-2345-9abc-def0-345678901234
# Create DataDrivenProblem
prob_dd = DataDrivenDiffEq.DirectDataDrivenProblem(X_pred_long, nn_contributions')

# ╔═╡ dd345eef-3456-abcd-ef01-456789012345
# Solve using sparse regression (SR3)
res_dd = DataDrivenDiffEq.solve(prob_dd,
                                basis,
                                DataDrivenSparse.SR3(1e-2),
                                maxiters = 10000)

# ╔═╡ ff567001-5678-cdef-0123-678901234567
# Display discovered equations
println("\n=== Discovered Missing Physics ===")

# ╔═╡ a0678112-6789-def0-1234-789012345678
println("Missing term for dx/dt:")

# ╔═╡ b1789223-789a-ef01-2345-890123456789
println(DataDrivenDiffEq.get_basis(res_dd)[1])

# ╔═╡ c289a334-89ab-f012-3456-901234567890
println("\nMissing term for dy/dt:")

# ╔═╡ d39ab445-9abc-0123-4567-012345678901
println(DataDrivenDiffEq.get_basis(res_dd)[2])

# ╔═╡ c8ef099a-ef01-5678-9abc-567890123456
println("\n=== True Missing Physics ===")

# ╔═╡ d9f01aab-f012-6789-abcd-678901234567
println("True missing term for dx/dt: -$(p_[2]) * u[1] * u[2]")

# ╔═╡ ea012bbc-0123-789a-bcde-789012345678
println("True missing term for dy/dt: $(p_[3]) * u[1] * u[2]")

# ╔═╡ f8901234-0123-4567-bcde-678901234567
# Visualize comparison between learned and true missing terms
plot_comp = Plots.plot(t_long, nn_contributions[1, :],
                       label = "Learned dx/dt", color = :blue, linewidth = 2)

# ╔═╡ fb123ccd-1234-89ab-cdef-890123456789
Plots.plot!(plot_comp, t_long, true_missing_terms[1, :],
           label = "True dx/dt", color = :red, linestyle = :dash, linewidth = 2)

# ╔═╡ ac234dde-2345-9abc-def0-901234567890
Plots.plot!(plot_comp, t_long, nn_contributions[2, :],
           label = "Learned dy/dt", color = :green, linewidth = 2)

# ╔═╡ bd345eef-3456-abcd-ef01-012345678901
Plots.plot!(plot_comp, t_long, true_missing_terms[2, :],
           label = "True dy/dt", color = :orange, linestyle = :dash, linewidth = 2,
           title = "Learned vs True Missing Terms", xlabel = "Time", ylabel = "Missing Terms")

# ╔═╡ a9012345-1234-5678-cdef-789012345678
md"""
## Summary

We successfully demonstrated how Universal Differential Equations can discover missing physics:

1. Generated synthetic data from Lotka-Volterra equations with noise
2. Built a hybrid model combining partial knowledge with neural networks
3. Trained using two-stage optimization (ADAM + LBFGS)
4. Applied symbolic regression to extract interpretable equations

The UDE approach discovered the missing predator-prey interaction terms, bridging mechanistic modeling and machine learning for interpretable scientific discovery.
"""

# ╔═╡ Cell order:
# ╟─a1b2c3d4-e5f6-7890-abcd-ef1234567890
# ╠═885f5685-5239-42aa-942c-2b9bdf7df221
# ╠═f5c9ea83-6dba-4848-8340-1a8c772bd942
# ╠═e13c7d4f-4cdb-43d5-9f77-33401cb501e3
# ╠═c7193526-d32f-4f2a-95aa-09e98b88b30c
# ╠═36985121-9fe5-4961-9ec2-3f5ee57cfaa2
# ╠═45dc0462-d63c-11ef-0f62-e5d029618cc2
# ╠═6d29e44b-3d70-4700-ad53-9e75df7ced0d
# ╠═51c67a4d-f49a-4bb5-a1c2-361ea0e8f20f
# ╠═ca453691-227c-44f1-8b48-8fb42c42d482
# ╠═34efa390-0d6d-48dd-b08b-0c1bc391ffc3
# ╠═448dd6ac-0d7f-4c1f-a9ce-3f9a586951c2
# ╠═760b16ea-410f-4f46-ae44-f859de9b75e4
# ╠═abc12345-6789-def0-1234-567890abcdef
# ╠═bcd23456-789a-ef01-2345-6789abcdef01
# ╠═cde34567-89ab-f012-3456-789abcdef012
# ╠═def45678-9abc-0123-4567-89abcdef0123
# ╠═ef456789-abcd-1234-5678-9abcdef01234
# ╟─b2c3d4e5-f6a7-8901-bcde-f12345678901
# ╠═c3d4e5f6-a7b8-9012-cdef-123456789012
# ╠═f5678901-bcde-2345-6789-abcdef012345
# ╠═a6789012-cdef-3456-789a-bcdef0123456
# ╠═b789a123-def0-4567-89ab-cdef01234567
# ╠═c89ab234-ef01-5678-9abc-def012345678
# ╠═d9abc345-f012-6789-abcd-ef0123456789
# ╠═eabcd456-0123-789a-bcde-f01234567890
# ╠═d4e5f6a7-b890-1234-def0-234567890123
# ╠═fabcd567-1234-89ab-cdef-012345678901
# ╠═aabcd678-2345-9abc-def0-123456789012
# ╠═bbcde789-3456-abcd-ef01-234567890123
# ╠═ccdef890-4567-bcde-f012-345678901234
# ╠═ddef0901-5678-cdef-0123-456789012345
# ╠═eef01a12-6789-def0-1234-567890123456
# ╟─e5f6a7b8-9012-3456-ef01-345678901234
# ╠═f6a7b890-1234-5678-f012-456789012345
# ╠═fbcde567-1234-89ab-cdef-012345678901
# ╠═acdef678-2345-9abc-def0-123456789012
# ╠═bdef0789-3456-abcd-ef01-234567890123
# ╠═cef01890-4567-bcde-f012-345678901234
# ╠═df012901-5678-cdef-0123-456789012345
# ╟─b8901234-5678-9abc-1234-678901234567
# ╠═c9012345-6789-abcd-2345-789012345678
# ╠═b3456d45-9abc-0123-4567-890123456789
# ╠═c4567e56-abcd-1234-5678-901234567890
# ╠═d5678f67-bcde-2345-6789-012345678901
# ╟─d0123456-789a-bcde-3456-890123456789
# ╠═e1234567-89ab-cdef-4567-901234567890
# ╠═e6789078-cdef-3456-789a-123456789012
# ╠═f7890189-def0-4567-89ab-234567890123
# ╠═a890129a-ef01-5678-9abc-345678901234
# ╠═b90123ab-f012-6789-abcd-456789012345
# ╠═f2345678-9abc-def0-5678-012345678901
# ╠═ca0123bc-0123-789a-bcde-567890123456
# ╠═db1234cd-1234-89ab-cdef-678901234567
# ╠═ec2345de-2345-9abc-def0-789012345678
# ╠═a3456789-bcde-f012-6789-123456789012
# ╠═b4567890-cdef-0123-789a-234567890123
# ╠═ae456ff0-4567-bcde-f012-901234567890
# ╠═bf567001-5678-cdef-0123-012345678901
# ╠═c0678112-6789-def0-1234-123456789012
# ╠═d1789223-789a-ef01-2345-234567890123
# ╠═e289a334-89ab-f012-3456-345678901234
# ╠═f39ab445-9abc-0123-4567-456789012345
# ╟─c5678901-def0-1234-89ab-345678901234
# ╠═d6789012-ef01-2345-9abc-456789012345
# ╠═a4abc556-abcd-1234-5678-567890123456
# ╠═b5bcd667-bcde-2345-6789-678901234567
# ╠═c6cde778-cdef-3456-789a-789012345678
# ╠═e8ef099a-ef01-5678-9abc-901234567890
# ╠═f9f01aab-f012-6789-abcd-012345678901
# ╠═e7890123-f012-3456-abcd-567890123456
# ╠═bb123ccd-1234-89ab-cdef-234567890123
# ╠═cc234dde-2345-9abc-def0-345678901234
# ╠═dd345eef-3456-abcd-ef01-456789012345
# ╠═ff567001-5678-cdef-0123-678901234567
# ╠═a0678112-6789-def0-1234-789012345678
# ╠═b1789223-789a-ef01-2345-890123456789
# ╠═c289a334-89ab-f012-3456-901234567890
# ╠═d39ab445-9abc-0123-4567-012345678901
# ╠═c8ef099a-ef01-5678-9abc-567890123456
# ╠═d9f01aab-f012-6789-abcd-678901234567
# ╠═ea012bbc-0123-789a-bcde-789012345678
# ╠═f8901234-0123-4567-bcde-678901234567
# ╠═fb123ccd-1234-89ab-cdef-890123456789
# ╠═ac234dde-2345-9abc-def0-901234567890
# ╠═bd345eef-3456-abcd-ef01-012345678901
# ╟─a9012345-1234-5678-cdef-789012345678