### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-5e6f-11ef-0000-1a2b3c4d5e6f
using BenchmarkTools

# ╔═╡ b2c3d4e5-6f70-11ef-1111-2b3c4d5e6f70
md"""
# Julia Workshop Solutions

This notebook contains solutions to the exercises from the UCI Data Science Initiative Julia workshop.
Compare your solutions with these implementations to learn different approaches.

## Instructions
- Review each solution and compare with your approach
- Try to understand why certain implementations were chosen
- Experiment with alternative solutions
"""

# ╔═╡ c3d4e5f6-7081-11ef-2222-3c4d5e6f7081
md"""
## Starter Problems Solutions
"""

# ╔═╡ d4e5f607-8192-11ef-3333-4d5e6f708192
md"""
### Strang Matrix Solution
"""

# ╔═╡ e5f60718-92a3-11ef-4444-5e6f7081a2b3
# Strang Matrix Solution
N = 10
A = zeros(N,N)
for i in 1:N, j in 1:N
    abs(i-j) <= 1 && (A[i,j] += 1)
    i == j && (A[i,j] -= 3)
end
A

# ╔═╡ f6071829-a3b4-11ef-5555-6f7081a2b3c4
md"""
### Factorial Solution
"""

# ╔═╡ 0718293a-b4c5-11ef-6666-70819233c4d5
function my_factorial(n)
    k = one(n)  # Initialize with correct type
    for i in 1:n
        k *= i
    end
    k
end

# ╔═╡ 1829304b-c5d6-11ef-7777-8192a344d5e6
# Test factorial function
@show my_factorial(4)
@show my_factorial(15)
@show my_factorial(big(30))

# ╔═╡ 2930415c-d6e7-11ef-8888-92a3b455e6f7
md"""
### Binomial Solution
"""

# ╔═╡ 3a41526d-e7f8-11ef-9999-a3b4c566f708
function binomial_rv(n, p)
    count = zero(n)
    U = rand(n)
    for i in 1:n
        U[i] < p && (count += 1)
    end
    count
end

# ╔═╡ 4b52637e-f809-11ef-aaaa-b4c5d677f819
# Test binomial function
bs = [binomial_rv(10, 0.5) for j in 1:10]

# ╔═╡ 5c63748f-091a-11ef-bbbb-c5d6e7880920
md"""
### Monte Carlo π Solution
"""

# ╔═╡ 6d748590-1a2b-11ef-cccc-d6e7f8891a31
# Monte Carlo π estimation
function estimate_pi(n)
    count = 0
    for i in 1:n
        u, v = 2rand(2) .- 1  # Random point in [-1,1] x [-1,1]
        d = sqrt(u^2 + v^2)   # Distance from origin
        d < 1 && (count += 1)  # Count if inside unit circle
    end
    area_estimate = count / n
    return area_estimate * 4  # π/4 is the probability
end

# ╔═╡ 7e8596a1-2b3c-11ef-dddd-e7f8992b3c42
pi_estimate = estimate_pi(1000000)

# ╔═╡ 8f96a7b2-3c4d-11ef-eeee-f80993c4d5e3
md"""
## Integration Problems Solutions
"""

# ╔═╡ 907b8c3-4d5e-11ef-ffff-0914d5e6f709
md"""
### Timeseries Generation Solution
"""

# ╔═╡ a18c9d4-5e6f-11ef-0001-1a15e6f70819
using Plots

# ╔═╡ b29dae5-6f70-11ef-0002-2b26f7081920
begin
    gr()

    alphas = [0.0, 0.5, 0.98]
    T = 200

    series = []
    labels = []

    for alpha in alphas
        x = zeros(T + 1)
        x[1] = 0.0
        for t in 1:T
            x[t+1] = alpha * x[t] + randn()
        end
        push!(series, x)
        push!(labels, "alpha = $alpha")
    end

    plot(series, label=reshape(labels,1,length(labels)), lw=3,
         title="AR1 Timeseries", xlabel="Time", ylabel="Value")
end

# ╔═╡ c3aebf6-7081-11ef-0003-3c37081a2b31
md"""
### Logistic Map Solution
"""

# ╔═╡ d4bfc07-8192-11ef-0004-4d481923c42
begin
    r = 2.9:.001:4
    numAttract = 150
    steady = ones(length(r),1) * 0.25

    # Get to steady state
    for i=1:400
        @. steady = r * steady * (1 - steady)
    end

    # Grab values at the attractor
    x = zeros(length(steady), numAttract)
    x[:,1] = steady
    @inbounds for i=2:numAttract
        @. x[:,i] = r * x[:,i-1] * (1 - x[:,i-1])
    end

    plot(collect(r), x, seriestype=:scatter, markersize=.002,
         legend=false, color=:black,
         title="Logistic Bifurcation Diagram",
         xlabel="r", ylabel="Attractor values")
end

# ╔═╡ e5c0d18-92a3-11ef-0005-5e592a34d53
md"""
## Intermediate Problems Solutions
"""

# ╔═╡ f6d1e29-a3b4-11ef-0006-6f6a3b45e64
md"""
### MyRange and LinSpace Solution
"""

# ╔═╡ 07e2f3a-b4c5-11ef-0007-7074c556f75
# Part 1: MyRange implementation
struct MyRange
    start
    step
    stop
end

# ╔═╡ 18f404b-c5d6-11ef-0008-8185d667f86
function _MyRange(a::MyRange, i::Int)
    tmp = a.start + a.step * (i - 1)
    if tmp > a.stop
        error("Index is out of bounds!")
    else
        return tmp
    end
end

# ╔═╡ 2a0515c-d6e7-11ef-0009-9296e778f97
Base.getindex(a::MyRange, i::Int) = _MyRange(a, i)

# ╔═╡ 3b1626d-e7f8-11ef-000a-a3a9f889fa8
# Test MyRange
begin
    my_range = MyRange(1, 2, 20)
    @show my_range[5]
    @show (1:2:20)[5]
end

# ╔═╡ 4c2737e-f809-11ef-000b-b4b90990aa9
# Part 2: LinSpace implementation
struct MyLinSpace
    start
    stop
    n
end

# ╔═╡ 5d3848f-091a-11ef-000c-c5c91aa1bb0
function Base.getindex(a::MyLinSpace, i::Int)
    dx = (a.stop - a.start) / (a.n - 1)
    a.start + dx * (i - 1)
end

# ╔═╡ 6e4959a-1a2b-11ef-000d-d6da2bb2cc1
# Test MyLinSpace
begin
    l = MyLinSpace(1, 2, 50)
    @show l[6]
    @show range(1, stop=2, length=50)[6]
end

# ╔═╡ 7f5a6ab-2b3c-11ef-000e-e7eb3cc3dd2
# Part 3: Call overloading for interpolation
(a::MyRange)(x) = a.start + a.step * (x - 1)

# ╔═╡ 906b7bc-3c4d-11ef-000f-f8fc4dd4ee3
begin
    test_range = MyRange(1, 2, 20)
    @show test_range(1.5)
end

# ╔═╡ a17c8cd-4d5e-11ef-0010-0910bee5ff4
md"""
### Operator Problem Solution
"""

# ╔═╡ b28d9de-5e6f-11ef-0011-1a11cff6005
struct StrangMatrix end

# ╔═╡ c39eaef-6f70-11ef-0012-2b22dff7116
using LinearAlgebra

# ╔═╡ d4afbfa-7081-11ef-0013-3c33eff8227
function LinearAlgebra.mul!(C, A::StrangMatrix, B::AbstractVector)
    n = length(B)
    for i in 2:n-1
        C[i] = B[i-1] - 2B[i] + B[i+1]
    end
    C[1] = -2B[1] + B[2]
    C[end] = B[end-1] - 2B[end]
    C
end

# ╔═╡ e5bfc0b-8192-11ef-0014-4d44f0093328
Base.:*(A::StrangMatrix, B::AbstractVector) = (C = similar(B); mul!(C, A, B))

# ╔═╡ f6c0d1c-92a3-11ef-0015-5e592a34d53e
# Test Strang matrix
A_strang = StrangMatrix()
test_vec = ones(10)
A_strang * test_vec

# ╔═╡ 07d1e2d-a3b4-11ef-0016-6f6a3b45e64f
# Advanced: Sized Strang matrix for iterative solvers
struct SizedStrangMatrix
    size
end

# ╔═╡ 18e2f3e-b4c5-11ef-0017-7074c556f75g
Base.eltype(A::SizedStrangMatrix) = Float64
Base.size(A::SizedStrangMatrix) = A.size
Base.size(A::SizedStrangMatrix, i::Int) = A.size[i]

# ╔═╡ 29f404f-c5d6-11ef-0018-8185d667f86h
function LinearAlgebra.mul!(C, A::SizedStrangMatrix, B)
    n = length(B)
    for i in 2:n-1
        C[i] = B[i-1] - 2B[i] + B[i+1]
    end
    C[1] = -2B[1] + B[2]
    C[end] = B[end-1] - 2B[end]
    C
end

# ╔═╡ 3a0516c-d6e7-11ef-0019-9296e778f97i
Base.:*(A::SizedStrangMatrix, B::AbstractVector) = (C = similar(B); mul!(C, A, B))

# ╔═╡ 4b1627d-e7f8-11ef-001a-a3a9f889fa8j
md"""
### Regression Problem Solution
"""

# ╔═╡ 5c2738e-f809-11ef-001b-b4b90990aa9k
begin
    # Prepare data
    X = rand(1000, 3)               # feature matrix
    a0 = rand(3)                     # ground truths
    y = X * a0 + 0.1 * randn(1000)  # generate response

    # Add intercept column
    X2 = hcat(ones(1000), X)

    # OLS solution using backslash operator
    β = X2 \ y

    println("OLS coefficients (with intercept): ", β)
end

# ╔═╡ 6d3849f-091a-11ef-001c-c5c91aa1bb0l
# Part 2: Regression plot
begin
    X_simple = rand(100)
    y_simple = 2 * X_simple + 0.1 * randn(100)

    b = X_simple \ y_simple

    scatter(X_simple, y_simple, label="Data", alpha=0.6,
            title="Regression Plot on Fake Data",
            xlabel="X", ylabel="Y")

    # Add regression line
    x_line = range(minimum(X_simple), maximum(X_simple), length=100)
    plot!(x_line, b * x_line, lw=3, label="Regression Line", color=:red)
end

# ╔═╡ 7e4950a-1a2b-11ef-001d-d6da2bb2cc1m
md"""
### Type Hierarchy Solution
"""

# ╔═╡ 8f5a6bb-2b3c-11ef-001e-e7eb3cc3dd2n
# Define abstract types
abstract type AbstractPerson end
abstract type AbstractStudent <: AbstractPerson end

# ╔═╡ a06b7cc-3c4d-11ef-001f-f8fc4dd4ee3o
# Define concrete types
struct Person <: AbstractPerson
    name::String
end

# ╔═╡ b17c8dd-4d5e-11ef-0020-0910bee5ff4p
struct Student <: AbstractStudent
    name::String
    grade::Int
end

# ╔═╡ c28d9ee-5e6f-11ef-0021-1a11cff6005q
struct GraduateStudent <: AbstractStudent
    name::String
    grade::Int
end

# ╔═╡ d39eaff-6f70-11ef-0022-2b22dff7116r
# Define dispatch methods
person_info(p::AbstractPerson) = println("Name: ", p.name)
person_info(s::AbstractStudent) = println("Name: ", s.name, ", Grade: ", s.grade)
person_info(x) = error("Not a person type!")

# ╔═╡ e4afb0a-7081-11ef-0023-3c33eff8227s
# Test the type hierarchy
person_info(Person("Alice"))
person_info(Student("Bob", 10))
person_info(GraduateStudent("Carol", 6))

# ╔═╡ f5bfc1b-8192-11ef-0024-4d44f0093328t
md"""
### Distribution Quantile Solution
"""

# ╔═╡ 06c0d2c-92a3-11ef-0025-5e592a34d53u
using Distributions

# ╔═╡ 17d1e3d-a3b4-11ef-0026-6f6a3b45e64v
function myquantile(d::UnivariateDistribution, q::Number)
    θ = mean(d)
    tol = Inf
    max_iter = 100
    iter = 0

    while tol > 1e-5 && iter < max_iter
        θold = θ
        θ = θ - (cdf(d, θ) - q) / pdf(d, θ)  # Newton's method
        tol = abs(θold - θ)
        iter += 1
    end
    θ
end

# ╔═╡ 28e2f4e-b4c5-11ef-0027-7074c556f75w
# Test the quantile function
for dist in [Gamma(5, 1), Normal(0, 1), Beta(2, 4)]
    q = 0.75
    my_q = myquantile(dist, q)
    true_q = quantile(dist, q)
    println("Distribution: ", dist)
    println("  My quantile: ", my_q)
    println("  True quantile: ", true_q)
    println("  Error: ", abs(my_q - true_q))
    println()
end

# ╔═╡ 39f405f-c5d6-11ef-0028-8185d667f86x
md"""
## Advanced Problems Solutions
"""

# ╔═╡ 4a05170-d6e7-11ef-0029-9296e778f97y
md"""
### Metaprogramming Solution
"""

# ╔═╡ 5b16281-e7f8-11ef-002a-a3a9f889fa8z
macro myevalpoly(x, p...)
    ex = :($(p[end]))
    for i = length(p)-1:-1:1
        ex = :(muladd($x, $ex, $(p[i])))
    end
    ex
end

# ╔═╡ 6c27392-f809-11ef-002b-b4b90990aa9aa
# Test the macro
x_test = 2.0
result = @myevalpoly x_test 1 2 3 4  # 1 + 2x + 3x² + 4x³
expected = 1 + 2*2 + 3*4 + 4*8
@show result == expected

# ╔═╡ 7d384a3-091a-11ef-002c-c5c91aa1bb0ab
md"""
### Wilkinson's Polynomial Solution
"""

# ╔═╡ 8e495b4-1a2b-11ef-002d-d6da2bb2cc1ac
# Part 1: Convert root form to coefficient form
function roots_to_coefficients(roots)
    n = length(roots)
    coeffs = zeros(n + 1)
    coeffs[1] = 1.0

    for root in roots
        # Multiply by (x - root)
        new_coeffs = zeros(n + 1)
        new_coeffs[2:end] = coeffs[1:end-1]
        new_coeffs[1:end] -= root * coeffs[1:end]
        coeffs = new_coeffs
    end

    return coeffs
end

# ╔═╡ 9f5a6c5-2b3c-11ef-002e-e7eb3cc3dd2ad
# Part 2: Find roots using companion matrix
function polynomial_roots_companion(coeffs)
    n = length(coeffs) - 1
    if n <= 0
        return []
    end

    # Normalize so leading coefficient is 1
    coeffs = coeffs / coeffs[end]

    # Build companion matrix
    C = zeros(n, n)
    if n > 1
        C[2:end, 1:end-1] = I(n-1)
    end
    C[:, end] = -coeffs[1:n]

    # Eigenvalues are the roots
    return eigvals(C)
end

# ╔═╡ a06b7d6-3c4d-11ef-002f-f8fc4dd4ee3ae
# Part 3: Plot perturbed roots
begin
    using Random
    Random.seed!(123)

    # Original roots
    original_roots = 1:20

    # Convert to coefficients
    coeffs = roots_to_coefficients(original_roots)

    # Generate perturbed polynomials and find roots
    n_perturbations = 50
    all_roots = []

    for i in 1:n_perturbations
        # Perturb coefficients
        perturbation = 1 .+ 1e-10 * randn(length(coeffs))
        perturbed_coeffs = coeffs .* perturbation

        # Find roots
        roots = polynomial_roots_companion(perturbed_coeffs)
        push!(all_roots, roots)
    end

    # Plot original and perturbed roots
    scatter(real.(original_roots), imag.(original_roots),
            label="Original", markersize=8, color=:red)

    for roots in all_roots
        scatter!(real.(roots), imag.(roots),
                label="", markersize=2, alpha=0.3, color=:blue)
    end

    plot!(title="Wilkinson's Polynomial Root Perturbation",
          xlabel="Real Part", ylabel="Imaginary Part",
          legend=:topright)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.6.0"
Distributions = "~0.25.0"
Plots = "~1.40.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "..."

[deps.BenchmarkTools]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "..."
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

# ... (manifest continues)
"""

# ╔═╡ Cell order:
# ╠═a1b2c3d4-5e6f-11ef-0000-1a2b3c4d5e6f
# ╟─b2c3d4e5-6f70-11ef-1111-2b3c4d5e6f70
# ╟─c3d4e5f6-7081-11ef-2222-3c4d5e6f7081
# ╟─d4e5f607-8192-11ef-3333-4d5e6f708192
# ╠═e5f60718-92a3-11ef-4444-5e6f7081a2b3
# ╟─f6071829-a3b4-11ef-5555-6f7081a2b3c4
# ╠═0718293a-b4c5-11ef-6666-70819233c4d5
# ╠═1829304b-c5d6-11ef-7777-8192a344d5e6
# ╟─2930415c-d6e7-11ef-8888-92a3b455e6f7
# ╠═3a41526d-e7f8-11ef-9999-a3b4c566f708
# ╠═4b52637e-f809-11ef-aaaa-b4c5d677f819
# ╟─5c63748f-091a-11ef-bbbb-c5d6e7880920
# ╠═6d748590-1a2b-11ef-cccc-d6e7f8891a31
# ╠═7e8596a1-2b3c-11ef-dddd-e7f8992b3c42
# ╟─8f96a7b2-3c4d-11ef-eeee-f80993c4d5e3
# ╟─907b8c3-4d5e-11ef-ffff-0914d5e6f709
# ╠═a18c9d4-5e6f-11ef-0001-1a15e6f70819
# ╠═b29dae5-6f70-11ef-0002-2b26f7081920
# ╟─c3aebf6-7081-11ef-0003-3c37081a2b31
# ╠═d4bfc07-8192-11ef-0004-4d481923c42
# ╟─e5c0d18-92a3-11ef-0005-5e592a34d53
# ╟─f6d1e29-a3b4-11ef-0006-6f6a3b45e64
# ╠═07e2f3a-b4c5-11ef-0007-7074c556f75
# ╠═18f404b-c5d6-11ef-0008-8185d667f86
# ╠═2a0515c-d6e7-11ef-0009-9296e778f97
# ╠═3b1626d-e7f8-11ef-000a-a3a9f889fa8
# ╠═4c2737e-f809-11ef-000b-b4b90990aa9
# ╠═5d3848f-091a-11ef-000c-c5c91aa1bb0
# ╠═6e4959a-1a2b-11ef-000d-d6da2bb2cc1
# ╠═7f5a6ab-2b3c-11ef-000e-e7eb3cc3dd2
# ╠═906b7bc-3c4d-11ef-000f-f8fc4dd4ee3
# ╟─a17c8cd-4d5e-11ef-0010-0910bee5ff4
# ╠═b28d9de-5e6f-11ef-0011-1a11cff6005
# ╠═c39eaef-6f70-11ef-0012-2b22dff7116
# ╠═d4afbfa-7081-11ef-0013-3c33eff8227
# ╠═e5bfc0b-8192-11ef-0014-4d44f0093328
# ╠═f6c0d1c-92a3-11ef-0015-5e592a34d53e
# ╠═07d1e2d-a3b4-11ef-0016-6f6a3b45e64f
# ╠═18e2f3e-b4c5-11ef-0017-7074c556f75g
# ╠═29f404f-c5d6-11ef-0018-8185d667f86h
# ╠═3a0516c-d6e7-11ef-0019-9296e778f97i
# ╟─4b1627d-e7f8-11ef-001a-a3a9f889fa8j
# ╠═5c2738e-f809-11ef-001b-b4b90990aa9k
# ╠═6d3849f-091a-11ef-001c-c5c91aa1bb0l
# ╟─7e4950a-1a2b-11ef-001d-d6da2bb2cc1m
# ╠═8f5a6bb-2b3c-11ef-001e-e7eb3cc3dd2n
# ╠═a06b7cc-3c4d-11ef-001f-f8fc4dd4ee3o
# ╠═b17c8dd-4d5e-11ef-0020-0910bee5ff4p
# ╠═c28d9ee-5e6f-11ef-0021-1a11cff6005q
# ╠═d39eaff-6f70-11ef-0022-2b22dff7116r
# ╠═e4afb0a-7081-11ef-0023-3c33eff8227s
# ╟─f5bfc1b-8192-11ef-0024-4d44f0093328t
# ╠═06c0d2c-92a3-11ef-0025-5e592a34d53u
# ╠═17d1e3d-a3b4-11ef-0026-6f6a3b45e64v
# ╠═28e2f4e-b4c5-11ef-0027-7074c556f75w
# ╟─39f405f-c5d6-11ef-0028-8185d667f86x
# ╟─4a05170-d6e7-11ef-0029-9296e778f97y
# ╠═5b16281-e7f8-11ef-002a-a3a9f889fa8z
# ╠═6c27392-f809-11ef-002b-b4b90990aa9aa
# ╟─7d384a3-091a-11ef-002c-c5c91aa1bb0ab
# ╠═8e495b4-1a2b-11ef-002d-d6da2bb2cc1ac
# ╠═9f5a6c5-2b3c-11ef-002e-e7eb3cc3dd2ad
# ╠═a06b7d6-3c4d-11ef-002f-f8fc4dd4ee3ae
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002