### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-5e6f-11ef-0000-1a2b3c4d5e6f
md"""
# Julia Workshop Exercises

This notebook contains exercises from the UCI Data Science Initiative Julia workshop.
Work through these problems at your own pace. Solutions are provided in the companion `Workshop_Solutions.jl` notebook.

## Instructions
- Complete each exercise in the provided code cells
- Test your solutions with the given test cases
- Check your answers against the solutions notebook when ready
"""

# ╔═╡ b2c3d4e5-6f70-11ef-1111-2b3c4d5e6f70
md"""
## Starter Problems

### Note: The * problems are from https://lectures.quantecon.org/jl/julia_by_example.html
"""

# ╔═╡ c3d4e5f6-7081-11ef-2222-3c4d5e6f7081
md"""
### Strang Matrix Problem

Use Julia's array and control flow syntax in order to define the NxN Strang matrix:

$$ \left[\begin{array}{ccccc}
-2 & 1\\
1 & -2 & 1\\
 & \ddots & \ddots & \ddots\\
 &  & \ddots & \ddots & 1\\
 &  &  & 1 & -2
\end{array}\right] $$

i.e. a matrix with `-2` on the diagonal, 1 on the off-diagonals, and 0 elsewhere.
"""

# ╔═╡ d4e5f607-8192-11ef-3333-4d5e6f708192
# Your solution here
N = 10
# Create the Strang matrix

# ╔═╡ e5f60718-92a3-11ef-4444-5e6f7081a2b3
md"""
### Factorial Problem*

Using a `for` loop, write a function `my_factorial(n)` that computes the `n`th factorial. Try your function on integers like `15`

**Bonus**: Use `BigInt` inputs like `big(100)`. Make your function's output type match the input type for `n`. You'll know that you'll successfully matched the input type if your output does not "overflow" to negative, and you can check `typeof(x)`. Hint, you may want to initialize a value using `one(x)`, which is the value `1` in the type that matches `x`.
"""

# ╔═╡ f6071829-a3b4-11ef-5555-6f7081a2b3c4
function my_factorial(n)
    # Your solution here

end

# ╔═╡ 0718293a-b4c5-11ef-6666-70819233c4d5
# Test your solution
# my_factorial(4)
# my_factorial(15)
# my_factorial(big(30))

# ╔═╡ 1829304b-c5d6-11ef-7777-8192a344d5e6
md"""
### Binomial Problem*

A random variable `X~Bin(n,p)` is defined the number of successes in `n` trials where each trial has a success probability `p`. For example, if `Bin(10,0.5)`, then `X` is the number of coin flips that turn up heads in `10` flips.

Using only `rand()` (uniform random numbers), write a function `binomial_rv(n,p)` that produces one draw of `Bin(n,p)`.
"""

# ╔═╡ 2930415c-d6e7-11ef-8888-92a3b455e6f7
function binomial_rv(n, p)
    # Your solution here

end

# ╔═╡ 3a41526d-e7f8-11ef-9999-a3b4c566f708
# Test your solution
# bs = [binomial_rv(10, 0.5) for j in 1:10]

# ╔═╡ 4b52637e-f809-11ef-aaaa-b4c5d677f819
md"""
### Monte Carlo π Problem*

Use random number generation to estimate π. To do so, mentally draw the unit circle. It is encompassed in the square `[-1,1]×[-1,1]`. The area of the circle is `πr² = π`. The area of the square is `4`. Thus if points are randomly taken evenly from `[-1,1]×[-1,1]`, then the probability they land in the circle (`x² + y² ≤ 1`) is `π/4`. Use this to estimate π.
"""

# ╔═╡ 5c63748f-091a-11ef-bbbb-c5d6e7880920
# Your solution here
n = 10000
# Estimate π

# ╔═╡ 6d748590-1a2b-11ef-cccc-d6e7f8891a31
md"""
## Integration Problems

These problems integrate basic workflow tools to solve some standard data science and scientific computing problems.
"""

# ╔═╡ 7e8596a1-2b3c-11ef-dddd-e7f8992b3c42
md"""
### Timeseries Generation Problem*

An AR1 timeseries is defined by

$$ x_{t+1} = \alpha x_t + \epsilon_{t+1} $$

where $x_0 = 0$ and $t=0,\ldots,T$. The shocks ${\epsilon_t}$ are i.i.d. standard normal (`N(0,1)`, given by `randn()`). Using $T=200$ for:

1. $\alpha = 0$
2. $\alpha = 0.5$
3. $\alpha = 0.9$

use Plots.jl to plot a timecourse for each of the parameters. Label the lines for the values of $\alpha$ that generate them using the `label` argument in `plot`.
"""

# ╔═╡ 8f96a7b2-3c4d-11ef-eeee-f80993c4d5e3
# using Plots
# Your solution here

# ╔═╡ 907b8c3-4d5e-11ef-ffff-0914d5e6f709
md"""
### Logistic Map Problem

The logistic difference equation is defined by the recursion

$$ b_{n+1}=r \cdot b_{n}(1-b_{n}) $$

where $b_{n}$ is the number of bunnies at time $n$. Starting with $b_{0}=.25$, by around $400$ iterations this will reach a steady state. This steady state (or steady periodic state) is dependent on $r$. Write a function which plots the steady state attractor. This is done as follows:

1) Solve for the steady state(s) for each given $r$ (i.e. iterate the relation 400 times).

2) Calculate "every state" in the steady state attractor. This means, at steady state (after the first 400 iterations), save the next 150 values. Call this set of values $y_s(r)$.

3) Do steps (1) and (2) with $r\in\left(2.9,4\right)$, `dr=.001`. Plot $r$ x-axis vs $y_s(r)$ (value seen in the attractor) using Plots.jl. Your result should be the [Logistic equation bifurcation diagram](https://upload.wikimedia.org/wikipedia/commons/7/7d/LogisticMap_BifurcationDiagram.png).
"""

# ╔═╡ a18c9d4-5e6f-11ef-0001-1a15e6f70819
# Your solution here

# ╔═╡ b29dae5-6f70-11ef-0002-2b26f7081920
md"""
## Intermediate Problems
"""

# ╔═╡ c3aebf6-7081-11ef-0003-3c37081a2b31
md"""
### MyRange and LinSpace Problem

#### Part 1

Let's create our own implementation of the range type. The `Range` type is what you get from `1:2:20`. It's form is `start:step:stop`. If you know `start`, `step`, and `stop`, how do you calculate the `i`th value? Create a type `MyRange` which stores `start`, `step`, and `stop`. Can you create a function `_MyRange(a,i)` which for `a` being a `MyRange`, it returns what `a[i]` should be? After getting this correct, use the [Julia array interface](https://docs.julialang.org/en/stable/manual/interfaces/#Indexing-1) in order to define the function for the `a[i]` syntax on your type.

#### Part 2

A LinSpace object is a lazy representation of `N` values from `start` to `stop`. Use the Array interface to implement a lazy version of the LinSpace. Test against `range(start,stop=stop,length=N)`.

#### Part 3

Check out call overloading. Overload the call on the UnitStepRange to give an interpolated value at intermediate points, i.e. if `a=1:2:10`, then `a(1.5)=2`.

#### Part 4

Do your implementations obey dimensional analysis? Try using the package `Unitful` to build arrays of numbers with units (i.e. an array of numbers who have values of Newtons), and see if you can make your LinSpace not give errors.
"""

# ╔═╡ d4bfc07-8192-11ef-0004-4d481923c42
# Part 1: MyRange implementation
struct MyRange
    # Your fields here

end

# ╔═╡ e5c0d18-92a3-11ef-0005-5e592a34d53
# Part 2: LinSpace implementation
struct MyLinSpace
    # Your fields here

end

# ╔═╡ f6d1e29-a3b4-11ef-0006-6f6a3b45e64
md"""
### Operator Problem

In mathematics, a matrix is known to be a linear operator. In many cases, this can have huge performance advantages because, if you know a function which "acts like a matrix" but does not form the matrix itself, you can save the time that it takes to allocate the matrix (sometimes the matrix may not fit in memory!)

Recall the Strang matrix. Define a type `StrangMatrix` and define a dispatch such that `A*x` acts like a Strang matrix on a vector.

**Advanced Bonus**: Iterative solvers solve `Ax=b` and only require the definition of matrix multiplication. Thus utilize IterativeSolvers.jl to solve `Ax=b` for `b=rand(100)` using your lazy matrix type. Hint: you will need to define `mul!` from `LinearAlgebra` (standard library). You will also need to define a different version of your Strang matrix which holds a size and has `Base.eltype` defined.
"""

# ╔═╡ 07e2f3a-b4c5-11ef-0007-7074c556f75
# Your StrangMatrix type and multiplication

# ╔═╡ 18f404b-c5d6-11ef-0008-8185d667f86
md"""
### Regression Problem

Given an Nx3 array of data (`randn(N,3)`) and a Nx1 array of outcomes, produce the data matrix `X` which appends a column of 1's to the front of the data matrix, and solve for the 4x1 array `β` via `βX = b` using `qrfact`, or `\`, or [the definition of the OLS estimator](https://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation). (Note: This is linear regression).

Compare your results to that of using `llsq` from `MultivariateStats.jl` (note: you need to go find the documentation to find out how to use this!). Compare your results to that of using ordinary least squares regression from `GLM.jl`.

#### Part 2

Using your OLS estimator or one of the aforementioned packages, solve for the regression line using the (X,y) data below. Plot the (X,y) scatter plot using `scatter!` from Plots.jl. Add the regression line. Add a title saying "Regression Plot on Fake Data", and label the x and y axis.
"""

# ╔═╡ 2a0515c-d6e7-11ef-0009-9296e778f97
# Prepare Data For Regression Problem
begin
    X_reg = rand(1000, 3)               # feature matrix
    a0 = rand(3)                         # ground truths
    y_reg = X_reg * a0 + 0.1 * randn(1000)  # generate response

    # Data For Regression Problem Part 2
    X_reg2 = rand(100)
    y_reg2 = 2 * X_reg2 + 0.1 * randn(100)
end

# ╔═╡ 3b1626d-e7f8-11ef-000a-a3a9f889fa8
# Your regression solution

# ╔═╡ 4c2737e-f809-11ef-000b-b4b90990aa9
md"""
### Type Hierarchy Problem

Make a function `person_info(x)` where, if `x` is a any type of person, print their name. However, if `x` is a student, print their name and their grade. Make a new type which is a graduate student, and have it print their name and grade as well. If `x` is anything else, throw an error. Do not use branching (`if`), use multiple dispatch to solve the problem!

Note that in order to do this you will need to re-structure the type hierarchy. Make an AbstractPerson and AbstractStudent type, define the subclassing structure, and write dispatches on these abstract types. Note that you cannot define subclasses of concrete types!
"""

# ╔═╡ 5d3848f-091a-11ef-000c-c5c91aa1bb0
# Your type hierarchy and dispatch solution

# ╔═╡ 6e4959a-1a2b-11ef-000d-d6da2bb2cc1
md"""
### Distribution Quantile Problem (From Josh Day)

To find the quantile of a number `q` in a distribution, one can use a Newton method

$$ \theta_{n+1} = \theta_{n} - \frac{cdf(\theta)-q}{pdf(\theta)} $$

to have $\theta_{n} \rightarrow$ the value of for the `q`th quantile. Use multiple dispatch to write a generic algorithm for which calculates the `q`th quantile of any `UnivariateDistribution` in Distributions.jl, and test your result against the `quantile(d::UnivariateDistribution,q::Number)` function.

Hint: Use $\theta_{0} = $ mean of the distribution
"""

# ╔═╡ 7f5a6ab-2b3c-11ef-000e-e7eb3cc3dd2
# Your quantile solution

# ╔═╡ 906b7bc-3c4d-11ef-000f-f8fc4dd4ee3
md"""
## Advanced Problems
"""

# ╔═╡ a17c8cd-4d5e-11ef-0010-0910bee5ff4
md"""
### Metaprogramming Problem

[Metaprogramming in Julia](https://docs.julialang.org/en/stable/manual/metaprogramming/) is the practice of writing code that generates code. There are many uses for metaprogramming, but it generally falls into two categories:

1. Implementing "new language features" you want.
2. Implementing syntactic sugar.

Evaluation of a polynomial is a common task in many disciplines. Julia's Base provides `@evalpoly x a0 a1 a2 ...` that implements `a0 + x*a1 + x^2 * a2 + ...` using Horner's rule, which is writing it out as: `(((an*x) + a(n-1))*x + ...) * x + a0`.

Implement your own version of the `@evalpoly` macro called `@myevalpoly`.

**Note**: While you can create values using macros in the top level scope, this is not good practice and will not work in function scopes. Instead, you should return an expression for the computation of the polynomial.
"""

# ╔═╡ b28d9de-5e6f-11ef-0011-1a11cff6005
# Your macro implementation

# ╔═╡ c39eaef-6f70-11ef-0012-2b22dff7116
md"""
### Plot the roots of Wilkinson's polynomial with perturbation

[Wilkinson's polynomial](https://en.wikipedia.org/wiki/Wilkinson%27s_polynomial) has the form
$$
w(x)=\underbrace{\prod_{i=1}^{20}(x-i)}_\text{root form}=\underbrace{a_1+a_2x+a_3x^2+\cdots+a_{21}x^{20}}_\text{coefficient form}.
$$

It is a famous example of ill-conditioning in numerical analysis. One can show this visually by plotting the roots of polynomials with perturbed coefficients $\hat{a}_k=a_k(1+10^{-10}r_k)$, where $r_k$ is a normally distributed random number.

This problem has three parts, which are:

1. Convert root form to coefficient form. (Compute $a_k$)
2. Calculate roots of a polynomial by using the [companion matrix](https://en.wikipedia.org/wiki/Companion_matrix).
3. Plot the roots of polynomials
"""

# ╔═╡ d4afbfa-7081-11ef-0013-3c33eff8227
# Your Wilkinson's polynomial solution

# ╔═╡ Cell order:
# ╟─a1b2c3d4-5e6f-11ef-0000-1a2b3c4d5e6f
# ╟─b2c3d4e5-6f70-11ef-1111-2b3c4d5e6f70
# ╟─c3d4e5f6-7081-11ef-2222-3c4d5e6f7081
# ╠═d4e5f607-8192-11ef-3333-4d5e6f708192
# ╟─e5f60718-92a3-11ef-4444-5e6f7081a2b3
# ╠═f6071829-a3b4-11ef-5555-6f7081a2b3c4
# ╠═0718293a-b4c5-11ef-6666-70819233c4d5
# ╟─1829304b-c5d6-11ef-7777-8192a344d5e6
# ╠═2930415c-d6e7-11ef-8888-92a3b455e6f7
# ╠═3a41526d-e7f8-11ef-9999-a3b4c566f708
# ╟─4b52637e-f809-11ef-aaaa-b4c5d677f819
# ╠═5c63748f-091a-11ef-bbbb-c5d6e7880920
# ╟─6d748590-1a2b-11ef-cccc-d6e7f8891a31
# ╟─7e8596a1-2b3c-11ef-dddd-e7f8992b3c42
# ╠═8f96a7b2-3c4d-11ef-eeee-f80993c4d5e3
# ╟─907b8c3-4d5e-11ef-ffff-0914d5e6f709
# ╠═a18c9d4-5e6f-11ef-0001-1a15e6f70819
# ╟─b29dae5-6f70-11ef-0002-2b26f7081920
# ╟─c3aebf6-7081-11ef-0003-3c37081a2b31
# ╠═d4bfc07-8192-11ef-0004-4d481923c42
# ╠═e5c0d18-92a3-11ef-0005-5e592a34d53
# ╟─f6d1e29-a3b4-11ef-0006-6f6a3b45e64
# ╠═07e2f3a-b4c5-11ef-0007-7074c556f75
# ╟─18f404b-c5d6-11ef-0008-8185d667f86
# ╠═2a0515c-d6e7-11ef-0009-9296e778f97
# ╠═3b1626d-e7f8-11ef-000a-a3a9f889fa8
# ╟─4c2737e-f809-11ef-000b-b4b90990aa9
# ╠═5d3848f-091a-11ef-000c-c5c91aa1bb0
# ╟─6e4959a-1a2b-11ef-000d-d6da2bb2cc1
# ╠═7f5a6ab-2b3c-11ef-000e-e7eb3cc3dd2
# ╟─906b7bc-3c4d-11ef-000f-f8fc4dd4ee3
# ╟─a17c8cd-4d5e-11ef-0010-0910bee5ff4
# ╠═b28d9de-5e6f-11ef-0011-1a11cff6005
# ╟─c39eaef-6f70-11ef-0012-2b22dff7116
# ╠═d4afbfa-7081-11ef-0013-3c33eff8227