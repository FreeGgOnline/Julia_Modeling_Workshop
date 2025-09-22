### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000001
begin
    using Pkg
    Pkg.activate(".")
    using PlutoUI
    using LinearAlgebra
    using SparseArrays
    using BenchmarkTools
    using StaticArrays
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000006
with_terminal() do
    println("Column-major iteration:")
    @btime col_major_sum($A_test)
    println("\nRow-major iteration:")
    @btime row_major_sum($A_test)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000008
begin
    # Broadcasting examples
    x_vec = [1, 2, 3]
    y_vec = [4, 5, 6]'  # Row vector (1×3)

    # Broadcasting automatically handles dimension expansion
    broadcast_result = x_vec .+ y_vec
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000010
begin
    function unfused_operations(x, y, z)
        tmp1 = x .+ y
        tmp2 = tmp1 .* z
        return sin.(tmp2)
    end

    function fused_operations(x, y, z)
        return sin.((x .+ y) .* z)
    end

    # Or using @. macro for automatic broadcasting
    function fused_with_macro(x, y, z)
        @. sin((x + y) * z)
    end

    test_x = rand(1000)
    test_y = rand(1000)
    test_z = rand(1000)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000011
with_terminal() do
    println("Unfused operations:")
    @btime unfused_operations($test_x, $test_y, $test_z)
    println("\nFused operations:")
    @btime fused_operations($test_x, $test_y, $test_z)
    println("\nFused with @. macro:")
    @btime fused_with_macro($test_x, $test_y, $test_z)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000018
begin
    # Create a sparse matrix
    n = 1000
    # Create a tridiagonal matrix
    sparse_mat = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))

    # Convert to dense for comparison
    dense_mat = Matrix(sparse_mat)

    # Random vector for multiplication
    sparse_vec = rand(n)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000019
with_terminal() do
    println("Memory usage:")
    println("Sparse: ", Base.summarysize(sparse_mat), " bytes")
    println("Dense: ", Base.summarysize(dense_mat), " bytes")

    println("\nMatrix-vector multiplication:")
    println("Sparse:")
    @btime $sparse_mat * $sparse_vec
    println("Dense:")
    @btime $dense_mat * $sparse_vec
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000021
begin
    # Different ways to build sparse matrices
    I_idx = [1, 2, 3, 4, 5]
    J_idx = [1, 2, 3, 4, 5]
    V_vals = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Using sparse constructor
    S1 = sparse(I_idx, J_idx, V_vals, 5, 5)

    # Using spzeros and setting values
    S2 = spzeros(5, 5)
    for (i, j, v) in zip(I_idx, J_idx, V_vals)
        S2[i, j] = v
    end

    S1
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000023
begin
    # Static arrays vs regular arrays
    regular_vec = [1.0, 2.0, 3.0]
    static_vec = SVector(1.0, 2.0, 3.0)

    regular_mat = [1.0 2.0; 3.0 4.0]
    static_mat = SMatrix{2,2}(1.0, 3.0, 2.0, 4.0)

    function sum_regular(x, y)
        return x + y
    end

    function sum_static(x::SVector, y::SVector)
        return x + y
    end
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000024
with_terminal() do
    println("Regular array addition:")
    @btime sum_regular($regular_vec, $regular_vec)
    println("\nStatic array addition:")
    @btime sum_static($static_vec, $static_vec)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000026
begin
    # Example: 3D rotation matrices
    function rotate_3d(θ)
        return SMatrix{3,3}(
            cos(θ), -sin(θ), 0,
            sin(θ), cos(θ), 0,
            0, 0, 1
        )
    end

    point_3d = SVector(1.0, 0.0, 0.0)
    rotated = rotate_3d(π/4) * point_3d
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000028
begin
    large_array = rand(1000, 1000)

    # Copying vs viewing
    function process_with_copy(A)
        sub = A[1:100, 1:100]  # Creates a copy
        return sum(sub)
    end

    function process_with_view(A)
        sub = @view A[1:100, 1:100]  # Creates a view
        return sum(sub)
    end
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000029
with_terminal() do
    println("Processing with copy:")
    @btime process_with_copy($large_array)
    println("\nProcessing with view:")
    @btime process_with_view($large_array)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000033
begin
    # Example: Circular buffer array
    struct CircularArray{T} <: AbstractVector{T}
        data::Vector{T}
        start::Ref{Int}

        CircularArray{T}(n::Int) where T = new(Vector{T}(undef, n), Ref(1))
    end

    Base.size(c::CircularArray) = size(c.data)
    Base.getindex(c::CircularArray, i::Int) = c.data[mod1(c.start[] + i - 1, length(c.data))]

    function Base.setindex!(c::CircularArray, v, i::Int)
        c.data[mod1(c.start[] + i - 1, length(c.data))] = v
    end

    function rotate!(c::CircularArray, n::Int)
        c.start[] = mod1(c.start[] + n, length(c.data))
    end

    # Create and use circular array
    circ = CircularArray{Int}(5)
    for i in 1:5
        circ[i] = i
    end

    md"""
    Circular array example:
    - Initial: [1, 2, 3, 4, 5]
    - After rotating by 2, accessing element 1 gives: $(begin rotate!(circ, 2); circ[1] end)
    """
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000002
md"""
# More Details on Arrays and Matrices

This optional notebook provides deeper insights into working with arrays and matrices in Julia, covering advanced topics for performance optimization and specialized array types.

## Table of Contents
1. Array Memory Layout and Performance
2. Broadcasting and Vectorization
3. Linear Algebra Operations
4. Sparse Matrices
5. Static Arrays for Performance
6. Views and Memory Efficiency
7. Custom Array Types
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000003
md"""
## 1. Array Memory Layout and Performance

Julia arrays are stored in column-major order (like Fortran), which affects performance when iterating.
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000004
begin
    # Column-major vs Row-major iteration
    function col_major_sum(A)
        s = 0.0
        for j in 1:size(A,2)
            for i in 1:size(A,1)
                s += A[i,j]
            end
        end
        return s
    end

    function row_major_sum(A)
        s = 0.0
        for i in 1:size(A,1)
            for j in 1:size(A,2)
                s += A[i,j]
            end
        end
        return s
    end

    A_test = rand(1000, 1000)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000005
md"""
Compare performance of column-major vs row-major access:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000007
md"""
## 2. Broadcasting and Vectorization

Broadcasting allows element-wise operations on arrays of different shapes efficiently.
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000009
md"""
### Fusion of Broadcast Operations

Julia can fuse multiple broadcast operations into a single loop for efficiency:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000012
md"""
## 3. Linear Algebra Operations

Julia provides optimized BLAS operations for linear algebra:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000013
begin
    # Matrix operations
    M1 = rand(100, 100)
    M2 = rand(100, 100)
    v = rand(100)

    # Different ways to perform matrix operations
    function manual_matmul(A, B)
        C = zeros(size(A,1), size(B,2))
        for i in 1:size(A,1)
            for j in 1:size(B,2)
                for k in 1:size(A,2)
                    C[i,j] += A[i,k] * B[k,j]
                end
            end
        end
        return C
    end
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000014
with_terminal() do
    println("Manual matrix multiplication:")
    @btime manual_matmul($M1, $M2)
    println("\nBuilt-in matrix multiplication:")
    @btime $M1 * $M2
    println("\nIn-place multiplication with mul!:")
    C = similar(M1)
    @btime mul!($C, $M1, $M2)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000015
md"""
### Common Linear Algebra Operations
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000016
begin
    # Decompositions
    A_sym = rand(5, 5)
    A_sym = A_sym + A_sym'  # Make symmetric

    # LU decomposition
    lu_fact = lu(A_sym)

    # Eigenvalue decomposition
    eigen_fact = eigen(A_sym)

    # QR decomposition
    qr_fact = qr(A_sym)

    md"""
    - LU decomposition: $(lu_fact.L[1:3,1:3])
    - Eigenvalues: $(round.(eigen_fact.values, digits=3))
    - QR Q matrix: $(round.(qr_fact.Q[1:3,1:3], digits=3))
    """
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000017
md"""
## 4. Sparse Matrices

For matrices with mostly zeros, sparse representations save memory and computation:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000020
md"""
### Sparse Matrix Construction Patterns
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000022
md"""
## 5. Static Arrays for Performance

StaticArrays provide stack-allocated arrays with compile-time known sizes:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000025
md"""
### When to Use Static Arrays

Static arrays are beneficial when:
- Array size is small (typically < 100 elements)
- Size is known at compile time
- Performance is critical
- You want to avoid heap allocations
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000027
md"""
## 6. Views and Memory Efficiency

Views provide a way to work with subarrays without copying data:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000030
md"""
### Reshape and Reinterpret

Reshape and reinterpret provide zero-copy ways to change array structure:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000031
begin
    # Reshape without copying
    vec_data = collect(1:12)
    matrix_view = reshape(vec_data, 3, 4)

    # Reinterpret for type conversion
    int_array = Int32[1, 2, 3, 4]
    byte_view = reinterpret(UInt8, int_array)

    md"""
    Original vector: $vec_data

    Reshaped to 3×4: $matrix_view

    Int32 array: $int_array

    Reinterpreted as bytes: $byte_view
    """
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000032
md"""
## 7. Custom Array Types

Julia allows defining custom array types for specialized behavior:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000034
md"""
## Performance Tips Summary

1. **Memory Layout**: Iterate in column-major order for best cache performance
2. **Broadcasting**: Use `.` operators and `@.` macro for fused operations
3. **Preallocate**: Use `similar()` or preallocate arrays when possible
4. **In-place Operations**: Use functions ending in `!` (like `mul!`) to avoid allocations
5. **Static Arrays**: Use for small, fixed-size arrays in performance-critical code
6. **Views**: Use `@view` to avoid unnecessary copies
7. **Sparse Matrices**: Use for matrices with many zeros
8. **Type Stability**: Ensure functions return consistent types

## Benchmarking Best Practices

Always use `@btime` with interpolation (`$`) for accurate benchmarks:
```julia
# Good
@btime sum($my_array)

# Bad (includes compilation time)
@btime sum(my_array)
```
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000035
md"""
## Exercises

1. **Memory Layout**: Write a function that transposes a matrix efficiently by reading in column-major order

2. **Broadcasting**: Implement a normalized exponential function using broadcasting:
   - Take a vector `x`
   - Compute `exp.(x) ./ sum(exp.(x))`
   - Compare with a loop-based version

3. **Sparse Matrices**: Create a sparse representation of a finite difference operator for the 1D heat equation

4. **Static Arrays**: Implement a particle simulation where each particle has position and velocity as `SVector{3}`

5. **Views**: Process a large image by dividing it into tiles using views, applying a filter to each tile
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000036
md"""
## Additional Resources

- [Julia Arrays Documentation](https://docs.julialang.org/en/v1/manual/arrays/)
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [LinearAlgebra Standard Library](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
- [SparseArrays Documentation](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
"""

# ╔═╡ Cell order:
# ╟─1a2b3c4d-0000-0000-0000-000000000001
# ╟─1a2b3c4d-0000-0000-0000-000000000002
# ╟─1a2b3c4d-0000-0000-0000-000000000003
# ╠═1a2b3c4d-0000-0000-0000-000000000004
# ╟─1a2b3c4d-0000-0000-0000-000000000005
# ╠═1a2b3c4d-0000-0000-0000-000000000006
# ╟─1a2b3c4d-0000-0000-0000-000000000007
# ╠═1a2b3c4d-0000-0000-0000-000000000008
# ╟─1a2b3c4d-0000-0000-0000-000000000009
# ╠═1a2b3c4d-0000-0000-0000-000000000010
# ╠═1a2b3c4d-0000-0000-0000-000000000011
# ╟─1a2b3c4d-0000-0000-0000-000000000012
# ╠═1a2b3c4d-0000-0000-0000-000000000013
# ╠═1a2b3c4d-0000-0000-0000-000000000014
# ╟─1a2b3c4d-0000-0000-0000-000000000015
# ╠═1a2b3c4d-0000-0000-0000-000000000016
# ╟─1a2b3c4d-0000-0000-0000-000000000017
# ╠═1a2b3c4d-0000-0000-0000-000000000018
# ╠═1a2b3c4d-0000-0000-0000-000000000019
# ╟─1a2b3c4d-0000-0000-0000-000000000020
# ╠═1a2b3c4d-0000-0000-0000-000000000021
# ╟─1a2b3c4d-0000-0000-0000-000000000022
# ╠═1a2b3c4d-0000-0000-0000-000000000023
# ╠═1a2b3c4d-0000-0000-0000-000000000024
# ╟─1a2b3c4d-0000-0000-0000-000000000025
# ╠═1a2b3c4d-0000-0000-0000-000000000026
# ╟─1a2b3c4d-0000-0000-0000-000000000027
# ╠═1a2b3c4d-0000-0000-0000-000000000028
# ╠═1a2b3c4d-0000-0000-0000-000000000029
# ╟─1a2b3c4d-0000-0000-0000-000000000030
# ╠═1a2b3c4d-0000-0000-0000-000000000031
# ╟─1a2b3c4d-0000-0000-0000-000000000032
# ╠═1a2b3c4d-0000-0000-0000-000000000033
# ╟─1a2b3c4d-0000-0000-0000-000000000034
# ╟─1a2b3c4d-0000-0000-0000-000000000035
# ╟─1a2b3c4d-0000-0000-0000-000000000036
