using TriShellFiniteElement
using Ferrite
using Test

import Ferrite: reference_shape_value, reference_shape_gradient
@testset "TriShellFiniteElement.jl" begin
    # linear
    ip3 = TriShellFiniteElement.IP3()
    @test reference_shape_value(ip3, Vec{2}((0.0, 0.0)), 1) ≈ 1
    @test reference_shape_value(ip3, Vec{2}((1.0, 0.0)), 2) ≈ 1
    @test reference_shape_value(ip3, Vec{2}((0.0, 1.0)), 3) ≈ 1
    @test reference_shape_value(ip3, Vec{2}((0.5, 0.5)), 1) ≈ 0
    @test reference_shape_value(ip3, Vec{2}((0.5, 0.5)), 2) ≈ 0.5
    @test reference_shape_value(ip3, Vec{2}((0.5, 0.5)), 3) ≈ 0.5
    # overloaded functions
    @test Ferrite.getnbasefunctions(ip3) == 3
    @test Ferrite.adjust_dofs_during_distribution(ip3) == false
    # test partition to unity
    ξ = Vec{2}(rand(2))
    @test sum(Ferrite.reference_shape_value(ip3, ξ, i) for i in 1:3) ≈ 1
    # test that out of bounds shape numbers throw an error
    @test_throws ArgumentError Ferrite.reference_shape_value(ip3, ξ, 4) ≈ 0
    # quadratic
    ip6 = TriShellFiniteElement.IP6()
    @test reference_shape_value(ip6, Vec{2}((0.0, 0.0)), 1) ≈ 1
    @test reference_shape_value(ip6, Vec{2}((1.0, 0.0)), 2) ≈ 1
    @test reference_shape_value(ip6, Vec{2}((0.0, 1.0)), 3) ≈ 1
    @test reference_shape_value(ip6, Vec{2}((0.5, 0.0)), 4) ≈ 1
    @test reference_shape_value(ip6, Vec{2}((0.5, 0.5)), 5) ≈ 1
    @test reference_shape_value(ip6, Vec{2}((0.0, 0.5)), 6) ≈ 1
    # test partition to unity
    @test sum(Ferrite.reference_shape_value(ip6, ξ, i) for i in 1:3) ≈ 1
    # test that out of bounds shape numbers throw an error
    @test_throws ArgumentError Ferrite.reference_shape_value(ip6, ξ, 9) ≈ 0
    # overloaded functions
    @test Ferrite.getnbasefunctions(ip6) == 6
    @test Ferrite.adjust_dofs_during_distribution(ip6) == false
    ip3 = TriShellFiniteElement.IP3()
    # linear shape functions have constant gradient in the element
    @test all(Ferrite.reference_shape_gradient(ip3, Vec{2}(rand(2)), 1) .≈ [-1,-1])
    @test all(Ferrite.reference_shape_gradient(ip3, Vec{2}(rand(2)), 2) .≈ [ 1, 0])
    @test all(Ferrite.reference_shape_gradient(ip3, Vec{2}(rand(2)), 3) .≈ [ 0, 1])
    # quadratic have constant gradient at the nodes
    ip6 = TriShellFiniteElement.IP6()
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0,1)), 1) .≈ [-1,-1])
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0,1)), 2) .≈ [ 1, 0])
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0,1)), 3) .≈ [ 0, 1])
    # ∂N₄/∂ξ₁ = 4(1-2ξ₁-ξ₂), ∂N₄/∂ξ₂ = -4ξ₁
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0.5,0.5)), 4) .≈ [-2,-2])
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0.0,0.0)), 4) .≈ [4,0])
    # ∂N₅/∂ξ₁ = 4ξ₂, ∂N₅/∂ξ₂ = 4ξ₁
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0.5,0.5)), 5) .≈ [2,2])
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((1.0,0.0)), 5) .≈ [0,4])
    # ∂N₆/∂ξ₁ = -4ξ₂, ∂N₆/∂ξ₂ = 4(1-ξ₁-2ξ₂)
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0.5,0.5)), 6) .≈ [-2,-2])
    @test all(Ferrite.reference_shape_gradient(ip6, Vec{2}((0.0,0.0)), 6) .≈ [0, 4])

    import TriShellFiniteElement: get_jacobian
    ip = TriShellFiniteElement.IP3()
    x = [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((0.0, 1.0))]
    ξ = Vec{2}((0.5, 0.5))
    jacobian = get_jacobian(ξ, ip, x)
    @test all(jacobian .≈ [1.0 0.0; 0.0 1.0])
    x = [Vec{2}((0.0, 0.0)), Vec{2}((2.0, 0.0)), Vec{2}((0.0, 3.0))]
    jacobian = get_jacobian(ξ, ip, x)
    @test all(jacobian .≈ [2.0 0.0; 0.0 3.0])

    ip = TriShellFiniteElement.IP6()
    x = [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((0.0, 1.0))]
    jacobian = get_jacobian(ξ, ip, x)
    @test all(jacobian .≈ [1.0 0.0; 0.0 1.0])
    x = [Vec{2}((0.0, 0.0)), Vec{2}((2.0, 0.0)), Vec{2}((0.0, 3.0))]
    jacobian = get_jacobian(ξ, ip, x)
    @test all(jacobian .≈ [2.0 0.0; 0.0 3.0])
end

@testset "Special tests" begin
    include("global_to_local_coordinate_testing.jl")
    @test X1 ≈ 0
end