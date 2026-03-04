using TriShellFiniteElement
using Ferrite
using LinearAlgebra
using Test

# Common setup
ip3 = Lagrange{RefTriangle, 1}()
ip6 = Lagrange{RefTriangle, 2}()
qr1 = QuadratureRule{RefTriangle}(1)
qr2 = QuadratureRule{RefTriangle}(2)

# Unit right triangle in 3D (z=0 plane), Ferrite node ordering:
# Lagrange{RefTriangle,1}: N1=ξ₁, N2=ξ₂, N3=1-ξ₁-ξ₂
# → node 1 at ξ=(1,0) → (1,0,0), node 2 at ξ=(0,1) → (0,1,0), node 3 at ξ=(0,0) → (0,0,0)
# Gives J1=(1,0,0), J2=(0,1,0), area=0.5, local frame = identity
x_unit = [Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 0.0))]

E = 210e3
ν = 0.3
t = 1.0
G = E / (2 * (1 + ν))

@testset "Membrane constitutive matrix" begin
    D = TriShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, t)
    @test size(D) == (3, 3)
    @test D ≈ D'
    @test D[1, 1] ≈ E * t / (1 - ν^2)
    @test D[2, 2] ≈ E * t / (1 - ν^2)
    @test D[1, 2] ≈ ν * E * t / (1 - ν^2)
    @test D[3, 3] ≈ G * t
    @test D[1, 3] ≈ 0
    @test D[2, 3] ≈ 0
    @test all(eigvals(D) .> 0)

    # scales linearly with thickness
    D2 = TriShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, 2t)
    @test D2 ≈ 2 * D
end

@testset "Bending constitutive matrix" begin
    D = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t)
    D_const = E * t^3 / (12 * (1 - ν^2))
    @test size(D) == (3, 3)
    @test D ≈ D'
    @test D[1, 1] ≈ D_const
    @test D[2, 2] ≈ D_const
    @test D[1, 2] ≈ D_const * ν
    @test D[3, 3] ≈ D_const * (1 - ν) / 2
    @test D[1, 3] ≈ 0
    @test D[2, 3] ≈ 0
    @test all(eigvals(D) .> 0)

    # scales as t³
    D2 = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, 2t)
    @test D2 ≈ 8 * D
end

@testset "Shear constitutive matrix" begin
    D = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)
    @test size(D) == (2, 2)
    @test D ≈ D'
    @test D[1, 1] ≈ 5 / 6 * G * t
    @test D[2, 2] ≈ 5 / 6 * G * t
    @test D[1, 2] ≈ 0
    @test all(eigvals(D) .> 0)

    # scales linearly with thickness
    D2 = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, 2t)
    @test D2 ≈ 2 * D
end

@testset "Membrane stiffness matrix" begin
    scv = ShellCellValues(qr1, ip3, ip3)
    reinit!(scv, x_unit)
    Dm = TriShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, t)
    ke = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm, scv)

    @test size(ke) == (6, 6)
    @test ke ≈ ke'

    λ = eigvals(Symmetric(ke))
    @test all(λ .≥ -1e-8)                                            # positive semi-definite
    tol = 1e-6 * maximum(abs.(λ))
    @test count(abs.(λ) .< tol) == 3                                  # 3 rigid body modes

    # Rigid body modes: Tx, Ty, Rz
    # DOF ordering: (u1,v1, u2,v2, u3,v3); node coords: P1=(1,0), P2=(0,1), P3=(0,0)
    Tx = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    Ty = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    Rz = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]  # u=-y, v=x: P1→(0,1), P2→(-1,0), P3→(0,0)
    @test Tx' * ke * Tx ≈ 0 atol = 1e-10
    @test Ty' * ke * Ty ≈ 0 atol = 1e-10
    @test Rz' * ke * Rz ≈ 0 atol = 1e-10

    # stiffness scales linearly with E
    Dm2 = TriShellFiniteElement.calculate_membrane_constitutive_matrix(2E, ν, t)
    ke2 = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm2, scv)
    @test ke2 ≈ 2 * ke

    # uniform element scaling leaves membrane stiffness unchanged:
    # larger area (×s²) and smaller strain gradients (×1/s²) cancel out exactly
    x_double = [Vec{3}((2.0, 0.0, 0.0)), Vec{3}((0.0, 2.0, 0.0)), Vec{3}((0.0, 0.0, 0.0))]
    scv2 = ShellCellValues(qr1, ip3, ip3)
    reinit!(scv2, x_double)
    ke_double = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm, scv2)
    @test ke_double ≈ ke

    # analytical value: node 1 at (1,0) has ∇N1=(1,0), so ke[1,1] = D[1,1]*area = E/(1-ν²)*t/2
    @test ke[1, 1] ≈ E / (1 - ν^2) * t / 2
end

@testset "Bending stiffness matrix" begin
    scv = ShellCellValues(qr1, ip3, ip3)
    reinit!(scv, x_unit)
    Db = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t)
    ke = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(Db, scv)

    # DOF layout: (w,θx,θy)×3 corner nodes + 3 zero padding slots = 12×12
    @test size(ke) == (12, 12)
    @test ke ≈ ke'
    @test all(eigvals(Symmetric(ke)) .≥ -1e-8)                       # positive semi-definite

    # Rigid body rotations about x and y produce zero bending energy.
    # DOF ordering per node: (w,θx,θy); node coords: P1=(1,0), P2=(0,1), P3=(0,0)
    # Convention 1: γyz = ∂w/∂y + θx → Rx: w=y_i, θx=-1
    Rx = [0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    # Convention 1: γxz = ∂w/∂x - θy → Ry: w=x_i, θy=+1
    Ry = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    @test Rx' * ke * Rx ≈ 0 atol = 1e-8
    @test Ry' * ke * Ry ≈ 0 atol = 1e-8

    # stiffness scales linearly with E
    Db2 = TriShellFiniteElement.calculate_bending_constitutive_matrix(2E, ν, t)
    ke2 = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(Db2, scv)
    @test ke2 ≈ 2 * ke

    # stiffness scales as t³
    Db_t2 = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, 2t)
    ke_t2 = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(Db_t2, scv)
    @test ke_t2 ≈ 8 * ke
end

@testset "Shear stiffness matrix" begin
    scv = ShellCellValues(qr2, ip3, ip6)
    reinit!(scv, x_unit)
    Ds = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)
    ke = TriShellFiniteElement.calculate_element_shear_stiffness_matrix(Ds, scv)

    @test size(ke) == (18, 18)
    @test ke ≈ ke'
    @test all(eigvals(Symmetric(ke)) .≥ -1e-8)                       # positive semi-definite

    # stiffness scales linearly with E
    Ds2 = TriShellFiniteElement.calculate_shear_constitutive_matrix(2E, ν, t)
    ke2 = TriShellFiniteElement.calculate_element_shear_stiffness_matrix(Ds2, scv)
    @test ke2 ≈ 2 * ke
end

@testset "Combined elastic stiffness matrix" begin
    scv_mb = ShellCellValues(qr1, ip3, ip3)
    scv_s  = ShellCellValues(qr2, ip3, ip6)
    reinit!(scv_mb, x_unit)
    reinit!(scv_s,  x_unit)
    ke = TriShellFiniteElement.elastic_stiffness_matrix(scv_mb, scv_s, E, ν, t)

    @test size(ke) == (15, 15)
    @test ke ≈ ke'

    λ = eigvals(Symmetric(ke))
    @test all(λ .≥ -1e-8)                                            # positive semi-definite
    tol = 1e-6 * maximum(abs.(λ))
    @test count(abs.(λ) .< tol) == 6                                  # 6 rigid body modes

    # DOF order after Ferrite reindexing:
    # (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
    # Node coords: P1=(1,0), P2=(0,1), P3=(0,0)
    Tx = [1,0,0, 1,0,0, 1,0,0, 0,0, 0,0, 0,0]                       # translate x
    Ty = [0,1,0, 0,1,0, 0,1,0, 0,0, 0,0, 0,0]                       # translate y
    Tz = [0,0,1, 0,0,1, 0,0,1, 0,0, 0,0, 0,0]                       # translate z
    Rx = [0,0,0, 0,0,1, 0,0,0, -1,0, -1,0, -1,0]                    # w=y_i: P1→0, P2→1, P3→0; θx=-1
    Ry = [0,0,1, 0,0,0, 0,0,0, 0,1, 0,1, 0,1]                       # w=x_i: P1→1, P2→0, P3→0; θy=+1
    Rz = [0,1,0, -1,0,0, 0,0,0, 0,0, 0,0, 0,0]                      # u=-y, v=x: P1→(0,1), P2→(-1,0), P3→(0,0)
    for mode in (Tx, Ty, Tz, Rx, Ry, Rz)
        @test mode' * ke * mode ≈ 0 atol = 1e-6
    end
end

# Rigid body modes shared by MITC3 and MITC3+ tests
# DOF order: (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
# Node coords: P1=(1,0), P2=(0,1), P3=(0,0)
const RBM_Tx = Float64[1,0,0, 1,0,0, 1,0,0, 0,0, 0,0, 0,0]
const RBM_Ty = Float64[0,1,0, 0,1,0, 0,1,0, 0,0, 0,0, 0,0]
const RBM_Tz = Float64[0,0,1, 0,0,1, 0,0,1, 0,0, 0,0, 0,0]
const RBM_Rx = Float64[0,0,0, 0,0,1, 0,0,0, -1,0, -1,0, -1,0]   # w=y_i; θx=-1
const RBM_Ry = Float64[0,0,1, 0,0,0, 0,0,0,  0,1,  0,1,  0,1]   # w=x_i; θy=+1
const RBM_Rz = Float64[0,1,0, -1,0,0, 0,0,0, 0,0,  0,0,  0,0]   # u=-y, v=x

@testset "MITC3 shear stiffness matrix" begin
    scv = ShellCellValues(qr2, ip3, ip3)
    reinit!(scv, x_unit)
    Ds = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)
    ke = TriShellFiniteElement.calculate_element_shear_stiffness_matrix_MITC3(Ds, scv)

    @test size(ke) == (9, 9)
    @test ke ≈ ke'
    @test all(eigvals(Symmetric(ke)) .≥ -1e-10)   # positive semi-definite

    # Rigid body modes in (w,θx,θy)×3 ordering give zero shear energy
    # Tz: w=1, θ=0
    Tz_s = [1,0,0, 1,0,0, 1,0,0]
    # Rx: w=y_i, θx=-1; node coords: y1=0, y2=1, y3=0
    Rx_s = [0,-1,0, 1,-1,0, 0,-1,0]
    # Ry: w=x_i, θy=+1; node coords: x1=1, x2=0, x3=0
    Ry_s = [1,0,1, 0,0,1, 0,0,1]
    for mode in (Tz_s, Rx_s, Ry_s)
        @test mode' * ke * mode ≈ 0 atol = 1e-8
    end

    # scales linearly with E
    Ds2 = TriShellFiniteElement.calculate_shear_constitutive_matrix(2E, ν, t)
    ke2 = TriShellFiniteElement.calculate_element_shear_stiffness_matrix_MITC3(Ds2, scv)
    @test ke2 ≈ 2ke
end

@testset "MITC3 combined stiffness matrix" begin
    scv = ShellCellValues(qr2, ip3, ip3)
    reinit!(scv, x_unit)
    ke = TriShellFiniteElement.elastic_stiffness_matrix_MITC3(scv, E, ν, t)

    @test size(ke) == (15, 15)
    @test ke ≈ ke'

    λ = eigvals(Symmetric(ke))
    @test all(λ .≥ -1e-8)                                           # positive semi-definite
    tol = 1e-6 * maximum(abs.(λ))
    # Standard MITC3 uses linear assumed strains (tying at A, B, C), matching the order of
    # P1 displacement-derived shear. This eliminates the spurious zero-energy mode present
    # in constant tying, giving exactly 6 zero eigenvalues (the 6 rigid body modes).
    @test count(abs.(λ) .< tol) == 6                                 # exactly 6 RBMs

    for mode in (RBM_Tx, RBM_Ty, RBM_Tz, RBM_Rx, RBM_Ry, RBM_Rz)
        @test mode' * ke * mode ≈ 0 atol = 1e-6
    end
end

@testset "MITC3+ combined stiffness matrix" begin
    scv = ShellCellValues(qr2, ip3, ip3)
    reinit!(scv, x_unit)
    ke = TriShellFiniteElement.elastic_stiffness_matrix_MITC3plus(scv, E, ν, t)

    @test size(ke) == (15, 15)
    @test ke ≈ ke'

    λ = eigvals(Symmetric(ke))
    @test all(λ .≥ -1e-8)                                           # positive semi-definite
    tol = 1e-6 * maximum(abs.(λ))
    # Standard MITC3 shear (linear assumed strains) eliminates the spurious mode,
    # giving exactly 6 zero eigenvalues (the 6 rigid body modes).
    @test count(abs.(λ) .< tol) == 6                                 # exactly 6 RBMs

    for mode in (RBM_Tx, RBM_Ty, RBM_Tz, RBM_Rx, RBM_Ry, RBM_Rz)
        @test mode' * ke * mode ≈ 0 atol = 1e-6
    end
end
