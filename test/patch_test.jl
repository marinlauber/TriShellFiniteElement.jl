using TriShellFiniteElement, Ferrite, LinearAlgebra, Test

# ─── Irregular 5-element patch mesh ──────────────────────────────────────────
#
# 5 boundary nodes at irregular positions + 1 interior node.
# Topology (CCW triangles, normal in +z):
#
#   5(0,1.5)          4(1.5,2)
#     \    ●(0.9,0.7)   /
#      \    6(interior)  /
#   1(0,0) - 2(1,0) - 3(2,0.5)

const PATCH_NODES = [
    Vec{3}((0.0, 0.0, 0.0)),   # 1 — corner
    Vec{3}((1.0, 0.0, 0.0)),   # 2 — corner
    Vec{3}((2.0, 0.5, 0.0)),   # 3 — irregular boundary
    Vec{3}((1.5, 2.0, 0.0)),   # 4 — irregular boundary
    Vec{3}((0.0, 1.5, 0.0)),   # 5 — irregular boundary
    Vec{3}((0.9, 0.7, 0.0)),   # 6 — interior (irregular)
]
const PATCH_CELLS = [(1,2,6), (2,3,6), (3,4,6), (4,5,6), (5,1,6)]

function patch_grid()
    Grid([Triangle(c) for c in PATCH_CELLS], Node.(PATCH_NODES))
end

# Triangle or Quadrilateral patch grid
function patch_grid_2(;primitive=Triangle)
    nodes = [
        Vec{3}(( 0.0,  0.0, 0.0)),
        Vec{3}((10.0,  0.0, 0.0)),
        Vec{3}((10.0, 10.0, 0.0)),
        Vec{3}(( 0.0, 10.0, 0.0)),
        Vec{3}(( 2.0,  2.0, 0.0)),
        Vec{3}(( 8.0,  3.0, 0.0)),
        Vec{3}(( 8.0,  7.0, 0.0)),
        Vec{3}(( 4.0,  7.0, 0.0)),
    ]
    if primitive==Triangle
        cells = [(1,2,5), (2,6,5), (2,3,6), (3,7,6), (3,8,7),
                 (3,4,8), (4,5,8), (4,1,5), (5,6,8), (6,7,8)]
    else
        cells = [(1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8),
                 (5,6,7,8)]
    end
    return Grid([primitive(c) for c in cells], Node.(nodes))
end

function patch_dofhandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # (u, v, w)
    add!(dh, :θ, ip^2)   # (θx, θy)
    close!(dh)
end

# Build node → (u_dofs, θ_dofs) map from celldofs.
function node_dof_map(dh)
    grid = dh.grid
    nd_u = Dict{Int,Vector{Int}}()
    nd_θ = Dict{Int,Vector{Int}}()
    for ci in 1:getncells(grid)
        cd    = celldofs(dh, ci)
        nodes = grid.cells[ci].nodes
        for (li, gn) in enumerate(nodes)
            if !haskey(nd_u, gn)
                nd_u[gn] = cd[3(li-1) .+ (1:3)]      # u,v,w
                nd_θ[gn] = cd[9 + 2(li-1) .+ (1:2)]  # θx,θy
            end
        end
    end
    nd_u, nd_θ
end

# Schur complement solve for patch_grid_2():
#   boundary nodes 1–4 (corners), interior nodes 5–8.
# Prescribes exact BCs at boundary nodes using their actual coordinates,
# solves for all interior DOFs, and checks each interior node.
# u_exact(x) → [u, v, w, θx, θy]  (5-vector)
function run_patch_test(assemble_fn!, u_exact; E=210e3, ν=0.3, t=1.0, atol=5e-5)
    grid = patch_grid_2()
    dh   = patch_dofhandler(grid)

    K = allocate_matrix(dh)
    assemble_fn!(K, dh, E, ν, t)
    Kd = Matrix(K)

    nd_u, nd_θ = node_dof_map(dh)
    # patch_grid_2: nodes 1–4 are the outer corners; nodes 5–8 are interior
    bnd_nodes = 1:4
    int_nodes = 5:getnnodes(grid)

    int_dofs = vcat([[nd_u[n]; nd_θ[n]] for n in int_nodes]...)
    bnd_dofs = setdiff(1:ndofs(dh), int_dofs)

    u_bc = zeros(ndofs(dh))
    for n in bnd_nodes
        x = grid.nodes[n].x
        e = u_exact(x)
        u_bc[nd_u[n]] .= e[1:3]
        u_bc[nd_θ[n]] .= e[4:5]
    end

    u_int = -(Kd[int_dofs, int_dofs] \ (Kd[int_dofs, bnd_dofs] * u_bc[bnd_dofs]))

    for n in int_nodes
        node_dofs = [nd_u[n]; nd_θ[n]]
        idx = [findfirst(==(d), int_dofs) for d in node_dofs]
        x = grid.nodes[n].x
        expected = let e = u_exact(x); [e[1:3]; e[4:5]] end
        @test u_int[idx] ≈ expected atol = atol
    end
end

# ─── Assembly helpers ─────────────────────────────────────────────────────────

function assemble_P2_condensation!(K, dh, E, ν, t)
    scv_mb = ShellCellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,1}(), Lagrange{RefTriangle,1}())
    scv_s  = ShellCellValues(QuadratureRule{RefTriangle}(2), Lagrange{RefTriangle,1}(), Lagrange{RefTriangle,2}())
    TriShellFiniteElement.assemble_global_Ke!(K, dh, scv_mb, scv_s, E, ν, t)
end

function assemble_MITC3!(K, dh, E, ν, t)
    scv = ShellCellValues(QuadratureRule{RefTriangle}(2), Lagrange{RefTriangle,1}(), Lagrange{RefTriangle,1}())
    TriShellFiniteElement.assemble_global_Ke_MITC3!(K, dh, scv, E, ν, t)
end

function assemble_MITC3plus!(K, dh, E, ν, t)
    scv = ShellCellValues(QuadratureRule{RefTriangle}(2), Lagrange{RefTriangle,1}(), Lagrange{RefTriangle,1}())
    TriShellFiniteElement.assemble_global_Ke_MITC3plus!(K, dh, scv, E, ν, t)
end

# ─── Exact fields ─────────────────────────────────────────────────────────────

const ε₀ = 1e-3   # membrane strain amplitude

# Membrane: u = ε₀ x, v = −ν ε₀ y, w = θ = 0
# Strain: ε_xx = ε₀, ε_yy = −ν ε₀, γ_xy = 0 (all constant).
# This field is linear → exactly reproduced by P1 in u, v.
function u_membrane(x; ν=0.3)
    [ε₀*x[1], -ν*ε₀*x[2], 0.0, 0.0, 0.0]
end

# ─── Tests ────────────────────────────────────────────────────────────────────

# NOTE on the bending (constant-curvature) patch test:
#
# Only the P2+condensation formulation passes this test exactly.
# P2+condensation uses constant tying for shear, which satisfies
# B_cov · u_bending = 0 at the element level for the exact nodal values,
# making K_s · u_exact = 0 element-by-element.
#
# Standard MITC3 uses linear assumed strains (tying at A, B, C), which
# eliminates the spurious zero-energy mode but does NOT satisfy K_s · u_exact = 0
# at the element level. On an irregular mesh there is no cancellation in assembly,
# so MITC3/MITC3+ do not pass this exact bending patch test.
# They are locking-free by construction, but that is a convergence property,
# not an exact patch test guarantee.

@testset "Membrane patch test" begin
    for (label, assemble!) in [
            ("P2+condensation", assemble_P2_condensation!),
            ("MITC3",           assemble_MITC3!),
            ("MITC3+",          assemble_MITC3plus!),
        ]
        @testset "$label" begin
            run_patch_test(assemble!, u_membrane)
        end
    end
end

@testset "Bending patch test" begin
    κ₀ = 1e-3   # constant curvature κ_xx
    # w = κ₀/2 x², θy = κ₀ x, θx = 0, u = v = 0
    u_bending(x) = [0.0, 0.0, 0.5κ₀*x[1]^2, 0.0, κ₀*x[1]]
    # Only P2+condensation (constant tying shear) passes exactly; see note above.
    for (label, assemble!) in [
            ("P2+condensation", assemble_P2_condensation!),
        ]
        @testset "$label" begin
            run_patch_test(assemble!, u_bending)
        end
    end
end
