# Simply supported square plate under uniform transverse load — convergence study
#
# Geometry: L×L square plate in the z=0 plane, L=1.
# BCs (hard simply supported):
#   w=0 on all four edges
#   θx=0 on edges x=0 and x=L  (tangent-rotation lock on x-normal edges)
#   θy=0 on edges y=0 and y=L  (tangent-rotation lock on y-normal edges)
# Load: uniform transverse (z) pressure q=1.
# Material: E=1, ν=0.3; thickness t varied for thin/thick tests.
#
# Reference (Timoshenko & Woinowsky-Krieger, eq. 10, Table 8):
#   w_max = α·q·L⁴/D,  α = 0.00406,  D = E·t³/(12(1-ν²))
#
# Compares P2+condensation, MITC3, and MITC3+ at N=4,8,16,32 on
# uniform meshes and a randomly distorted mesh.

using TriShellFiniteElement, Ferrite, LinearAlgebra, Printf, Random

# ── Mesh helpers ──────────────────────────────────────────────────────────────

function plate_grid(n; L=1.0)
    corners = [Tensors.Vec{2}((0.0, 0.0)),
               Tensors.Vec{2}((L,   0.0)),
               Tensors.Vec{2}((L,   L  )),
               Tensors.Vec{2}((0.0, L  ))]
    generate_grid(Triangle, (n, n), corners) |> ShellMesh
end

# Perturb interior nodes by a random displacement no larger than pert×h (h=L/n).
function distorted_plate_grid(n; L=1.0, pert=0.4, seed=42)
    grid = plate_grid(n; L=L)
    rng  = MersenneTwister(seed)
    h    = L / n
    tol  = 1e-10 * L
    new_nodes = map(grid.nodes) do node
        x, y, z = Tuple(node.x)
        on_boundary = (x < tol || x > L-tol || y < tol || y > L-tol)
        if on_boundary
            node
        else
            dx = pert * h * (rand(rng) - 0.5) * 2
            dy = pert * h * (rand(rng) - 0.5) * 2
            Node(Tensors.Vec{3}((x+dx, y+dy, z)))
        end
    end
    Grid(grid.cells, new_nodes)
end

# ── Problem setup ─────────────────────────────────────────────────────────────

function plate_dofhandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # (u, v, w)
    add!(dh, :θ, ip^2)   # (θx, θy)
    close!(dh)
    return dh
end

function plate_constraints(dh; L=1.0)
    ch = ConstraintHandler(dh)
    # Soft simply supported: only w = 0 on all four edges.
    # Rotations θx and θy are free → bending moments M_nn = 0 naturally.
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "left"),   x -> 0.0, [3]))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "right"),  x -> 0.0, [3]))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "bottom"), x -> 0.0, [3]))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "top"),    x -> 0.0, [3]))
    # In-plane rigid-body prevention (membrane DOFs are decoupled from bending
    # for a flat plate under vertical load; these do not affect w or θ):
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "left"),   x -> 0.0, [1]))  # u=0 on left
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "bottom"), x -> 0.0, [2]))  # v=0 on bottom
    close!(ch)
    return ch
end

# Assemble uniform transverse load q in the z-direction.
# Uses 1-point quadrature (exact for constant q on flat P1 elements).
function assemble_uniform_load!(f, dh, q)
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(1)
    cv = CellValues(qr, ip, ip^3)
    n_dpc = ndofs_per_cell(dh)
    fe    = zeros(n_dpc)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(fe, 0.0)
        for qp in 1:getnquadpoints(cv)
            dA = getdetJdV(cv, qp)
            for i in 1:getnbasefunctions(ip)
                N = shape_value(cv, qp, i)
                fe[3i] += N * q * dA   # w DOF = 3rd DOF of node i in :u field
            end
        end
        f[celldofs(cell)] .+= fe
    end
    return f
end

# ── Solvers ───────────────────────────────────────────────────────────────────

function solve_plate(assemble_fn!, grid; E=1.0, ν=0.3, t=0.01, q=1.0)
    addfacetset!(grid, "left",   x -> x[1] ≈ 0.0)
    addfacetset!(grid, "right",  x -> x[1] ≈ 1.0)
    addfacetset!(grid, "bottom", x -> x[2] ≈ 0.0)
    addfacetset!(grid, "top",    x -> x[2] ≈ 1.0)

    dh = plate_dofhandler(grid)
    ch = plate_constraints(dh)

    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))

    assemble_fn!(K, dh, E, ν, t)
    assemble_uniform_load!(f, dh, q)

    apply!(K, f, ch)
    u = K \ f

    # Evaluate w at the plate centre (0.5, 0.5, 0.0)
    ph     = PointEvalHandler(grid, [Tensors.Vec{3}((0.5, 0.5, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, u, :u))
    return u_eval[3]   # z-displacement = transverse deflection
end

# ── Assembly wrappers (match patch_test.jl style) ─────────────────────────────

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

# ── Reference solution ────────────────────────────────────────────────────────

function reference_deflection(; E=1.0, ν=0.3, t=0.01, q=1.0, L=1.0)
    D = E * t^3 / (12 * (1 - ν^2))
    α = 0.00406   # Timoshenko & Woinowsky-Krieger, square plate
    return α * q * L^4 / D
end

# ── Convergence study ─────────────────────────────────────────────────────────

function run_convergence(; t=0.01, ns=(4,8,16,32), distorted=false, E=1.0, ν=0.3, q=1.0)
    w_ref = reference_deflection(; E, ν, t, q)
    formulations = [
        ("P2+cond", assemble_P2_condensation!),
        ("MITC3",   assemble_MITC3!),
        ("MITC3+",  assemble_MITC3plus!),
    ]

    tag = distorted ? "distorted" : "uniform"
    println("\n=== Simply supported square plate  t=$(t)  mesh=$(tag) ===")
    println("w_ref = $(@sprintf("%.6e", w_ref))")
    println()
    @printf("%-12s", "N")
    for (label, _) in formulations; @printf("  %-14s", label) end
    println()
    @printf("%-12s", "(elements)")
    for _ in formulations; @printf("  %-14s", "(rel. error)") end
    println()
    println("-"^(12 + 16*length(formulations)))

    for n in ns
        grid_fn = distorted ? () -> distorted_plate_grid(n) : () -> plate_grid(n)
        @printf("%-12d", 2n^2)
        for (_, assemble!) in formulations
            grid = grid_fn()
            w = solve_plate(assemble!, grid; E, ν, t, q)
            rel_err = abs(w - w_ref) / abs(w_ref)
            @printf("  %-14s", @sprintf("%.4e", rel_err))
        end
        println()
    end
    println()
end

# ── Main ──────────────────────────────────────────────────────────────────────

# Thin plate (t/L = 0.01) — shear locking test
run_convergence(t=0.01)
run_convergence(t=0.01, distorted=true)

# Thick plate (t/L = 0.1) — shear-dominated regime
run_convergence(t=0.1)
run_convergence(t=0.1, distorted=true)
