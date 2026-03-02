using TriShellFiniteElement
using Ferrite


# maps the 2D nodes of a mesh onto the 3D coordinates
# by applying the `map` function to the nodes (default: flat z=0 plane)
function ShellMesh(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes])
end

function create_cook_grid(nx, ny)
    corners = [Tensors.Vec{2}((0.0,  0.0)),
               Tensors.Vec{2}((48.0, 44.0)),
               Tensors.Vec{2}((48.0, 60.0)),
               Tensors.Vec{2}((0.0,  44.0))]
    return generate_grid(Triangle, (nx, ny), corners) |> ShellMesh
end

# assemble element stiffness matrices into K
function assemble_shell!(K, dh, scv_mb, scv_s, E, ν, t)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(scv_mb, cell)
        reinit!(scv_s,  cell)
        ke = TriShellFiniteElement.elastic_stiffness_matrix(scv_mb, scv_s, E, ν, t)
        Te = TriShellFiniteElement.rotation_matrix_for_element_stiffness(scv_mb.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return K
end

# integrate edge traction force into f
# DOF ordering assumed: :u field first (3 DOFs per node, interleaved u,v,w),
# :θ field second.  This matches Ferrite's ordering when fields are added in that order.
function assemble_traction_force!(f, dh, facetset, traction)
    edge_local_nodes = Ferrite.reference_facets(RefTriangle)  # ((1,2),(2,3),(3,1))
    n_dpc = ndofs_per_cell(dh)
    fe    = zeros(n_dpc)
    for fc in FacetIterator(dh, facetset)
        x  = getcoordinates(fc)
        fn = fc.current_facet_id             # local facet index: 1, 2, or 3
        ia, ib = edge_local_nodes[fn]        # local node indices on this edge
        edge_len = norm(x[ib] - x[ia])
        fill!(fe, 0.0)
        # 1-point midpoint quadrature: both edge nodes receive equal weight 0.5
        # (exact for the linear shape functions used here)
        for (node, N) in ((ia, 0.5), (ib, 0.5))
            for c in 1:3    # u, v, w components
                fe[3(node-1)+c] += N * traction[c] * edge_len
            end
        end
        f[celldofs(fc)] .+= fe
    end
    return f
end


# ── Grid, DofHandler, boundary conditions ──────────────────────────────────────

n    = 50
grid = create_cook_grid(n, n)

addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
addfacetset!(grid, "traction", x -> x[1] ≈ 48.0)

ip  = Lagrange{RefTriangle,1}()
ip3 = Lagrange{RefTriangle,1}()
ip6 = Lagrange{RefTriangle,2}()

qr1 = QuadratureRule{RefTriangle}(1)
qr2 = QuadratureRule{RefTriangle}(2)

scv_mb = ShellCellValues(qr1, ip3, ip3)
scv_s  = ShellCellValues(qr2, ip3, ip6)

dh = DofHandler(grid)
add!(dh, :u, ip^3)   # translational DOFs (u, v, w)
add!(dh, :θ, ip^2)   # rotational DOFs (θx, θy)
close!(dh)

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x),   [1, 2, 3]))
add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> [0.0, 0.0], [1, 2]))
close!(dbc)

# ── Material and loading ────────────────────────────────────────────────────────

E = 0.7        # stiffness (N/dm²)
t = 0.5        # thickness (dm)
ν = 0.3333

# traction in N/dm/thickness; right edge height = 60 - 44 = 16 → total force = 1 N
traction = Tensors.Vec{3}((0.0, 1/16, 0.0))

# ── Assembly and solve ─────────────────────────────────────────────────────────

Ke = allocate_matrix(dh)
f  = zeros(ndofs(dh))

assemble_shell!(Ke, dh, scv_mb, scv_s, E, ν, t)
assemble_traction_force!(f, dh, getfacetset(grid, "traction"), traction)

apply!(Ke, f, dbc)
@time u = Ke \ f

# ── Post-processing ────────────────────────────────────────────────────────────

ph     = PointEvalHandler(grid, [Tensors.Vec{3}((48.0, 60.0, 0.0))])
u_eval = first(evaluate_at_points(ph, dh, u, :u))
@show u_eval

VTKGridFile("mindlin_shell", dh) do vtk
    write_solution(vtk, dh, u)
end
