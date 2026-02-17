using TriShellFiniteElement
using FerriteViz, GLMakie

function create_cook_grid(nx, ny)
    corners = [Tensors.Vec{2}((0.0, 0.0)),
               Tensors.Vec{2}((48.0, 44.0)),
               Tensors.Vec{2}((48.0, 60.0)),
               Tensors.Vec{2}((0.0, 44.0))]
    grid = generate_grid(Triangle, (nx, ny), corners)
    # facesets for boundary conditions
    addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    return grid
end

# Grid, dofhandler, boundary condition
n = 50
grid = create_cook_grid(n, n)
# a = FerriteViz.wireframe(grid,markersize=10,strokewidth=2)

# interpolation order
ip = Lagrange{RefTriangle,1}() #to define fields only
ip3 = TriShellFiniteElement.IP3()
ip6 = TriShellFiniteElement.IP6()
qr1 = QuadratureRule{RefTriangle}(1)
qr3 = QuadratureRule{RefTriangle}(2)
facet_qr = FacetQuadratureRule{RefTriangle}(3)

# cell and face value for u
cv = CellValues(qr1, ip3, ip3)
fv = FacetValues(facet_qr, ip^3)

# degrees of freedom for displacements and rotations
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
close!(dbc)

# integrate the traction force
function traction_force_vector!(cell, fv, traction)
    fe = zeros(15)
    n_basefuncs = getnbasefunctions(fv)
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) ∈ getfacetset(grid, "traction")
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:n_basefuncs
                    δu = shape_value(fv, q_point, i)
                    # thickness is applied here
                    fe[i] += (δu ⋅ traction) * dΓ
                end
            end
        end
    end
    return fe
end

# explicit assembly of the stifness and force vector
function assemble_shell!(K, F, dh, qr1, qr3, ip3, ip6, fqr, E, ν, t, traction)
    assembler = start_assemble(K, F)
    for cell in CellIterator(dh)
        x = getcoordinates(cell)
        ke = TriShellFiniteElement.elastic_stiffness_matrix!(qr1, qr3, ip3, ip6, E, ν, t, x)
        fe = traction_force_vector!(cell, fv, traction)
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, F
end

# material properties
E = 0.7 # 70 Pa in N/dm^2
t = 0.5
ν = 0.3333

# Assembly and solve
Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))

# traction vector in units N/dm/thickness
traction = Tensors.Vec{3}((0.0, 1/16*t, 0))

# assemble the system
Ke, f = assemble_shell!(Ke, f, dh, qr1, qr3, ip3, ip6, facet_qr, E, ν, t, traction)
# apply the BCs
apply!(Ke, f, dbc)
# solve
u = Ke \ f

# plot the displacement field
plotter = FerriteViz.MakiePlotter(dh, u)
FerriteViz.solutionplot(plotter,field=:u)

# VTKGridFile("mindlin_shell", dh) do vtk
    # write_solution(vtk, dh, u)
# end