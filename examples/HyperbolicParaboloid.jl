using TriShellFiniteElement,Ferrite,LinearAlgebra,TimerOutputs

E = 200000.0 # MPa
ν = 0.30     # n-a
t = 2.0      # thickness in mm

# domain ω ∈ ]-1/2; 1/2[ and 3D grid
grid = ShellMesh(generate_grid(Ferrite.Triangle, (20, 20),
                               Ferrite.Vec(-0.5, -0.5),
                               Ferrite.Vec(0.5, 0.5)
                              ),
                  map=(n)->(100*n.x[1], 100*n.x[2], 100*(n.x[1]^2-n.x[2]^2)))

# interpolations paces
ip = Lagrange{RefTriangle,1}()
ip_s = Lagrange{RefTriangle,2}()
# quadrature rules
qr1 = QuadratureRule{RefTriangle}(1)
qr2 = QuadratureRule{RefTriangle}(2)

# cell and face value for u
scv_mb = ShellCellValues(qr1, ip, ip) # membrane and bending
scv_s = ShellCellValues(qr2, ip, ip_s) # shear

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# add the Dirichlet Boundary on the faces of the model
addfacetset!(grid, "clamped",  x -> x[1] ≈  50.0)
addfacetset!(grid, "traction", x -> x[1] ≈ -50.0)

# add the boundary condition to the dh
bcs = ConstraintHandler(dh)
add!(bcs, Dirichlet(:u, getfacetset(grid, "clamped"),  (x, t) -> zero(x), [1,2,3]))
add!(bcs, Dirichlet(:θ, getfacetset(grid, "clamped"),  (x, t) -> [0.0,0.0], [1,2]))
close!(bcs)

# Pre-allocation of vectors for the solution and Newton increments
_ndofs = ndofs(dh)
un = zeros(_ndofs) # previous solution vector
u = zeros(_ndofs)
Δu = zeros(_ndofs)
ΔΔu = zeros(_ndofs)
apply!(un, bcs)

# Create sparse matrix and residual vector
K = allocate_matrix(dh)
Ke = allocate_matrix(dh)
g = zeros(_ndofs)

# integrate the traction force
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
                # println("adding ", N * traction[c] * edge_len, " to fe[", 3*(node-1)+c, "]")
                fe[3(node-1)+c] += N * traction[c] * edge_len
            end
        end
        f[celldofs(fc)] .+= fe
    end
    return nothing
end

# explicit assembly of the stiffness and force vector
function assemble_shell!(K, dh, scv_mb, scv_s, E, ν, t)
    assembler = start_assemble(K)
    @timeit "assemble shell" for cell in CellIterator(dh)
        @timeit "reinit! element" (reinit!(scv_mb, cell); reinit!(scv_s, cell))
        @timeit "assemble element" ke = TriShellFiniteElement.elastic_stiffness_matrix(scv_mb, scv_s, E, ν, t)
        Te = TriShellFiniteElement.rotation_matrix_for_element_stiffness(scv_mb.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return nothing
end

function assemble_global_shell!(K, g, u, dh, scv_mb, scv_s, E, ν, t, traction)
    # # assemble the global elastic stiffness matrix
    # assemble_shell!(K, dh, scv_mb, scv_s, E, ν, t)
    # assemble the traction force
    # @timeit "assemble traction" assemble_traction_force!(g, dh, getfacetset(grid, "traction"), traction)
    #  make residual vector
    g .+= K*u
    # geometric stiffness matrix
    # σ_global = [Vec{3}((0,0,0)) for _ in 1:getncells(dh.grid)]
    # @timeit "asseble Kgeom" TriShellFiniteElement.assemble_global_Kg!(K, dh, scv_mb, σ_global)
    # K = Ke # tangent is elastic plus geometric
    # assume the stiffness is the tangent matrix
    return nothing
end


using IterativeSolvers
newton_itr = 0
traction = Ferrite.Vec{3}((0., 0., 40.0*t))

using WriteVTK
pvd = paraview_collection("hyperbolic_paraboloid")
VTKGridFile("hyperbolic_paraboloid-0", dh) do vtk
    Ferrite.write_constraints(vtk, bcs)
    Ferrite.write_facetset(vtk, grid, "clamped")
    Ferrite.write_facetset(vtk, grid, "traction")
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

# only needed once
assemble_shell!(K, dh, scv_mb, scv_s, E, ν, t)
assemble_traction_force!(g, dh, getfacetset(grid, "traction"), traction)
reset_timer!()
@time while true
    global newton_itr += 1
    # Construct the current guess
    u .= un .+ Δu
    # Compute residual and tangent for current guess
    assemble_global_shell!(K, g, u, dh, scv_mb, scv_s, E, ν, t, traction)
    # Apply boundary conditions
    apply_zero!(K, g, bcs)
    # Compute the residual norm and compare with tolerance
    normg = norm(g)
    @show normg
    if normg < 1e-6
        println("converged in $newton_itr iterations")
        break
    elseif newton_itr > 30
        println("Reached maximum Newton iterations, aborting")
        break
    end

    # Compute increment using conjugate gradients
    @timeit "linear solve" IterativeSolvers.cg!(ΔΔu, K, g; maxiter = 1000)

    apply_zero!(ΔΔu, bcs)
    Δu .-= ΔΔu
end
print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)

VTKGridFile("hyperbolic_paraboloid-1", dh) do vtk
    write_solution(vtk, dh, u); pvd[1.0] = vtk
end
vtk_save(pvd)


# cell = collect(CellIterator(dh))[1]
# reinit!(scv_mb, cell)
# reinit!(scv_s, cell)
# ke_m = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(
#     TriShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, t), scv_mb)
# ke_b = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(
#         TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t), scv_mb)
# ke_s = TriShellFiniteElement.calculate_element_shear_stiffness_matrix(
#         TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t), scv_s)

# idx   = [1:10; 13; 16]
# ke_bs = (ke_b + ke_s)[idx, idx]

# inda  = 1:9
# indi  = 10:12
# println(ke_bs[indi, indi])
# ke_bs = ke_bs[inda, inda] - ke_bs[inda, indi] * inv(ke_bs[indi, indi]) * ke_bs[indi, inda]
# ke_bs

# # Assemble into 15×15 using intermediate component ordering:
# # (u1,v1,w1,θx1,θy1, u2,v2,w2,θx2,θy2, u3,v3,w3,θx3,θy3)
# ke = zeros(15, 15)
# induv = [1, 2, 6, 7, 11, 12]
# indwt = [3, 4, 5, 8, 9, 10, 13, 14, 15]
# ke[induv, induv] = ke_m
# ke[indwt, indwt] = ke_bs

# # Reindex to Ferrite field ordering:
# # (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
# ind_field = [1, 2, 3, 6, 7, 8, 11, 12, 13, 4, 5, 9, 10, 14, 15]
# ke[ind_field, ind_field]