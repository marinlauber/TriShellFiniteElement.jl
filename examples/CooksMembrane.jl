using TriShellFiniteElement
using Ferrite

function create_cook_grid(nx, ny)
    corners = [Tensors.Vec{2}((0.0,  0.0)),
               Tensors.Vec{2}((48.0, 44.0)),
               Tensors.Vec{2}((48.0, 60.0)),
               Tensors.Vec{2}((0.0,  44.0))]
    return generate_grid(Triangle, (nx, ny), corners) |> ShellMesh
end

# assemble element stiffness matrices into K
function assemble_shell!(K, dh, scv_mb, scv_s, E, ν, t, case=1)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(scv_mb, cell)
        reinit!(scv_s,  cell)
        if case==1
            ke = TriShellFiniteElement.elastic_stiffness_matrix(scv_mb, scv_s, E, ν, t)
        elseif case==2
            ke = TriShellFiniteElement.elastic_stiffness_matrix_MITC3(scv_mb, E, ν, t)
        else
            ke = TriShellFiniteElement.elastic_stiffness_matrix_MITC3plus(scv_mb, E, ν, t)
        end
        Te = TriShellFiniteElement.rotation_matrix_for_element_stiffness(scv_mb.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return nothing
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
    return nothing
end


function s_norm(u, uₕ)
    # K.J - Bath https://doi.org/10.1016/S0045-7949(03)00010-5

end

# ── Grid, DofHandler, boundary conditions ──────────────────────────────────────

function solve(n=16;case=1)
    # make a grid
    grid = create_cook_grid(n, n)
    addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
    addfacetset!(grid, "traction", x -> x[1] ≈ 48.0)

    # interpolation
    ip = Lagrange{RefTriangle,1}()
    ip_s = Lagrange{RefTriangle,2}()
    # quadrature for membrane/bending and shear
    qr1 = QuadratureRule{RefTriangle}(1)
    qr2 = QuadratureRule{RefTriangle}(2)

    # membrane/bending and shear ShellCellValues
    scv_mb = ShellCellValues(qr1, ip, ip)
    scv_s  = ShellCellValues(qr2, ip, ip_s)

    # set degrees of freedom
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # translational DOFs (u, v, w)
    add!(dh, :θ, ip^2)   # rotational DOFs (θx, θy)
    close!(dh)

    # apply boundary conditions
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x),   [1, 2, 3]))
    add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> [0.0, 0.0], [1, 2]))
    close!(dbc)

    # 40 Y. Ko et al. / Computers and Structures 192 (2017) 34–49 http://dx.doi.org/10.1016/j.compstruc.2017.07.003
    E = 1.0        # stiffness (N)
    t = 0.5        # thickness (dm)
    ν = 1.0/3.0

    # traction in N/dm/thickness; right edge height = 60 - 44 = 16 → total force = 1 N
    traction = Tensors.Vec{3}((0.0, 1/16*t, 0.0))

    # assemble and solve
    Ke = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    assemble_shell!(Ke, dh, scv_mb, scv_s, E, ν, t, case)
    assemble_traction_force!(f, dh, getfacetset(grid, "traction"), traction)

    apply!(Ke, f, dbc)
    @time u = Ke \ f

    # extract solution at point
    ph     = PointEvalHandler(grid, [Tensors.Vec{3}((48.0, 60.0, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, u, :u))
end

# make a exact solution
u_e = solve(256;case=1)
# run some discretization
error = []
for c in 1:3, n in [8,16,32,64,128]
    u_h = solve(n;case=c)
    push!(error, norm(u_e - u_h) / norm(u_e))
end
using GLMakie

let
    f = Figure(size=(300,400))
    ax = f[1, 1] = Axis(f, xscale=log10, yscale=log10,
                        xlabel="h", ylabel="Eₕ", title="Cook's membrane",
                        xminorticksvisible=true, xminorgridvisible=true,
                        xminorticks=IntervalsBetween(5))
    lines!(48 ./ [8,16,32,64,128], error[1:5], linestyle=:solid, label="P2+condensation")
    lines!(48 ./ [8,16,32,64,128], error[6:10], linestyle=:dash, label="MITC3")
    lines!(48 ./ [8,16,32,64,128], error[11:15], linestyle=:dot, label="MITC3+")
    lines!([0.6,6.0],[0.0001,0.1],linestyle=:dash,color="black",label="ideal scaling")
    xlims!(0.3,6.0); ylims!(0.001,0.1)
    axislegend(ax, position = :rb); f
end

# import Ferrite: function_gradient
# function function_gradient(scv::ShellCellValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))
#     ∇u₁₁ = 0.0; ∇u₁₂ = 0.0
#     ∇u₂₁ = 0.0; ∇u₂₂ = 0.0
#     @inbounds for (i, j) in pairs(dof_range)
#         dx, dy = scv.∇N[qp,i]
#         base = 3*(i-1)
#         u = u[base + 1]
#         v = u[base + 2]
#         ∇u₁₁ += u * dx
#         ∇u₁₂ += u * dy
#         ∇u₂₁ += v * dx
#         ∇u₂₂ += v * dy
#     end
#     return Tensor{2,2,Float64}((∇u₁₁, ∇u₁₂, ∇u₂₁, ∇u₂₂))
# end

# Cijkl = E/(1-ν^2) .* [
#         1.0  ν    0.0
#         ν    1.0  0.0
#         0.0  0.0  (1-ν)/2
#     ]


# σ_qp = zeros(3*getncells(grid))
# for (i,cell) in enumerate(CellIterator(dh))
#     cell_dofs = celldofs(cell)[dof_range(dh, :u)] # only u field is passed
#     ue = u[cell_dofs] # displacements only
#     for qp in 1:getnquadpoints(scv_mb)
#         # dΩ = getdetJdV(scv_mb, qp)
#         ∇u = function_gradient(scv_mb, qp, ue)
#         # deformation gradient
#         F = one(∇u) + ∇u
#         C = tdot(F) # F' ⋅ F
#         # material model
#         Egl = 0.5*(C - one(C))
#         # Voigt notation strain and Voigt Second Piola
#         Evec = [Egl[1,1], Egl[2,2], 2Egl[1,2]]
#         Svec = Cijkl * Evec
#         S = [Svec[1]  Svec[3]
#              Svec[3]  Svec[2]]
#         # Cauchy stress
#         J = det(F)
#         σ_vm = (1/J) * F * S * F'
#         σ_qp[3*(i-1)+1] = σ_vm[1,1]
#         σ_qp[3*(i-1)+2] = σ_vm[2,2]
#         σ_qp[3*(i-1)+3] = σ_vm[1,2]
#     end
# end
# mises_values = zeros(getncells(grid))
# κ_values = zeros(getncells(grid))
# for (el, cell_states) in enumerate(eachcol(states))
#     for state in cell_states
#         mises_values[el] += vonMises(state.σ)
#         κ_values[el] += state.k * material.H
#     end
#     mises_values[el] /= length(cell_states) # average von Mises stress
#     κ_values[el] /= length(cell_states)     # average drag stress
# end

# σ_qp = zeros(3*getncells(grid))
# for (i,cell) in enumerate(CellIterator(dh))
#     σ_qp[3*(i-1)+1] = first(first(sum(getcoordinates(cell), dims=1))/3)
# end

VTKGridFile("mindlin_shell", dh) do vtk
    write_solution(vtk, dh, u)
    # write_cell_data(vtk, σ_qp, "σ")
end

# function compute_stress!(scv::ShellCellValues, ue::Vector{Float64}, E, ν, t)

#     # Plane stress material matrix (no thickness scaling here)
#     C = E/(1-ν^2) .* [
#         1.0  ν    0.0
#         ν    1.0  0.0
#         0.0  0.0  (1-ν)/2
#     ]

#     nqp = length(scv.detJdV)

#     σ_qp = Vector{Matrix{Float64}}(undef, nqp)

#     for qp in 1:nqp

#         ∇u = function_gradient(scv, qp, ue)
#         F = one(I) + ∇u

#         # --------------------------------------------------
#         # Green-Lagrange strain
#         # --------------------------------------------------

#         Cmat = F' * F
#         Egl  = 0.5 * (Cmat - I)

#         # Convert to Voigt
#         Evec = [
#             Egl[1,1]
#             Egl[2,2]
#             2Egl[1,2]
#         ]

#         # --------------------------------------------------
#         # 2nd PK stress
#         # --------------------------------------------------

#         Svec = C * Evec

#         S = [
#             Svec[1]  Svec[3]
#             Svec[3]  Svec[2]
#         ]

#         # --------------------------------------------------
#         # Cauchy stress
#         # --------------------------------------------------

#         J = det(F)
#         σ = (1/J) * F * S * F'

#         σ_qp[qp] = σ
#     end

#     return σ_qp
# end