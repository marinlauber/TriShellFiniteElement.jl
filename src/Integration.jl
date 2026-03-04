
"""
    calculate_element_membrane_stiffness_matrix(D, scv::ShellCellValues)
"""
function calculate_element_membrane_stiffness_matrix(D, scv::ShellCellValues)
    n  = getnbasefunctions(scv.ip_geo)
    ke = zeros(2n, 2n)
    for q in eachindex(scv.detJdV)
        B = hcat(ntuple(n) do i
            dx, dy = scv.∇N[q, i]
            [dx   0.0
             0.0  dy
             dy   dx]
        end...)
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end

"""
    calculate_element_bending_stiffness_matrix(D, scv::ShellCellValues)
"""
function calculate_element_bending_stiffness_matrix(D, scv::ShellCellValues)
    n_geo = getnbasefunctions(scv.ip_geo)
    ke = zeros(4n_geo, 4n_geo)
    for q in eachindex(scv.detJdV)
        B = hcat(
            ntuple(n_geo) do i
                dx, dy = scv.∇N[q, i]
                [0.0  0.0   dx
                 0.0  -dy   0.0
                 0.0  -dx   dy]
            end..., zeros(3, 3),   # padding for the 3 bubble nodes in the shear element beofre was zero(3,9)
        )
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end

"""
    calculate_element_shear_stiffness_matrix(D, scv::ShellCellValues)
"""
function calculate_element_shear_stiffness_matrix(D, scv::ShellCellValues)
    n_shape = size(scv.N, 2)
    n_geo   = getnbasefunctions(scv.ip_geo)
    ke = zeros(3n_shape, 3n_shape)
    for q in eachindex(scv.detJdV)
        ξ = scv.qr.points[q]
        B = hcat(map(1:n_shape) do i
            dx, dy = scv.∇N[q, i]
            B_node = [dx   0.0   0.0
                      dy   0.0   0.0]
            if i ≤ n_geo   # corner nodes carry θ; use ip_geo (P1) for θ interpolation
                N = Ferrite.reference_shape_value(scv.ip_geo, ξ, i)
                B_node += [0.0   0.0   -N
                           0.0   N     0.0]
            end
            B_node
        end...)
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end

# # in place integrator
# function bending_terms!(Bb, scv::ShellCellValues, qp)
#     fill!(Bb, 0.0)
#     @inbounds for i in 1:getnbasefunctions(scv.ip_shape)
#         dx, dy = scv.∇N[qp, i]
#         col = (i-1)*5
#         # κxx
#         Bb[1,col+4] = dx
#         # κyy
#         Bb[2,col+5] = dy
#         # κxy
#         Bb[3,col+4] = dy
#         Bb[3,col+5] = dx
#     end
# end
# function membrane_terms!(Bm, scv::ShellCellValues, qp)
#     fill!(Bm, 0.0)
#     @inbounds for i in 1:getnbasefunctions(scv.ip_shape) # check this
#         dx, dy = scv.∇N[qp, i]
#         col = (i-1)*5
#         # εxx
#         Bm[1,col+1] = dx
#         # εyy
#         Bm[2,col+2] = dy
#         # γxy
#         Bm[3,col+1] = dy
#         Bm[3,col+2] = dx
#     end
# end
# function shear_terms!(Bs, scv::ShellCellValues, qp)
#     fill!(Bs, 0.0)
#     for i in 1:getnbasefunctions(scv.ip_geo)
#         ξ = scv.qr.points[qp]
#         N = Ferrite.reference_shape_value(scv.ip_shape, ξ, i)
#         dx, dy = scv.∇N[qp, i]
#         col = (i-1)*5
#         # γxz
#         Bs[1,col+3] = dx
#         Bs[1,col+4] = -N
#         # γyz
#         Bs[2,col+3] = dy
#         Bs[2,col+5] = -N
#     end
# end

"""
    elastic_stiffness_matrix(scv_mb::ShellCellValues, scv_s::ShellCellValues, E, ν, t)

scv_mb : ShellCellValues for membrane + bending
         (e.g. qr1, Lagrange{RefTriangle,1} geometry, Lagrange{RefTriangle,1} shape)
scv_s  : ShellCellValues for shear
         (e.g. qr2, Lagrange{RefTriangle,1} geometry, Lagrange{RefTriangle,2} shape)
Both must be reinit!-ed before calling this function.
"""
function elastic_stiffness_matrix(scv_mb::ShellCellValues, scv_s::ShellCellValues, E, ν, t)
    ke_m = calculate_element_membrane_stiffness_matrix(
        calculate_membrane_constitutive_matrix(E, ν, t), scv_mb)
    ke_b = calculate_element_bending_stiffness_matrix(
        calculate_bending_constitutive_matrix(E, ν, t), scv_mb)
    ke_s = calculate_element_shear_stiffness_matrix(
        calculate_shear_constitutive_matrix(E, ν, t), scv_s)

    # Remove zero rows/cols (zero w rows in the padded 18×18 bending matrix)
    idx   = [1:10; 13; 16]
    ke_bs = ke_b + ke_s[idx, idx]

    # Static condensation: eliminate bubble w DOFs (nodes 4–6 → indices 10–12)
    inda  = 1:9; indi  = 10:12
    ke_bs = ke_bs[inda, inda] - ke_bs[inda, indi] * (ke_bs[indi, indi] \ ke_bs[indi, inda])

    # Reindex to Ferrite field ordering:
    # (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
    # maybe we can do that in the assembler only, not here
    ke = zeros(Float64, 15, 15)
    I=[1,2,4,5,7,8]; J=[3,10,11,6,12,13,9,14,15]
    ke[I, I] = ke_m
    ke[J, J] = ke_bs
    return ke
end

# function elastic_stiffness_matrix(scv_mb::ShellCellValues, scv_s::ShellCellValues, E, ν, t)
#     ndofs = 15 #TODO hard-coded for now
#     ke = zeros(ndofs, ndofs)

#     Dm = calculate_membrane_constitutive_matrix(E,ν,t)
#     # Db = calculate_bending_constitutive_matrix(E,ν,t)
#     # Ds = calculate_shear_constitutive_matrix(E,ν,t)

#     Bm = zeros(3,ndofs)
#     Bb = zeros(3,ndofs)
#     # Bs = zeros(2,ndofs)

#     # tmp3 = zeros(3,ndofs)
#     # tmp2 = zeros(2,ndofs)

#     for qp in 1:getnquadpoints(scv_mb.qr)

#         dA = getdetJdV(scv_mb,qp)

#         membrane_terms!(Bm,scv_mb,qp)
#         bending_terms!(Bb,scv_mb,qp)
#         # shear_terms!(Bs,scv_s,qp)

#         # Membrane
#         mul!(tmp3,Dm,Bm)
#         mul!(ke,Bm',tmp3,dA,1.0)

#         # Bending
#         # mul!(tmp3,Db,Bb)
#         # mul!(ke,Bb',tmp3,dA,1.0)
#     # end
#     # # @inbounds for qp in 1:getnquadpoints(scv_s.qr) # not the same
#     #     # Shear
#     #     mul!(tmp2,Ds,Bs)
#     #     mul!(ke,Bs',tmp2,dA,1.0)
#     end

#     return ke
# end

"""
    calculate_element_geometric_stiffness_matrix(scv::ShellCellValues, σ)
"""
function calculate_element_geometric_stiffness_matrix(scv::ShellCellValues, σ)
    n_geo = getnbasefunctions(scv.ip_geo)
    kg = zeros(15, 15)
    for q in eachindex(scv.detJdV)
        # G matrices: row k = ∂(displacement component k)/∂(x or y), for all 15 DOFs.
        # Ferrite DOF ordering: (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1,...)
        # Only the translational DOFs 1–3*n_geo are non-zero.
        Nuvw_x = zeros(3, 15)
        Nuvw_y = zeros(3, 15)
        for i in 1:n_geo
            dx, dy = scv.∇N[q, i]
            for k in 1:3   # u, v, w components
                Nuvw_x[k, 3(i-1)+k] = dx
                Nuvw_y[k, 3(i-1)+k] = dy
            end
        end
        GGx  = Nuvw_x' * Nuvw_x
        GGy  = Nuvw_y' * Nuvw_y
        GGxy = Nuvw_x' * Nuvw_y + Nuvw_y' * Nuvw_x
        kg  += (σ[1] * GGx + σ[2] * GGy + σ[3] * GGxy) * scv.detJdV[q]
    end
    return kg
end


"""
    rotation_matrix_for_element_stiffness(T3)

T3 is the 3×3 local frame matrix from ShellCellValues. Returns a 15×15 block-diagonal rotation matrix to rotate the element stiffness matrix
from the local frame to the global frame. The 3×3 rotation is applied to each of the three nodes for the u,v,w DOFs, and the 2×2 rotation (T3
with the last row/col removed) is applied to the θ DOFs.
"""
function rotation_matrix_for_element_stiffness(T3::AbstractMatrix)
    T2 = T3[1:2, 1:2]
    Te = Matrix(1.0I, 15, 15)
    for ind in (1:3, 4:6, 7:9)
        Te[ind, ind] = T3
    end
    for ind in (10:11, 12:13, 14:15)
        Te[ind, ind] = T2
    end
    return Te
end

"""
    assemble_global_Ke!(Ke, dh, scv_mb::ShellCellValues, scv_s::ShellCellValues, E, ν, t)

Assemble the global stiffness matrix Ke using the provided scv for membrane/bending and shear.
"""
function assemble_global_Ke!(Ke, dh, scv_mb::ShellCellValues, scv_s::ShellCellValues, E, ν, t)
    assembler = start_assemble(Ke)
    for cell in CellIterator(dh)
        x = getcoordinates(cell)
        reinit!(scv_mb, x)
        reinit!(scv_s,  x)
        ke = elastic_stiffness_matrix(scv_mb, scv_s, E, ν, t)
        Te = rotation_matrix_for_element_stiffness(scv_mb.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return Ke
end

"""
    assemble_global_Kg!(Kg, dh, scv_mb::ShellCellValues, σ_global)
"""
function assemble_global_Kg!(Kg, dh, scv_mb::ShellCellValues, σ_global)
    assembler = start_assemble(Kg)
    for (i, cell) in enumerate(CellIterator(dh))
        x = getcoordinates(cell)
        reinit!(scv_mb, x)
        T = scv_mb.local_frame

        # Rotate stress tensor from global to local coordinates
        str_mat_global = [σ_global[i][1]  σ_global[i][3]
                          σ_global[i][3]  σ_global[i][2]]
        str_mat_local  = T[1:2, 1:2]' * str_mat_global * T[1:2, 1:2]
        σ_local        = [str_mat_local[1,1], str_mat_local[2,2], str_mat_local[1,2]]

        kg = calculate_element_geometric_stiffness_matrix(scv_mb, σ_local)
        Te = rotation_matrix_for_element_stiffness(T)
        assemble!(assembler, celldofs(cell), Te * kg * Te')
    end
    return Kg
end
