"""
    calculate_element_shear_stiffness_matrix_MITC3(Ds, scv::ShellCellValues)

Ferrite's P1 triangle: N₁=ξ₁, N₂=ξ₂, N₃=1−ξ₁−ξ₂ (nodes at (1,0),(0,1),(0,0)).
Node 3 sits at the ORIGIN (ξ₁=ξ₂=0). Tying points (edge midpoints):
    ξ_A = (0.5, 0.0) — midpoint of edge 3-1  (on the ξ₂=0 edge)
    ξ_B = (0.5, 0.5) — midpoint of edge 1-2
    ξ_C = (0.0, 0.5) — midpoint of edge 2-3  (on the ξ₁=0 edge)

Cartesian shear: γ = J_loc⁻ᵀ · ẽ_cov
where J_loc = [J1·t1 J2·t1; J1·t2 J2·t2] (stored in scv.J_loc after reinit!).
Returns 9×9 matrix; DOF order: (w,θx,θy) per node × 3 nodes.
Standard MITC3 (Lee & Bathe 2004): assumed covariant shear strains vary linearly,
matching the order of the P1 displacement-derived strains:
ẽ_{ξ₁,3}(ξ₁,ξ₂) = (1−ξ₂)·e_{ξ₁,3}^A + ξ₂·e_{ξ₁,3}^B   — linear in ξ₂
ẽ_{ξ₂,3}(ξ₁,ξ₂) = (1−ξ₁)·e_{ξ₂,3}^A + ξ₁·e_{ξ₂,3}^C   — linear in ξ₁
This gives a full-rank element (6 rigid body modes) with correct convergence.
The bending patch test holds in the assembled sense (contributions cancel across
elements) but not element-by-element on general irregular meshes.
"""
function calculate_element_shear_stiffness_matrix_MITC3(Ds, scv::ShellCellValues)
    ip   = scv.ip_geo
    n    = getnbasefunctions(ip)    # 3 for P1 triangle
    ke   = zeros(3n, 3n)

    J    = scv.J_loc
    Jinv = inv(J)

    ξA = Vec{2}((0.5, 0.0));  ξB = Vec{2}((0.5, 0.5));  ξC = Vec{2}((0.0, 0.5))
    NA = ntuple(i -> Ferrite.reference_shape_value(ip, ξA, i), n)
    NB = ntuple(i -> Ferrite.reference_shape_value(ip, ξB, i), n)
    NC = ntuple(i -> Ferrite.reference_shape_value(ip, ξC, i), n)
    dNdξ = ntuple(i -> Ferrite.reference_shape_gradient(ip, ξA, i), n)  # constant for P1

    for q in eachindex(scv.detJdV)
        ξ1, ξ2 = scv.qr.points[q][1], scv.qr.points[q][2]

        B_cov = zeros(2, 3n)
        for i in 1:n
            col  = 3(i-1) + 1
            α1   = (1 - ξ2) * NA[i] + ξ2 * NB[i]   # weight for ẽ_{ξ₁} component
            α2   = (1 - ξ1) * NA[i] + ξ1 * NC[i]   # weight for ẽ_{ξ₂} component
            B_cov[1, col]   = dNdξ[i][1]
            B_cov[2, col]   = dNdξ[i][2]
            B_cov[1, col+1] =  α1 * J[2, 1]
            B_cov[2, col+1] =  α2 * J[2, 2]
            B_cov[1, col+2] = -α1 * J[1, 1]
            B_cov[2, col+2] = -α2 * J[1, 2]
        end

        B_cart = Jinv' * B_cov
        ke    += B_cart' * Ds * B_cart * scv.detJdV[q]
    end
    return ke
end

"""
    Constant-tying variant: assumed covariant strains are CONSTANT over the element,
                ẽ_{ξ₁,3} = e_{ξ₁,3}^A,   ẽ_{ξ₂,3} = e_{ξ₂,3}^C

which are identically zero for any pure-bending mode (element-level patch test).
Trade-off: the mode (θx=x_loc, θy=y_loc, w=0) also has zero assumed shear
(provable algebraically), so the element has a spurious 7th zero eigenvalue.
"""
function calculate_element_shear_stiffness_matrix_constant_tying(Ds, scv::ShellCellValues)
    ip   = scv.ip_geo
    n    = getnbasefunctions(ip)
    ke   = zeros(3n, 3n)

    J    = scv.J_loc
    Jinv = inv(J)

    ξA = Vec{2}((0.5, 0.0));  ξC = Vec{2}((0.0, 0.5))
    NA = ntuple(i -> Ferrite.reference_shape_value(ip, ξA, i), n)
    NC = ntuple(i -> Ferrite.reference_shape_value(ip, ξC, i), n)
    dNdξ = ntuple(i -> Ferrite.reference_shape_gradient(ip, ξA, i), n)

    B_cov = zeros(2, 3n)
    for i in 1:n
        col = 3(i-1) + 1
        B_cov[1, col]   = dNdξ[i][1]
        B_cov[2, col]   = dNdξ[i][2]
        B_cov[1, col+1] =  NA[i] * J[2, 1]
        B_cov[2, col+1] =  NC[i] * J[2, 2]
        B_cov[1, col+2] = -NA[i] * J[1, 1]
        B_cov[2, col+2] = -NC[i] * J[1, 2]
    end

    B_cart = Jinv' * B_cov
    area   = sum(scv.detJdV)
    ke    += B_cart' * Ds * B_cart * area
    return ke
end

"""
    MITC3+: adds the cubic bubble ψ_b = 27·ξ₁·ξ₂·(1−ξ₁−ξ₂) to the rotation field,

which improves bending accuracy for distorted meshes. The two internal bubble
DOFs (θx_b, θy_b) are condensed out at element level.

Returns 11×11 bending matrix; DOF order: (w,θx,θy)×3 nodes + (θx_b, θy_b).
Shear is unchanged from MITC3 (ψ_b = 0 at all tying points).
Use with qr2 or higher (bubble gradient is quadratic → integrand is degree 4).
"""
function calculate_element_bending_stiffness_matrix_MITC3plus(Db, scv::ShellCellValues)
    n_geo = getnbasefunctions(scv.ip_geo)  # 3
    n_dof = 3n_geo + 2                     # 11 (9 corner + 2 bubble rotations)
    Jinv  = inv(scv.J_loc)
    ke    = zeros(n_dof, n_dof)

    for q in eachindex(scv.detJdV)
        ξ1, ξ2 = Tuple(scv.qr.points[q])

        # Bubble shape function gradient in reference coords
        # ψ_b = 27·ξ₁·ξ₂·(1−ξ₁−ξ₂)
        dψb_dξ1 = 27ξ2 * (1 - 2ξ1 - ξ2)
        dψb_dξ2 = 27ξ1 * (1 - ξ1 - 2ξ2)

        # Bubble gradient in local Cartesian frame: ∇ψ_b = J_loc⁻ᵀ · [∂/∂ξ₁; ∂/∂ξ₂]
        dψb_dx, dψb_dy = Jinv' * Vec{2}((dψb_dξ1, dψb_dξ2))

        # 3×11 bending B matrix: [corner nodes (3×9) | bubble (3×2)]
        B = zeros(3, n_dof)
        for i in 1:n_geo
            dx, dy = scv.∇N[q, i]
            col = 3(i-1) + 1
            B[1, col+2] = dx           # κ_xx = ∂θy/∂x
            B[2, col+1] = -dy          # κ_yy = -∂θx/∂y
            B[3, col+1] = -dx          # 2κ_xy: -∂θx/∂x
            B[3, col+2] = dy           #        +∂θy/∂y
        end
        # Bubble contribution (cols 10=θx_b, 11=θy_b)
        B[1, 11]  = dψb_dx             # κ_xx from θy_b
        B[2, 10]  = -dψb_dy            # κ_yy from θx_b
        B[3, 10]  = -dψb_dx            # 2κ_xy from θx_b
        B[3, 11]  = dψb_dy             #       from θy_b

        ke += B' * Db * B * scv.detJdV[q]
    end
    return ke
end

"""
    elastic_stiffness_matrix_MITC3(scv::ShellCellValues, E, ν, t)

Single scv (reinit!-ed); use qr2 (3-point) or higher for exact integration.
Returns 15×15 ke in Ferrite DOF order:
  (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
"""
function elastic_stiffness_matrix_MITC3(scv::ShellCellValues, E, ν, t)
    ke_m  = calculate_element_membrane_stiffness_matrix(
                calculate_membrane_constitutive_matrix(E, ν, t), scv)   # 6×6
    ke_b12 = calculate_element_bending_stiffness_matrix(
                calculate_bending_constitutive_matrix(E, ν, t), scv)    # 12×12
    ke_s  = calculate_element_shear_stiffness_matrix_MITC3(
                calculate_shear_constitutive_matrix(E, ν, t), scv)      # 9×9

    ke_bs = ke_b12[1:9, 1:9] + ke_s   # 9×9: corner nodes only (no bubble padding)

    ke = zeros(Float64, 15, 15)
    Im = [1,2,4,5,7,8]
    Ib = [3,10,11, 6,12,13, 9,14,15]
    ke[Im, Im] = ke_m
    ke[Ib, Ib] = ke_bs
    return ke
end

"""
    elastic_stiffness_matrix_MITC3plus(scv::ShellCellValues, E, ν, t)

MITC3 shear + cubic bubble bending; bubble DOFs condensed at element level.
Single scv (reinit!-ed); use qr2 or higher.
Returns 15×15 ke in the same Ferrite DOF order as MITC3.
"""
function elastic_stiffness_matrix_MITC3plus(scv::ShellCellValues, E, ν, t)
    ke_m = calculate_element_membrane_stiffness_matrix(
                calculate_membrane_constitutive_matrix(E, ν, t), scv)   # 6×6
    ke_b = calculate_element_bending_stiffness_matrix_MITC3plus(
                calculate_bending_constitutive_matrix(E, ν, t), scv)    # 11×11
    ke_s = calculate_element_shear_stiffness_matrix_MITC3(
                calculate_shear_constitutive_matrix(E, ν, t), scv)      # 9×9

    # Embed 9×9 MITC3 shear in 11×11 (bubble θ DOFs have no shear contribution)
    ke_b[1:9, 1:9] += ke_s

    # Static condensation: remove bubble rotation DOFs (indices 10–11)
    ia = 1:9;  ib = 10:11
    ke_bs_c = ke_b[ia, ia] - ke_b[ia, ib] * (ke_b[ib, ib] \ ke_b[ib, ia])

    ke = zeros(Float64, 15, 15)
    Im = [1,2,4,5,7,8]
    Ib = [3,10,11, 6,12,13, 9,14,15]
    ke[Im, Im] = ke_m
    ke[Ib, Ib] = ke_bs_c
    return ke
end

"""
    calculate_element_bending_stiffness_matrix_constant_tying(Db, scv::ShellCellValues)

MITC3 and MITC3+ variants use a single scv (qr2 or higher recommended).
"""
function assemble_global_Ke_MITC3!(Ke, dh, scv::ShellCellValues, E, ν, t)
    assembler = start_assemble(Ke)
    for cell in CellIterator(dh)
        reinit!(scv, getcoordinates(cell))
        ke = elastic_stiffness_matrix_MITC3(scv, E, ν, t)
        Te = rotation_matrix_for_element_stiffness(scv.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return Ke
end

"""
    assemble_global_Ke_MITC3plus!(Ke, dh, scv::ShellCellValues, E, ν, t)
"""
function assemble_global_Ke_MITC3plus!(Ke, dh, scv::ShellCellValues, E, ν, t)
    assembler = start_assemble(Ke)
    for cell in CellIterator(dh)
        reinit!(scv, getcoordinates(cell))
        ke = elastic_stiffness_matrix_MITC3plus(scv, E, ν, t)
        Te = rotation_matrix_for_element_stiffness(scv.local_frame)
        assemble!(assembler, celldofs(cell), Te * ke * Te')
    end
    return Ke
end