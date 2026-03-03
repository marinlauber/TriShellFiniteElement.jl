module TriShellFiniteElement

using Ferrite, LinearAlgebra, Tensors
import Ferrite: reinit!

include("utils.jl")
export ShellMesh


"""
    ShellCellValues{}

Stores precomputed geometry quantities for a flat triangular shell element.
Works with Vec{3} node coordinates — no manual 2D projection required.
ip_geo and ip_shape can be any Ferrite interpolation (Lagrange, etc.).

On reinit!(scv, x):
   - Builds the 3×2 surface Jacobian from the geometry interpolation
   - Extracts the orthonormal local frame (t1, t2, n) via Gram-Schmidt
   - Projects shape function gradients to the local 2D tangent plane
   - Computes the area-weighted integration weight detJdV
"""
struct ShellCellValues{QR, IPG, IPS, T <: AbstractFloat} <: AbstractCellValues
    qr          :: QR
    ip_geo      :: IPG
    ip_shape    :: IPS
    N           :: Matrix{T}           # shape values       (n_qp × n_shape)
    ∇N          :: Matrix{Vec{2, T}}   # local 2D gradients (n_qp × n_shape)
    detJdV      :: Vector{T}           # area element × weight (n_qp,)
    local_frame :: Matrix{T}           # 3×3 rotation matrix [t1 | t2 | n]
    J_loc       :: Matrix{T}           # 2×2 local Jacobian  [J1·t1 J2·t1; J1·t2 J2·t2]
end
export ShellCellValues

Ferrite.getdetJdV(scv::ShellCellValues, q::Int) = scv.detJdV[q]

function ShellCellValues(qr::QuadratureRule, ip_geo::Interpolation, ip_shape::Interpolation)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    ShellCellValues(
        qr, ip_geo, ip_shape,
        zeros(n_qp, n_shape),
        fill(zero(Vec{2, Float64}), n_qp, n_shape),
        zeros(n_qp),
        zeros(3, 3),
        zeros(2, 2),
    )
end

reinit!(scv::ShellCellValues, cell) = reinit!(scv, getcoordinates(cell))
function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
    n_geo   = getnbasefunctions(scv.ip_geo)
    n_shape = getnbasefunctions(scv.ip_shape)

    for q in eachindex(scv.qr.weights)
        ξ = scv.qr.points[q]

        # Surface Jacobian columns: J1 = ∂x/∂ξ₁, J2 = ∂x/∂ξ₂  (both Vec{3})
        J1 = zero(eltype(x))
        J2 = zero(eltype(x))
        for i in 1:n_geo
            dNdξ = Ferrite.reference_shape_gradient(scv.ip_geo, ξ, i)
            J1  += x[i] * dNdξ[1]
            J2  += x[i] * dNdξ[2]
        end

        # Normal vector and area element (‖J1 × J2‖ = element area scale)
        n_vec         = J1 × J2
        area          = norm(n_vec)
        scv.detJdV[q] = area * scv.qr.weights[q]

        # Orthonormal local frame via Gram-Schmidt
        t1     = J1 / norm(J1)
        n_unit = n_vec / area
        t2     = n_unit × t1

        # Store local frame and local Jacobian from the first quadrature point.
        # For linear geometry J is constant over the element, so this is exact.
        if q == 1
            scv.local_frame[:, 1] .= Tuple(t1)
            scv.local_frame[:, 2] .= Tuple(t2)
            scv.local_frame[:, 3] .= Tuple(n_unit)
            scv.J_loc[1, 1] = J1 ⋅ t1
            scv.J_loc[1, 2] = J2 ⋅ t1
            scv.J_loc[2, 1] = J1 ⋅ t2
            scv.J_loc[2, 2] = J2 ⋅ t2
        end

        # Metric tensor g = JᵀJ (2×2, symmetric)
        g11   = J1 ⋅ J1
        g12   = J1 ⋅ J2
        g22   = J2 ⋅ J2
        det_g = g11 * g22 - g12^2

        # Shape values and local 2D gradients
        for i in 1:n_shape
            scv.N[q, i] = Ferrite.reference_shape_value(scv.ip_shape, ξ, i)
            dNdξ        = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, i)

            # Contravariant gradient components (pseudoinverse: J⁺ = g⁻¹ Jᵀ)
            α1 = (g22 * dNdξ[1] - g12 * dNdξ[2]) / det_g
            α2 = (g11 * dNdξ[2] - g12 * dNdξ[1]) / det_g

            # Physical gradient (Vec{3} in the tangent plane), projected to (t1, t2)
            ∇N_global    = J1 * α1 + J2 * α2
            scv.∇N[q, i] = Vec{2}((∇N_global ⋅ t1, ∇N_global ⋅ t2))
        end
    end
    return nothing
end


# ─── Constitutive matrices ────────────────────────────────────────────────────

function calculate_membrane_constitutive_matrix(E, ν, t)
    G = E / (2 * (1 + ν))
    return [E/(1-ν^2)   ν*E/(1-ν^2)  0.0
            ν*E/(1-ν^2)  E/(1-ν^2)   0.0
            0.0          0.0           G ] .* t
end

function calculate_bending_constitutive_matrix(E, ν, t)
    c = E * t^3 / (12 * (1 - ν^2))
    return c * [1.0   ν    0.0
                ν     1.0  0.0
                0.0   0.0  (1-ν)/2]
end

function calculate_shear_constitutive_matrix(E, ν, t)
    G = E / (2 * (1 + ν))
    κ = 5 / 6
    return [κ*G*t  0.0
            0.0    κ*G*t]
end


# ─── Element stiffness matrices ───────────────────────────────────────────────

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
function membrane_terms!(Bm, scv::ShellCellValues, qp)
    fill!(Bm, 0.0)
    @inbounds for i in 1:getnbasefunctions(scv.ip_shape) # check this
        dx, dy = scv.∇N[qp, i]
        col = (i-1)*5
        # εxx
        Bm[1,col+1] = dx
        # εyy
        Bm[2,col+2] = dy
        # γxy
        Bm[3,col+1] = dy
        Bm[3,col+2] = dx
    end
end

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
            end..., zeros(3, 3),   # padding for the 3 bubble nodes in the shear element
        )
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end
function bending_terms!(Bb, scv::ShellCellValues, qp)
    fill!(Bb, 0.0)
    @inbounds for i in 1:getnbasefunctions(scv.ip_shape)
        dx, dy = scv.∇N[qp, i]
        col = (i-1)*5
        # κxx
        Bb[1,col+4] = dx
        # κyy
        Bb[2,col+5] = dy
        # κxy
        Bb[3,col+4] = dy
        Bb[3,col+5] = dx
    end
end

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
function shear_terms!(Bs, scv::ShellCellValues, qp)
    fill!(Bs, 0.0)
    for i in 1:getnbasefunctions(scv.ip_geo)
        ξ = scv.qr.points[qp]
        N = Ferrite.reference_shape_value(scv.ip_shape, ξ, i)
        dx, dy = scv.∇N[qp, i]
        col = (i-1)*5
        # γxz
        Bs[1,col+3] = dx
        Bs[1,col+4] = -N
        # γyz
        Bs[2,col+3] = dy
        Bs[2,col+5] = -N
    end
end


# ─── MITC3 assumed shear ──────────────────────────────────────────────────────
#
# MITC3: covariant shear strains sampled at 3 edge-midpoint tying points then
# interpolated linearly over the element, eliminating transverse-shear locking.
#
# Ferrite's P1 triangle: N₁=ξ₁, N₂=ξ₂, N₃=1−ξ₁−ξ₂ (nodes at (1,0),(0,1),(0,0)).
# Tying points (edge midpoints in reference coords):
#   ξ_A = (0.5, 0.0) — midpoint of edge 3-1
#   ξ_B = (0.5, 0.5) — midpoint of edge 1-2
#   ξ_C = (0.0, 0.5) — midpoint of edge 2-3
#
# Assumed covariant strains at quadrature point (ξ₁,ξ₂):
#   ẽ_{ξ₁,3} = (1−ξ₂)·e_{ξ₁,3}^A + ξ₂·e_{ξ₁,3}^C
#   ẽ_{ξ₂,3} = ξ₁·e_{ξ₂,3}^B  + (1−ξ₁)·e_{ξ₂,3}^C
#
# Cartesian shear: γ = J_loc⁻ᵀ · ẽ_cov
# where J_loc = [J1·t1 J2·t1; J1·t2 J2·t2] (stored in scv.J_loc after reinit!).
#
# Returns 9×9 matrix; DOF order: (w,θx,θy) per node × 3 nodes.
# Use with qr2 (3-point) or higher for exact integration.

function calculate_element_shear_stiffness_matrix_MITC3(Ds, scv::ShellCellValues)
    ip   = scv.ip_geo
    n    = getnbasefunctions(ip)    # 3 for P1 triangle
    ke   = zeros(3n, 3n)

    J    = scv.J_loc               # 2×2 local Jacobian
    Jinv = inv(J)

    # Shape function values at the three tying points
    ξA = Vec{2}((0.5, 0.0));  ξB = Vec{2}((0.5, 0.5));  ξC = Vec{2}((0.0, 0.5))
    NA = ntuple(i -> Ferrite.reference_shape_value(ip, ξA, i), n)
    NB = ntuple(i -> Ferrite.reference_shape_value(ip, ξB, i), n)
    NC = ntuple(i -> Ferrite.reference_shape_value(ip, ξC, i), n)

    # Reference-space shape gradients (constant for P1, eval at any point)
    dNdξ = ntuple(i -> Ferrite.reference_shape_gradient(ip, ξA, i), n)

    for q in eachindex(scv.detJdV)
        ξ1, ξ2 = Tuple(scv.qr.points[q])

        # 2×(3n) covariant B matrix at (ξ1,ξ2)
        B_cov = zeros(2, 3n)
        for i in 1:n
            col   = 3(i-1) + 1
            Nα1   = (1-ξ2)*NA[i] + ξ2*NC[i]   # interpolation weight for ẽ_{ξ1,3}
            Nα2   = ξ1*NB[i] + (1-ξ1)*NC[i]   # interpolation weight for ẽ_{ξ2,3}

            # w DOF: reference-space gradient
            B_cov[1, col]   = dNdξ[i][1]
            B_cov[2, col]   = dNdξ[i][2]
            # θx DOF: +J_loc[2,α]·Nα  (covariant shear sign convention 1)
            B_cov[1, col+1] =  Nα1 * J[2, 1]
            B_cov[2, col+1] =  Nα2 * J[2, 2]
            # θy DOF: −J_loc[1,α]·Nα
            B_cov[1, col+2] = -Nα1 * J[1, 1]
            B_cov[2, col+2] = -Nα2 * J[1, 2]
        end

        # Covariant → Cartesian: γ = J_loc⁻ᵀ · ẽ_cov
        B_cart = Jinv' * B_cov
        ke    += B_cart' * Ds * B_cart * scv.detJdV[q]
    end
    return ke
end

# ─── MITC3+ bending with cubic bubble ─────────────────────────────────────────
#
# MITC3+: adds the cubic bubble ψ_b = 27·ξ₁·ξ₂·(1−ξ₁−ξ₂) to the rotation field,
# which improves bending accuracy for distorted meshes.  The two internal bubble
# DOFs (θx_b, θy_b) are condensed out at element level.
#
# Returns 11×11 bending matrix; DOF order: (w,θx,θy)×3 nodes + (θx_b, θy_b).
# Shear is unchanged from MITC3 (ψ_b = 0 at all tying points).
# Use with qr2 or higher (bubble gradient is quadratic → integrand is degree 4).

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


# ─── Combined element stiffness ───────────────────────────────────────────────
#
# scv_mb : ShellCellValues for membrane + bending
#          (e.g. qr1, Lagrange{RefTriangle,1} geometry, Lagrange{RefTriangle,1} shape)
# scv_s  : ShellCellValues for shear
#          (e.g. qr2, Lagrange{RefTriangle,1} geometry, Lagrange{RefTriangle,2} shape)
# Both must be reinit!-ed before calling this function.

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

# ─── Combined stiffness: MITC3 ────────────────────────────────────────────────
#
# Single scv (reinit!-ed); use qr2 (3-point) or higher for exact integration.
# Returns 15×15 ke in Ferrite DOF order:
#   (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)

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

# ─── Combined stiffness: MITC3+ ───────────────────────────────────────────────
#
# MITC3 shear + cubic bubble bending; bubble DOFs condensed at element level.
# Single scv (reinit!-ed); use qr2 or higher.
# Returns 15×15 ke in the same Ferrite DOF order as MITC3.

function elastic_stiffness_matrix_MITC3plus(scv::ShellCellValues, E, ν, t)
    ke_m = calculate_element_membrane_stiffness_matrix(
                calculate_membrane_constitutive_matrix(E, ν, t), scv)   # 6×6
    ke_b = calculate_element_bending_stiffness_matrix_MITC3plus(
                calculate_bending_constitutive_matrix(E, ν, t), scv)    # 11×11
    ke_s = calculate_element_shear_stiffness_matrix_MITC3(
                calculate_shear_constitutive_matrix(E, ν, t), scv)      # 9×9

    # Embed 9×9 MITC3 shear in 11×11 (bubble θ DOFs have no shear contribution)
    ke_bs = copy(ke_b)
    ke_bs[1:9, 1:9] += ke_s

    # Static condensation: remove bubble rotation DOFs (indices 10–11)
    ia = 1:9;  ib = 10:11
    ke_bs_c = ke_bs[ia, ia] - ke_bs[ia, ib] * (ke_bs[ib, ib] \ ke_bs[ib, ia])

    ke = zeros(Float64, 15, 15)
    Im = [1,2,4,5,7,8]
    Ib = [3,10,11, 6,12,13, 9,14,15]
    ke[Im, Im] = ke_m
    ke[Ib, Ib] = ke_bs_c
    return ke
end


# ─── Geometric stiffness ──────────────────────────────────────────────────────

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


# ─── Rotation matrix for DOF transformation ───────────────────────────────────

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


# ─── Global assembly ──────────────────────────────────────────────────────────

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

# MITC3 and MITC3+ variants use a single scv (qr2 or higher recommended).
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


end # module TriShellFiniteElement
