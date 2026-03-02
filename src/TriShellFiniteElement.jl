module TriShellFiniteElement

using Ferrite, LinearAlgebra, Tensors
import Ferrite: reinit!

# ─── Custom interpolations ────────────────────────────────────────────────────

struct IP6 <: ScalarInterpolation{RefTriangle, 2}
end

function Ferrite.reference_shape_value(ip::IP6, ξ::Vec{2}, i::Int)
    ξ₁, ξ₂ = ξ[1], ξ[2]
    i == 1 && return 1 - ξ₁ - ξ₂
    i == 2 && return ξ₁
    i == 3 && return ξ₂
    i == 4 && return 4ξ₁ * (1 - ξ₁ - ξ₂)
    i == 5 && return 4ξ₁ * ξ₂
    i == 6 && return 4ξ₂ * (1 - ξ₁ - ξ₂)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

Ferrite.getnbasefunctions(::IP6) = 6
Ferrite.adjust_dofs_during_distribution(::IP6) = false


struct IP3 <: ScalarInterpolation{RefTriangle, 2}
end

function Ferrite.reference_shape_value(ip::IP3, ξ::Vec{2}, i::Int)
    ξ₁, ξ₂ = ξ[1], ξ[2]
    i == 1 && return 1 - ξ₁ - ξ₂
    i == 2 && return ξ₁
    i == 3 && return ξ₂
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

Ferrite.getnbasefunctions(::IP3) = 3
Ferrite.adjust_dofs_during_distribution(::IP3) = false


# ─── ShellCellValues ──────────────────────────────────────────────────────────
#
# Stores precomputed geometry quantities for a flat triangular shell element.
# Works with Vec{3} node coordinates — no manual 2D projection required.
#
# On reinit!(scv, x):
#   - Builds the 3×2 surface Jacobian from the geometry interpolation
#   - Extracts the orthonormal local frame (t1, t2, n) via Gram-Schmidt
#   - Projects shape function gradients to the local 2D tangent plane
#   - Computes the area-weighted integration weight detJdV

mutable struct ShellCellValues{QR, IPG, IPS, T <: AbstractFloat}
    qr          :: QR
    ip_geo      :: IPG
    ip_shape    :: IPS
    N           :: Matrix{T}           # shape values       (n_qp × n_shape)
    ∇N          :: Matrix{Vec{2, T}}   # local 2D gradients (n_qp × n_shape)
    detJdV      :: Vector{T}           # area element × weight (n_qp,)
    local_frame :: Matrix{T}           # 3×3 rotation matrix [t1 | t2 | n]
end

function ShellCellValues(qr, ip_geo, ip_shape)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    ShellCellValues(
        qr, ip_geo, ip_shape,
        zeros(n_qp, n_shape),
        fill(zero(Vec{2, Float64}), n_qp, n_shape),
        zeros(n_qp),
        zeros(3, 3),
    )
end

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

        # Store local frame from the first quadrature point.
        # For linear geometry (IP3) J is constant over the element, so this
        # is exact; for higher-order geometry one would store per-qp frames.
        if q == 1
            scv.local_frame[:, 1] .= Tuple(t1)
            scv.local_frame[:, 2] .= Tuple(t2)
            scv.local_frame[:, 3] .= Tuple(n_unit)
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
            ∇N_global   = J1 * α1 + J2 * α2
            scv.∇N[q, i] = Vec{2}((∇N_global ⋅ t1, ∇N_global ⋅ t2))
        end
    end
    return scv
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
    ke = zeros(6, 6)
    for q in eachindex(scv.detJdV)
        B = hcat(ntuple(3) do i
            dx, dy = scv.∇N[q, i]
            [dx   0.0
             0.0  dy
             dy   dx]
        end...)
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end

function calculate_element_bending_stiffness_matrix(D, scv::ShellCellValues)
    ke = zeros(18, 18)
    for q in eachindex(scv.detJdV)
        B = hcat(
            ntuple(3) do i
                dx, dy = scv.∇N[q, i]
                [0.0  0.0   dx
                 0.0  -dy   0.0
                 0.0  -dx   dy]
            end...,
            zeros(3, 9),
        )
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end

function calculate_element_shear_stiffness_matrix(D, scv::ShellCellValues)
    n_shape = size(scv.N, 2)
    ke = zeros(3n_shape, 3n_shape)
    for q in eachindex(scv.detJdV)
        B = hcat(map(1:n_shape) do i
            dx, dy = scv.∇N[q, i]
            B_node = [dx   0.0   0.0
                      dy   0.0   0.0]
            if i ≤ 3
                N = scv.N[q, i]
                B_node += [0.0   0.0   -N
                           0.0   N     0.0]
            end
            B_node
        end...)
        ke += B' * D * B * scv.detJdV[q]
    end
    return ke
end


# ─── Combined element stiffness ───────────────────────────────────────────────
#
# scv_mb : ShellCellValues for membrane + bending (qr1, ip3 geometry, ip3 shape)
# scv_s  : ShellCellValues for shear             (qr3, ip3 geometry, ip6 shape)
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
    ke_bs = (ke_b + ke_s)[idx, idx]

    # Static condensation: eliminate bubble w DOFs (nodes 4–6 → indices 10–12)
    inda  = 1:9
    indi  = 10:12
    ke_bs = ke_bs[inda, inda] - ke_bs[inda, indi] * inv(ke_bs[indi, indi]) * ke_bs[indi, inda]

    # Assemble into 15×15 using intermediate component ordering:
    # (u1,v1,w1,θx1,θy1, u2,v2,w2,θx2,θy2, u3,v3,w3,θx3,θy3)
    ke = zeros(15, 15)
    induv = [1, 2, 6, 7, 11, 12]
    indwt = [3, 4, 5, 8, 9, 10, 13, 14, 15]
    ke[induv, induv] = ke_m
    ke[indwt, indwt] = ke_bs

    # Reindex to Ferrite field ordering:
    # (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1, θx2,θy2, θx3,θy3)
    ind_field = [1, 2, 3, 6, 7, 8, 11, 12, 13, 4, 5, 9, 10, 14, 15]
    return ke[ind_field, ind_field]
end


# ─── Geometric stiffness ──────────────────────────────────────────────────────

function calculate_element_geometric_stiffness_matrix(scv::ShellCellValues, σ)
    kg = zeros(15, 15)
    for q in eachindex(scv.detJdV)
        # G matrices: row k = ∂(displacement component k)/∂(x or y), for all 15 DOFs.
        # Ferrite DOF ordering: (u1,v1,w1, u2,v2,w2, u3,v3,w3, θx1,θy1,...)
        # Only the translational DOFs 1–9 are non-zero.
        Nuvw_x = zeros(3, 15)
        Nuvw_y = zeros(3, 15)
        for i in 1:3
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
