module TriShellFiniteElement

using Ferrite, LinearAlgebra, Tensors

struct IP6 <: ScalarInterpolation{RefTriangle, 2}
end


function Ferrite.reference_shape_value(ip::IP6, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number == 1 && return 1 - ξ₁ - ξ₂
    shape_number == 2 && return ξ₁
    shape_number == 3 && return ξ₂
    shape_number == 4 && return 4* ξ₁ * (1 - ξ₁ - ξ₂)
    shape_number == 5 && return 4 * ξ₁ * ξ₂
    shape_number == 6 && return 4 * ξ₂ * (1 - ξ₁ - ξ₂)

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end


Ferrite.getnbasefunctions(::IP6) = 6

Ferrite.adjust_dofs_during_distribution(::IP6) = false


struct IP3 <: ScalarInterpolation{RefTriangle, 2}
end


function Ferrite.reference_shape_value(ip::IP3, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number == 1 && return 1 - ξ₁ - ξ₂
    shape_number == 2 && return ξ₁
    shape_number == 3 && return ξ₂

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end


# Ferrite.vertexdof_indices(::IP3) = ((1,2,3,4,5), (6,7,8,9,10), (11,12,13,14,15))
#organized by field u:1-9, θ:10-15
# Ferrite.vertexdof_indices(::IP3) = ((1,2,3,10,11), (4,5,6,12,13), (7,8,9,14,15))

Ferrite.getnbasefunctions(::IP3) = 3

Ferrite.adjust_dofs_during_distribution(::IP3) = false



function get_jacobian(ξ, ip, x)

    shape_number = 1
    dNdξ1 = Ferrite.reference_shape_gradient(ip, ξ, shape_number)

    shape_number = 2
    dNdξ2 = Ferrite.reference_shape_gradient(ip, ξ, shape_number)

    shape_number = 3
    dNdξ3 = Ferrite.reference_shape_gradient(ip, ξ, shape_number)


    J11 = [dNdξ1[1] dNdξ2[1] dNdξ3[1]] * [x[1][1], x[2][1], x[3][1]]
    J12 = [dNdξ1[1] dNdξ2[1] dNdξ3[1]] * [x[1][2], x[2][2], x[3][2]]
    J21 = [dNdξ1[2] dNdξ2[2] dNdξ3[2]] * [x[1][1], x[2][1], x[3][1]]
    J22 = [dNdξ1[2] dNdξ2[2] dNdξ3[2]] * [x[1][2], x[2][2], x[3][2]]


    J = [J11 J12
        J21 J22]

    return J

end


function calculate_membrane_constitutive_matrix(E, ν, t)

    G=E/(2*(1+ν))

    D=[    E/(1-ν^2) ν*E/(1-ν^2)    0
        ν*E/(1-ν^2)    E/(1-ν^2)    0
                0             0    G]

    D = D .* t

    return D

end




function calculate_bending_constitutive_matrix(E, ν, t)

    D_const = E * t^3 / (12 * (1 - ν^2))
    D = D_const * [1.0  ν    0.0
                    ν    1.0  0.0
                    0.0  0.0  (1-ν)/2]

    return D

end



function calculate_shear_constitutive_matrix(E, ν, t)

    G=E/(2*(1+ν))

    D=[  5/6*G*t    0.0
        0.0        5/6*G*t]

    return D

end

function calculate_element_membrane_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    num_shape_functions = getnbasefunctions(ip_shape)

    ke = zeros(Float64, 6, 6)


    for q_point in 1:getnquadpoints(cv)

        ξ = qr.points[q_point]
        J = get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)

        B_node_all = []

        for i in 1:3

            dNdξ1 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1]
            dNdξ2 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2]

            B_node = [dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]         0.0
                        0.0                                       dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]
                        dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]         dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]]

            push!(B_node_all, B_node)

        end

        B = hcat(B_node_all...)

        ke += B' * D * B .* det(J) .* qr.weights[q_point]

    end

    return ke

end



function calculate_element_bending_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    num_shape_functions = getnbasefunctions(ip_shape)

    ke = zeros(Float64, 18, 18)

    for q_point in 1:getnquadpoints(cv)

        ξ = qr.points[q_point]
        J = get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)

        B_node_all = []

        for i in 1:3

            dNdξ1 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1]
            dNdξ2 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2]

            B_node = [0.0       0.0             dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]
                        0.0       -(dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2])  0.0
                        0.0       -(dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2])  dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]]

            push!(B_node_all, B_node)

        end

        push!(B_node_all, zeros(3, 9))
        B = hcat(B_node_all...)

        ke += B' * D * B .* det(J) .* qr.weights[q_point]

    end

    return ke

end




function calculate_element_shear_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    num_shape_functions = getnbasefunctions(ip_shape)

    ke = zeros(Float64, 18, 18)

    for q_point in 1:getnquadpoints(cv)

        ξ = qr.points[q_point]
        J = get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)

        B_node_all = []

        for i=1:num_shape_functions

            dNdξ1 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1]
            dNdξ2 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2]

            B_node = [dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]         0.0     0.0
                                dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]         0.0     0.0]

            if i <= 3

                N = Ferrite.reference_shape_value(ip_shape, ξ, i)

                B_node += [0.0     0.0     -N
                           0.0      N     0.0]

            end

            push!(B_node_all, B_node)

        end

        B = hcat(B_node_all...)

        ke += B' * D * B .* det(J) .* qr.weights[q_point]

    end

    return ke

end



function elastic_stiffness_matrix!(qr1, qr3, ip3, ip6, E, ν, t, x)

    #####membrane
    cv = CellValues(qr1, ip3, ip3)
    reinit!(cv, x)

    Dm = TriShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, t)

    D = Dm
    ip_geo = ip3
    ip_shape = ip3
    qr = qr1
    ke_m = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    ######bending
    cv = CellValues(qr1, ip3, ip3)
    reinit!(cv, x)

    Db = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t)

    D = Db
    ip_geo = ip3
    ip_shape = ip3
    qr = qr1
    ke_b = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    ######shear
    cv = CellValues(qr3, ip6, ip3)
    reinit!(cv, x)

    Ds = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)

    D = Ds
    ip_geo = ip3
    ip_shape = ip6
    qr = qr3
    ke_s = TriShellFiniteElement.calculate_element_shear_stiffness_matrix(D, cv, ip_geo, ip_shape, qr, x)

    ke_bs = ke_b + ke_s

    #remove zeros
    indices = [1:10; 13; 16]
    ke_bs = ke_bs[indices, indices]

    #static condensation
    inda=1:9
    indi=10:12
    ke_bs = ke_bs[inda,inda]-ke_bs[inda,indi]*inv(ke_bs[indi,indi])*ke_bs[indi,inda]


    ke = zeros(Float64, 15, 15)

    induv=[1 2 6 7 11 12]
    indwt=[3 4 5 8 9 10 13 14 15]
    ke[induv,induv]=ke_m
    ke[indwt,indwt]=ke_bs

    #reorder from component to fields, Ferrite default
    ind_field = [1, 2, 3, 6, 7, 8, 11, 12, 13, 4, 5, 9, 10, 14, 15]
    ke = ke[ind_field, ind_field]

    return ke

end


function assemble_global_Ke!(Ke, dh, qr1, qr3, ip3, ip6, E, ν, t)

    assembler = start_assemble(Ke)
    for cell in CellIterator(dh)

        x_global = getcoordinates(cell)

        T = calculation_rotation_matrix(x_global)

        x_local = global_nodal_coords_to_planar_coords(x_global, T)

        ke = TriShellFiniteElement.elastic_stiffness_matrix!(qr1, qr3, ip3, ip6, E, ν, t, x_local)

        #rotate element stiffness matrix back to global coordinates!
        Te = rotation_matrix_for_element_stiffness(T)
        ke_global = Te * ke * Te'

        assemble!(assembler, celldofs(cell), ke_global)
    end
    return Ke
end


####

function generate_Nuvw_derivative(dNdξ_d)

    Nuvw_d = zeros(Float64, 3, 15)
    Nuvw_d[1, 1:2:5] .= dNdξ_d
    Nuvw_d[2, 2:2:6] .= dNdξ_d
    Nuvw_d[3, 7:3:13] .= dNdξ_d

    return Nuvw_d

end


function calculate_element_geometric_stiffness_matrix(cv, ip_geo, ip_shape, qr, x, σ)

    reinit!(cv, x)

    num_shape_functions = getnbasefunctions(ip_shape)

    kgx = zeros(Float64, 15, 15)
    kgy = zeros(Float64, 15, 15)
    kgxy = zeros(Float64, 15, 15)

    for q_point in 1:getnquadpoints(cv)

        ξ = qr.points[q_point]
        J = TriShellFiniteElement.get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)

        dNdξ_x = [cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1] for i=1:num_shape_functions]
        dNdξ_y = [cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2] for i=1:num_shape_functions]

        Nx = generate_Nuvw_derivative(dNdξ_x)
        Ny = generate_Nuvw_derivative(dNdξ_y)

        Nuvw_x = Nx .* Jinv[1, 1] + Ny .* Jinv[1, 2]
        Nuvw_y = Nx .* Jinv[2, 1] + Ny .* Jinv[2, 2]

        GGx = Nuvw_x' * Nuvw_x
        GGy = Nuvw_y' * Nuvw_y
        GGxy = Nuvw_x' * Nuvw_y + Nuvw_y' * Nuvw_x

        kgx += GGx .* det(J) .* qr.weights[q_point]
        kgy += GGy .* det(J) .* qr.weights[q_point]
        kgxy += GGxy .* det(J) .* qr.weights[q_point]

    end

    #σ is local coordinate system stress, at  the Gauss point (constant stress in this case), σ = [σx, σy, σxy]
    kg = σ[1] .* kgx + σ[2] .* kgy + σ[3] .* kgxy

    return kg

end


function geometric_stiffness_matrix!(cv, qr1, ip3, x, σ_element)

    reinit!(cv, x)
    kg = calculate_element_geometric_stiffness_matrix(cv, ip3, ip3, qr1, x, σ_element)

    return kg

end


function assemble_global_Kg!(Kg, dh, qr1, ip3, σ_global)

    cv = CellValues(qr1, ip3, ip3)
    #need to convert global stress σ to local stress at some point
    assembler = start_assemble(Kg)
    i = 1
    for cell in CellIterator(dh)

        #3D
        x_global = getcoordinates(cell)

        T = calculation_rotation_matrix(x_global)

        str_mat_global = [σ_global[i][1] σ_global[i][3]
                          σ_global[i][3] σ_global[i][2]]

        str_mat_local =  T[1:2,1:2]' * str_mat_global * T[1:2, 1:2]

        σ_local = [str_mat_local[1, 1], str_mat_local[2, 2], str_mat_local[1, 2]]

        x_local = global_nodal_coords_to_planar_coords(x_global, T)

        kg = geometric_stiffness_matrix!(cv, qr1, ip3, x_local, σ_local)

        #rotate element stiffness matrix back to global coordinates!
        Te = rotation_matrix_for_element_stiffness(T)
        kg_global = Te * kg * Te'

        assemble!(assembler, celldofs(cell), kg_global)
        i += 1
    end
    return Kg
end


function calculation_rotation_matrix(node)

    P1 = node[1]
    P2 = node[2]
    P3 = node[3]

    norm_vec=cross(P2-P1,P3-P1)
    norm_vec=norm_vec/norm(norm_vec)
    j3=norm_vec
    j1=(P2-P1)/norm(P2-P1)
    j2=cross(j3,j1)
    T=[j1 j2 j3]

    return T

end


function global_nodal_coords_to_planar_coords(cell_nodes_global, T)

    P1 = cell_nodes_global[1]
    P2 = cell_nodes_global[2]
    P3 = cell_nodes_global[3]

    P1a=T'*(P1-P1)
    P2a=T'*(P2-P1)
    P3a=T'*(P3-P1)

    cell_local = [P1a, P2a, P3a]

    cell_nodes_local = [Tensors.Vec((cell_local[i][1], cell_local[i][2])) for i in eachindex(cell_local)]

    return cell_nodes_local

end


function rotation_matrix_for_element_stiffness(T3)

    #no drilling dof yet

    T2=T3[1:2,1:2]

    T = Matrix(1.0I, 15, 15)

    ind=[1, 2, 3]
    T[ind,ind] = T3
    ind=[4 5 6]
    T[ind,ind] = T3
    ind=[7 8 9]
    T[ind,ind] = T3
    ind=[10, 11]
    T[ind,ind] = T2
    ind=[12, 13]
    T[ind,ind] = T2
    ind=[14, 15]
    T[ind,ind] = T2

    return T

end




end # module TriShellFiniteElement
