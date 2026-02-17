
using TriShellFiniteElement


E = 200000.0
t = 1.0
ν = 0.30

######
grid = generate_grid(Triangle, (1, 1), Vec((0.0, 0.0)), Vec((2.0, 1.0)));

ip = Lagrange{RefTriangle,1}() #to define fields only
ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)


ip_shape = ip3
ip_geo = ip3


cell = first(CellIterator(dh))

qr = qr1

cv = CellValues(qr1, ip3, ip3)

x = getcoordinates(cell)

reinit!(cv, x)


num_shape_functions = getnbasefunctions(ip_shape)

kgx = zeros(Float64, 15, 15)
kgy = zeros(Float64, 15, 15)
kgxy = zeros(Float64, 15, 15)

function integrate!(kgx, kgy, kgxy, cv, qr)
    for q_point in 1:getnquadpoints(cv)

        # q_point = 1

        ξ = qr.points[q_point]
        J = TriShellFiniteElement.get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)

        dNdξ_x = [cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1] for i=1:num_shape_functions]
        dNdξ_y = [cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2] for i=1:num_shape_functions]

        Nx = TriShellFiniteElement.generate_Nuvw_derivative(dNdξ_x)
        Ny = TriShellFiniteElement.generate_Nuvw_derivative(dNdξ_y)

        Nuvw_x = Nx .* Jinv[1, 1] + Ny .* Jinv[1, 2]
        Nuvw_y = Nx .* Jinv[2, 1] + Ny .* Jinv[2, 2]

        GGx = Nuvw_x' * Nuvw_x
        GGy = Nuvw_y' * Nuvw_y
        GGxy = Nuvw_x' * Nuvw_y + Nuvw_y' * Nuvw_x

        kgx += GGx .* det(J) .* qr.weights[q_point]
        kgy += GGy .* det(J) .* qr.weights[q_point]
        kgxy += GGxy .* det(J) .* qr.weights[q_point]

    end
    return kgx, kgy, kgxy
end

# integrate
kgx, kgy, kgxy = integrate!(kgx, kgy, kgxy, cv, qr)

# σx, σy, σxy right now these are in the element local coordinate system
σ = (-1.0, 0.0, 0.0)

#σ is local coordinate system stress, at  the Gauss point (constant stress in this case), σ = [σx, σy, σxy]
kg = σ[1] .* kgx + σ[2] .* kgy + σ[3] .* kgxy
