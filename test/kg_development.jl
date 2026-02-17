
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


# σx, σy, σxy right now these are in the element local coordinate system
σ = [(-1.0, 0.0, 0.0) for i=1:getncells(grid)]

# allocate and assemble the sparse matrix
Kg = allocate_matrix(dh)
TriShellFiniteElement.assemble_global_Kg!(Kg, dh, qr1, ip3, σ)

