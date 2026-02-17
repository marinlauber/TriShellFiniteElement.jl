using TriShellFiniteElement, FerriteViz

E = 200000.0
t = 1.0
ν = 0.30

######
grid = generate_grid(Triangle, (1, 1), Vec((0.0, 0.0)), Vec((1.0, 1.0)));

ip = Lagrange{RefTriangle,1}() #to define fields only
ip3 = TriShellFiniteElement.IP3()
ip6 = TriShellFiniteElement.IP6()
qr1 = QuadratureRule{RefTriangle}(1)
qr3 = QuadratureRule{RefTriangle}(2)

# 15 dof per element?
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)


u_range = dof_range(dh, :u)
θ_range = dof_range(dh, :θ)

global_dofs = celldofs(dh, 1)


# ch = ConstraintHandler(dh)
# dbc1 = Dirichlet(
#     :u,                        # Name of the field
#     getfacetset(grid, "left"), # Part of the boundary
#     x -> 1.0,                  # Function mapping coordinate to a prescribed value
# )
# close!(ch)

# addfacetset!(grid, "ymin",  (x) -> x[2] ≈ 0.0)


K = allocate_matrix(dh)

TriShellFiniteElement.assemble_global_Ke!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)





