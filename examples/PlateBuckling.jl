using TriShellFiniteElement
using Ferrite
using LinearAlgebra
using CairoMakie

p = 1.0      # N/mm²              # Uniform pressure load intensity
E = 200000.0 # MPa
ν = 0.30     # n-a
t = 2.0      # thickness in mm

# domain and grid
grid = generate_grid(Ferrite.Triangle, (2, 8),
                     Ferrite.Vec(0.0, 0.0),
                     Ferrite.Vec(100.0, 1000.0)) |> ShellMesh

# interpolation
ip = Lagrange{RefTriangle,1}()
ip_s = Lagrange{RefTriangle,2}()
qr = QuadratureRule{RefTriangle}(1)
qr_s = QuadratureRule{RefTriangle}(2)

# shell cell values for membrane and bending contributions
scv_mb = ShellCellValues(qr, ip, ip)
scv_s  = ShellCellValues(qr_s, ip, ip_s)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# add the Dirichlet Boundary on the faces of the model
addfacetset!(grid, "left_faces", x -> x[2] ≈ 0.0)
addfacetset!(grid, "right_faces", x -> x[2] ≈ 1000.0)
# internal nodes
addnodeset!(grid, "middle_node", x -> (x[1] ≈ 50.0 && x[2] ≈ 500.0))
addnodeset!(grid, "left_top", x -> (x[1] ≈ 0.0 && x[2] ≈ 0.0))
addnodeset!(grid, "right_top", x -> (x[1] ≈ 0.0 && x[2] ≈ 1000.0))

# add the boundary condition to the dh
ch = ConstraintHandler(dh)
# uy restrained at middle node
add!(ch, Dirichlet(:u, getnodeset(grid, "middle_node"),  (x, t) -> [0.0], [2]))
# ux restrained at top corner nodes
add!(ch, Dirichlet(:u, getnodeset(grid, "left_top"),  (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, getnodeset(grid, "right_top"), (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, getfacetset(grid, "left_faces"), (x, t) -> [0.0], [3]))
add!(ch, Dirichlet(:u, getfacetset(grid, "right_faces"), (x, t) -> [0.0], [3]))
close!(ch)

# elastic stiffness matrix
Ke = allocate_matrix(dh)
Ke = TriShellFiniteElement.assemble_global_Ke!(Ke, dh, scv_mb, scv_s, E, ν, t)

# apply the boundary conditions to the stiffness matrix
apply!(Ke, ch)

#assume a negative stress is compression
σ_elem = Vector{Ferrite.Vec{3,Float64}}()
for cell in CellIterator(dh)
    push!(σ_elem, Ferrite.Vec(0.0, -p, 0.0))
end

# geometric stiffness matrix
Kg = allocate_matrix(dh)
Kg = TriShellFiniteElement.assemble_global_Kg!(Kg, dh, scv_mb, σ_elem)
apply!(Kg, ch)

# solve the eigenvalue problem
eigenvalues = eigvals(Matrix(Ke), -Matrix(Kg))
eigenvectors = eigvecs(Matrix(Ke), -Matrix(Kg))

# Filter to finite positive eigenvalues and sort them
pos_indices = findall(v -> isfinite(v) && v > 0, eigenvalues)
sorted_indices = pos_indices[sortperm(eigenvalues[pos_indices])]

n_modes = min(5, length(sorted_indices))  # number of modes to extract
println("First $n_modes buckling eigenvalues (load factors):")
for i in 1:n_modes
    idx = sorted_indices[i]
    println("  Mode $i: λ = $(eigenvalues[idx])")
end

########################################################
######<<<<<<<<<<<<  Visualization >>>>>>>>>>>>>>########
########################################################

using CairoMakie

mode1_idx = sorted_indices[1]
mode1_vector = eigenvectors[:, mode1_idx]

# Normalize mode shape (important!)
mode1_vector ./= maximum(abs.(mode1_vector))

n_nodes = length(grid.nodes)
u_node = zeros(n_nodes, 3)   # we only need ux, uy, w

for cell in CellIterator(dh)
    cdofs = celldofs(cell)
    for (i, n) in enumerate(cell.nodes)
        # 5 dofs per node
        base = 5*(i-1)
        u_node[n,1] = mode1_vector[cdofs[base + 1]]  # ux
        u_node[n,2] = mode1_vector[cdofs[base + 2]]  # uy
        u_node[n,3] = mode1_vector[cdofs[base + 3]]  # w
    end
end

using GeometryBasics
faces = GeometryBasics.TriangleFace{Int}[]

for cell in grid.cells
    push!(faces, GeometryBasics.TriangleFace(cell.nodes...))
end

scale = 20.0

vertices = Point3f[]

for i in 1:n_nodes
    x = grid.nodes[i].x[1] + scale * u_node[i,1]
    y = grid.nodes[i].x[2] + scale * u_node[i,2]
    z = scale * u_node[i,3]

    push!(vertices, Point3f(x, y, z))
end

gb_mesh = GeometryBasics.Mesh(vertices, faces)

fig = Figure(size = (900,700))

ax = Axis3(
    fig[1,1],
    xlabel = "X (mm)",
    ylabel = "Y (mm)",
    zlabel = "Z (mm)",
    title = "Buckling Mode Shape",
    aspect = :data
)

mesh!(ax, gb_mesh,
      color = u_node[:,3],
      colormap = :turbo,
      colorrange = (-1, 1))

wireframe!(ax, gb_mesh,
           color = :black,
           linewidth = 0.5)

Colorbar(fig[1,2],
         colormap = :turbo,
         label = "Normalized w")

fig
