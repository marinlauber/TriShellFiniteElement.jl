# Development Log — TriShellFiniteElement.jl

## 1. Initial Implementation

The element started as a flat Reissner-Mindlin triangle with:

- **Membrane**: 3-node linear (P1) in-plane field (u, v), 6×6 stiffness.
- **Bending**: DKT-like 3-node (P1) out-of-plane field (w, θx, θy), 18×18 padded stiffness.
- **Shear**: Mixed P1 geometry + P2 (6-node) transverse displacement, 18×18 with static condensation of 3 edge-midpoint w-DOFs → 9×9.
- **Assembly**: 15×15 combined element matrix (5 DOFs per node: u, v, w, θx, θy).

Custom interpolation types `IP3` and `IP6` were used, together with manual 2D coordinate projection (`global_nodal_coords_to_planar_coords`) and a bespoke Jacobian computation.

---

## 2. Switch to Convention 1 Shear Signs

**Motivation**: The initial code used Convention 2 (`γxz = ∂w/∂x + θy`, `γyz = ∂w/∂y − θx`). The standard textbook convention (Bathe, Zienkiewicz, Hughes) is Convention 1:

```
γxz = ∂w/∂x − θy
γyz = ∂w/∂y + θx
```

This makes θy follow the right-hand rule about y (θy = +∂w/∂x in the thin-plate limit) and θx follow the right-hand rule about x (θx = −∂w/∂y), which is the physically intuitive sign convention.

**Change**: One sign flip in the shear B matrix:
```julia
# Convention 1
B_node += [0.0  0.0  -N    # γxz = ∂w/∂x − θy  →  −N on θy column
           0.0   N   0.0]  # γyz = ∂w/∂y + θx  →  +N on θx column
```

**Rigid body mode update**: The rigid body mode vectors in the tests were updated to reflect the new sign convention:
- **Rx** (rotation about x): w = y_i, **θx = −1** (was +1)
- **Ry** (rotation about y): w = x_i, **θy = +1** (unchanged)

---

## 3. Refactor: `ShellCellValues` — Surface Jacobian Approach

**Motivation**: The old code used internal Ferrite data (`cv.fun_values.dNdξ[i+(q-1)*n]`), manual 2D coordinate projection, and a bespoke Jacobian function. This was fragile and only worked for triangles in the z=0 plane.

**Key insight**: For a 2D surface embedded in 3D, the Jacobian is a 3×2 matrix `J = [J1 | J2]` where `J1 = ∂x/∂ξ₁`, `J2 = ∂x/∂ξ₂` (both `Vec{3}`). The local orthonormal frame and shape function gradients follow directly:

```
t1 = J1 / ‖J1‖
n  = (J1 × J2) / ‖J1 × J2‖
t2 = n × t1

area element:  ‖J1 × J2‖ × weight

local 2D gradient (via metric tensor g = JᵀJ):
  α1 = (g22·∂N/∂ξ1 − g12·∂N/∂ξ2) / det_g
  α2 = (g11·∂N/∂ξ2 − g12·∂N/∂ξ1) / det_g
  ∇N_global = J1·α1 + J2·α2
  ∇N_local  = Vec{2}((∇N_global⋅t1, ∇N_global⋅t2))
```

**Result**: A `ShellCellValues` struct that stores `N`, `∇N` (local 2D), `detJdV`, and `local_frame` (3×3 rotation matrix `[t1|t2|n]`). Works with any `Vec{3}` node coordinates — no manual projection required. Removed:
- `IP3`, `IP6` custom interpolation types
- `get_jacobian`, `calculation_rotation_matrix`
- `global_nodal_coords_to_planar_coords`

`ShellCellValues` is generic over any Ferrite interpolation (`ip_geo` for geometry, `ip_shape` for displacement field). The `reinit!` method is extended from Ferrite.

---

## 4. Remove Custom IP3/IP6 — Use Ferrite Built-ins

**Motivation**: The custom `IP3` / `IP6` types duplicated Ferrite's `Lagrange{RefTriangle,1}` and `Lagrange{RefTriangle,2}`. Removing them allows the element to work with any Ferrite reference shape.

**Change**: Replaced `IP3()` / `IP6()` with `Lagrange{RefTriangle,1}()` / `Lagrange{RefTriangle,2}()` everywhere. Removed all hardcoded `3`s in stiffness functions, using `getnbasefunctions(scv.ip_geo)` instead.

**Node ordering**: Ferrite's `Lagrange{RefTriangle,1}` uses:
- N1 = ξ₁ → node 1 at reference vertex (1,0)
- N2 = ξ₂ → node 2 at reference vertex (0,1)
- N3 = 1−ξ₁−ξ₂ → node 3 at reference vertex (0,0)

This differs from the old custom IP3 (which placed node 1 at the origin). Physical node coordinates must follow the same ordering: `x[1]=(1,0,0)`, `x[2]=(0,1,0)`, `x[3]=(0,0,0)` for the unit right triangle.

---

## 5. Fix Quadrature Rule and Shear B Matrix Bug

Two bugs were discovered after the IP3/IP6 removal:

### Bug 1: Negative quadrature weight
`QuadratureRule{RefTriangle}(3)` produces the 4-point Dunavant rule, which has a **negative weight** at the centroid. This makes the shear stiffness matrix indefinite. The fix is to use `QuadratureRule{RefTriangle}(2)` (3-point rule, all positive weights), which is also the standard choice for reduced integration of shear in Reissner-Mindlin elements.

### Bug 2: P2 corner functions ≠ partition of unity
In the shear B matrix, the rotation field θ is interpolated over corner nodes only (P1, 3 nodes). The code was using `scv.N[q, i]` — the P2 shape function values from `ip_shape = Lagrange{RefTriangle,2}` — for the θ terms. However, the P2 corner shape functions (`N_i^{P2} = ξ_i(2ξ_i−1)`) do **not** form a partition of unity: `N1^{P2} + N2^{P2} + N3^{P2} ≠ 1` at interior points.

This caused non-zero shear strains for rigid body rotations (Rx, Ry), breaking the patch test.

**Fix**: Evaluate `ip_geo` (P1) shape functions at the quadrature point for the θ terms:
```julia
N = Ferrite.reference_shape_value(scv.ip_geo, ξ, i)
```
P1 corner functions satisfy `N1^{P1} + N2^{P1} + N3^{P1} = 1` everywhere, so `Σ N_i * θy_i = θy` for a uniform rotation field, and the rigid body mode shear energy is correctly zero.

---

## DOF Ordering Summary

After the final reindex in `elastic_stiffness_matrix`, the 15 DOFs per element are:

```
(u1, v1, w1,  u2, v2, w2,  u3, v3, w3,  θx1, θy1,  θx2, θy2,  θx3, θy3)
```

This matches Ferrite's field-based DOF ordering when using separate `u`, `w`, and `θ` fields.
