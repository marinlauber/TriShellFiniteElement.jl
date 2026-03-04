# plane stress material models

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