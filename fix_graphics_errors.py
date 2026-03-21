import re

def fix_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', [
    ('let in_t = t.clone().ge(0.0) & t.clone().le(1.0);', 'let t = -c / b.clone();\n                let in_t = t.clone().ge(0.0) & t.clone().le(1.0);'),
    ('let x_int = t.clone() * t.clone() * ax + t.clone() * bx + cx;', 'let x_int = t.clone() * t.clone() * ax.clone() + t.clone() * bx.clone() + cx.clone();'),
    ('let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();', 'let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();'),
    ('let dy_plus = t_plus.clone() * (2.0 * ay) + by;', 'let dy_plus = t_plus.clone() * (2.0 * ay.clone()) + by.clone();'),
    ('let valid_plus = t_plus.clone().ge(0.0) & t_plus.clone().le(1.0);', 'let valid_plus = t_plus.clone().ge(0.0) & t_plus.clone().le(1.0);'),
    ('let x_minus = t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx;', 'let x_minus = t_minus.clone() * t_minus.clone() * ax.clone() + t_minus.clone() * bx.clone() + cx.clone();'),
    ('let dy_minus = t_minus.clone() * (2.0 * ay) + by;', 'let dy_minus = t_minus.clone() * (2.0 * ay.clone()) + by.clone();'),
    ('let t_plus = (-b + sqrt_disc) * inv_2a;\n            let t_minus = (-b - sqrt_disc) * inv_2a;', 'let t_plus = (-b.clone() + sqrt_disc.clone()) * inv_2a.clone();\n            let t_minus = (-b.clone() - sqrt_disc.clone()) * inv_2a.clone();')
])

fix_file('pixelflow-graphics/src/scene3d.rs', [
    ('let mask = valid_t & valid_deriv;', 'let mask = valid_t.clone() & valid_deriv.clone();'),
    ('let mat_val = material.at(hx, hy, hz, W);', 'let hx = ray_x.clone() * t.clone();\n    let hy = ray_y.clone() * t.clone();\n    let hz = ray_z.clone() * t.clone();\n    let mat_val = material.at(hx, hy, hz, W);'),
    ('valid_t & valid_deriv', 'valid_t.clone() & valid_deriv.clone()'),
    ('self.inner.eval(r_x, r_y, r_z, w)', 'self.inner.eval((r_x, r_y, r_z, w))')
])

print("Fixes applied.")
