import re

def fix_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

fix_file("pixelflow-graphics/src/fonts/ttf_curve_analytical.rs", [
    ("let t_plus = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();", "let t_plus_val = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();\n            let t_plus = t_plus_val.clone();"),
    ("let t_minus = sqrt_disc * -inv_2a + neg_b_2a;", "let t_minus_val = sqrt_disc * -inv_2a + neg_b_2a;\n            let t_minus = t_minus_val.clone();"),
    ("let t = (Y - cy) / by;", "let t_val = (Y - cy) / by;\n                let t = t_val.clone();"),
])

fix_file("pixelflow-graphics/src/scene3d.rs", [
    ("let mask = valid_t & valid_deriv;", "let mask = valid_t.clone() & valid_deriv.clone();"),
    ("let hx = X * t;", "let hx = X * t.clone();"),
    ("let hy = Y * t;", "let hy = Y * t.clone();"),
    ("let hz = Z * t;", "let hz = Z * t.clone();"),
    ("valid_t & valid_deriv\n});", "valid_t.clone() & valid_deriv.clone()\n});"),
])
