import re

def fix_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

fix_file("pixelflow-graphics/src/fonts/ttf_curve_analytical.rs", [
    ("let t_plus = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();", "let tplus = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();"),
    ("let t_minus = sqrt_disc * -inv_2a + neg_b_2a;", "let tminus = sqrt_disc * -inv_2a + neg_b_2a;"),
    ("t_plus", "tplus"),
    ("t_minus", "tminus"),
    ("let t = (Y - cy) / by;", "let tint = (Y - cy) / by;"),
    ("let in_t = t.clone().ge(0.0) & t.clone().le(1.0);", "let in_t = tint.clone().ge(0.0) & tint.clone().le(1.0);"),
    ("let x_int = t.clone() * t.clone() * ax + t.clone() * bx + cx;", "let x_int = tint.clone() * tint.clone() * ax + tint.clone() * bx + cx;"),
])

fix_file("pixelflow-graphics/src/scene3d.rs", [
    ("let hx = X * t;", "let hx = X * t.clone();"),
    ("let hy = Y * t;", "let hy = Y * t.clone();"),
    ("let hz = Z * t;", "let hz = Z * t.clone();"),
    ("let mask = valid_t & valid_deriv;", "let mask = valid_t.clone() & valid_deriv.clone();"),
    ("valid_t & valid_deriv\n});", "valid_t.clone() & valid_deriv.clone()\n});"),
    ("self.inner.eval(r_x, r_y, r_z, w)", "self.inner.eval((r_x, r_y, r_z, w))"),
])
