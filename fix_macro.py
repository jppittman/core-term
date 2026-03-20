import re

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'r') as f:
    text = f.read()

# Replace t_plus, t_minus with root1, root2
text = re.sub(r'\bt_plus\b', 'root1', text)
text = re.sub(r'\bt_minus\b', 'root2', text)
text = re.sub(r'\bx_plus\b', 'x1', text)
text = re.sub(r'\bx_minus\b', 'x2', text)
text = re.sub(r'\bdy_plus\b', 'dy1', text)
text = re.sub(r'\bdy_minus\b', 'dy2', text)
text = re.sub(r'\bcrossed_plus\b', 'c1', text)
text = re.sub(r'\bcrossed_minus\b', 'c2', text)
text = re.sub(r'\bvalid_plus\b', 'v1', text)
text = re.sub(r'\bvalid_minus\b', 'v2', text)
text = re.sub(r'\bsign_plus\b', 's1', text)
text = re.sub(r'\bsign_minus\b', 's2', text)
text = re.sub(r'\bcontrib_plus\b', 'contrib1', text)
text = re.sub(r'\bcontrib_minus\b', 'contrib2', text)

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'w') as f:
    f.write(text)

with open('pixelflow-graphics/src/scene3d.rs', 'r') as f:
    text = f.read()

text = re.sub(r'\bvalid_t\b', 'is_valid_t', text)
text = re.sub(r'\bvalid_deriv\b', 'is_valid_deriv', text)
text = re.sub(r'\bhx\b', 'hit_x', text)
text = re.sub(r'\bhy\b', 'hit_y', text)
text = re.sub(r'\bhz\b', 'hit_z', text)

with open('pixelflow-graphics/src/scene3d.rs', 'w') as f:
    f.write(text)
