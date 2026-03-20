import re

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'r') as f:
    content = f.read()

# Replace t_plus and t_minus with t_p and t_m to ensure clean replacements
# and no substring matching issues.
content = re.sub(r'\bt_plus\b', 'tp', content)
content = re.sub(r'\bt_minus\b', 'tm', content)

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'w') as f:
    f.write(content)
