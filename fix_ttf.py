import re

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'r') as f:
    content = f.read()

content = content.replace('t_plus', 'tp')
content = content.replace('t_minus', 'tm')
content = content.replace('tp', 't_plus')
content = content.replace('tm', 't_minus')
content = content.replace('t_plus.clone()', 't_plus.clone()')

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'w') as f:
    f.write(content)
