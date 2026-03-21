import re

def fix_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('pixelflow-compiler/src/lib.rs', [
    ('#![allow(dead_code)]\n', ''),
    ('#![forbid(unsafe_code)]\n', '#![forbid(unsafe_code)]\n#![allow(dead_code)]\n')
])

print("Fixes applied.")
