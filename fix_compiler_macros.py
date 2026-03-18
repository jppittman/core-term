import re

def fix_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

fix_file("pixelflow-compiler/src/lib.rs", [
    ("#![warn(dead_code)]", "#![allow(dead_code)]"),
    ("mod annotate;", "#![allow(dead_code)]\n#![allow(unused_variables)]\n#![allow(unused_imports)]\nmod annotate;")
])
