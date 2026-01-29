#!/bin/bash
grep -q "barycentric" pixelflow-graphics/src/fonts/ttf.rs
if [ $? -eq 0 ]; then
  # If "barycentric" is found, it's a bad commit
  exit 1
else
  # If "barycentric" is NOT found, it's a good commit
  exit 0
fi