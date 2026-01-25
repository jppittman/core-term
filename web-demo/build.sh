#!/bin/bash
set -e
wasm-pack build --target web --release
echo "Built! Open index.html in a browser (needs local server for CORS)"
echo "Run: python3 -m http.server 8080"
