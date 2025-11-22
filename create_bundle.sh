#!/bin/bash
# Create a .app bundle for core-term
# This is required for keyboard input to work on macOS

set -e

# Build in release mode
echo "Building core-term..."
cargo build --release

# Create bundle structure
BUNDLE_DIR="./CoreTerm.app/Contents"
rm -rf CoreTerm.app
mkdir -p "$BUNDLE_DIR/MacOS"
mkdir -p "$BUNDLE_DIR/Resources"

# Copy binary
cp target/release/core-term "$BUNDLE_DIR/MacOS/CoreTerm"

# Create Info.plist
cat > "$BUNDLE_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>CoreTerm</string>
    <key>CFBundleIdentifier</key>
    <string>com.core-term.terminal</string>
    <key>CFBundleName</key>
    <string>CoreTerm</string>
    <key>CFBundleDisplayName</key>
    <string>CoreTerm</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

echo "✅ Created CoreTerm.app"
echo "Run with: open CoreTerm.app"
echo "Or: ./CoreTerm.app/Contents/MacOS/CoreTerm"
