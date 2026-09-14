#!/usr/bin/env bash
#
# build-xcframework.sh
#
# Compiles the nimbus-splat-core Rust crate (Brush-based on-device splat
# trainer) for iOS device + simulator and packages it as
#   app/Frameworks/NimbusSplatCore.xcframework
# which project.yml links with `embed: true, codeSign: true`.
#
# Contract (NIMBUS_CONTRACTS.md "Build rules"): CI runs THIS script from the app
# directory BEFORE `xcodegen generate` / `xcodebuild`. macOS only (needs Xcode's
# lipo / install_name_tool / xcodebuild -create-xcframework and the Rust iOS
# targets). Paths are resolved from the script's own location, so the working
# directory does not matter.
#
# Output framework layout (per slice):
#   NimbusSplatCore.framework/
#     NimbusSplatCore            <- the cdylib, install_name @rpath/...
#     Headers/nimbus_splat_core.h
#     Modules/module.modulemap   <- `framework module NimbusSplatCore`
#     Info.plist
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$SCRIPT_DIR/rust/nimbus-splat-core"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"     # Sources/SplatEngine -> Sources -> app
OUT_DIR="$APP_DIR/Frameworks"
STAGE_DIR="$CRATE_DIR/target/xcframework"

FW_NAME="NimbusSplatCore"
DYLIB="libnimbus_splat_core.dylib"
HEADER="$CRATE_DIR/include/nimbus_splat_core.h"
MIN_IOS="17.0"

DEVICE_TARGET="aarch64-apple-ios"
SIM_TARGET="aarch64-apple-ios-sim"

echo "==> nimbus-splat-core -> $FW_NAME.xcframework"
echo "    crate:  $CRATE_DIR"
echo "    output: $OUT_DIR/$FW_NAME.xcframework"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found on PATH. Install Rust (rustup) before running this script." >&2
  exit 1
fi
if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "error: xcodebuild not found. This script must run on macOS with Xcode installed." >&2
  exit 1
fi

# iOS targets (CI also installs these; harmless if already present).
rustup target add "$DEVICE_TARGET" "$SIM_TARGET" >/dev/null 2>&1 || true

export IPHONEOS_DEPLOYMENT_TARGET="$MIN_IOS"

echo "==> cargo build ($DEVICE_TARGET)"
( cd "$CRATE_DIR" && cargo build --release --target "$DEVICE_TARGET" )
echo "==> cargo build ($SIM_TARGET)"
( cd "$CRATE_DIR" && cargo build --release --target "$SIM_TARGET" )

# --- assemble one NimbusSplatCore.framework from a built .dylib ----------------
# args: <rust-target-triple> <stage-subdir> <supported-platform>
make_framework() {
  local target="$1" subdir="$2" platform="$3"
  local src="$CRATE_DIR/target/$target/release/$DYLIB"
  local fw="$STAGE_DIR/$subdir/$FW_NAME.framework"

  if [ ! -f "$src" ]; then
    echo "error: expected build output missing: $src" >&2
    exit 1
  fi

  rm -rf "$fw"
  mkdir -p "$fw/Headers" "$fw/Modules"

  cp "$src" "$fw/$FW_NAME"
  # Frameworks are loaded via @rpath; without this the loader looks for the
  # crate's default install_name and fails at launch.
  install_name_tool -id "@rpath/$FW_NAME.framework/$FW_NAME" "$fw/$FW_NAME"

  cp "$HEADER" "$fw/Headers/nimbus_splat_core.h"

  cat > "$fw/Modules/module.modulemap" <<EOF
framework module $FW_NAME {
    umbrella header "nimbus_splat_core.h"
    export *
    module * { export * }
}
EOF

  cat > "$fw/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>$FW_NAME</string>
  <key>CFBundleIdentifier</key><string>com.nimbus3d.NimbusSplatCore</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>$FW_NAME</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>MinimumOSVersion</key><string>$MIN_IOS</string>
  <key>CFBundleSupportedPlatforms</key><array><string>$platform</string></array>
</dict>
</plist>
EOF

  echo "    built $fw"
}

echo "==> assembling frameworks"
make_framework "$DEVICE_TARGET" "ios"     "iPhoneOS"
make_framework "$SIM_TARGET"    "ios-sim" "iPhoneSimulator"

echo "==> create-xcframework"
rm -rf "$OUT_DIR/$FW_NAME.xcframework"
mkdir -p "$OUT_DIR"
xcodebuild -create-xcframework \
  -framework "$STAGE_DIR/ios/$FW_NAME.framework" \
  -framework "$STAGE_DIR/ios-sim/$FW_NAME.framework" \
  -output "$OUT_DIR/$FW_NAME.xcframework"

echo "==> done"
find "$OUT_DIR/$FW_NAME.xcframework" -maxdepth 2 -print
