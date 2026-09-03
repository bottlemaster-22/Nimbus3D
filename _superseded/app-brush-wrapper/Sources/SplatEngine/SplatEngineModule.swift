//
//  SplatEngineModule.swift — module anchor. Owned by the SplatEngine agent.
//
//  This module implements `SplatTrainer` (Sources/Core/Contracts.swift) by
//  wrapping the Rust Brush engine (github.com/ArthurBrussee/brush, wgpu-on-Metal)
//  compiled into Frameworks/NimbusSplatCore.xcframework. The Rust crate, its
//  C FFI header, and the xcframework build script live with this module.
//  Add real implementation files alongside this one; do not edit Core contracts.
//

/// Namespace marker for the SplatEngine module.
public enum SplatEngineModule {
    public static let name = "SplatEngine"
}
