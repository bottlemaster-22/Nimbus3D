//
//  GLTFTypes.swift — Export module
//
//  Codable model of the subset of glTF 2.0 that Nimbus3D emits. These structs
//  serialize directly to the glTF JSON with `JSONEncoder` (optionals are omitted
//  by the synthesized `encodeIfPresent`, which is exactly glTF's "absent = default"
//  convention). Used for both the text `.gltf` and the binary `.glb` container.
//
//  Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
//

import Foundation

// glTF numeric enums, kept as plain Ints so JSON matches the spec exactly.
enum GLTFComponentType {
    static let float: Int = 5126
    static let unsignedInt: Int = 5125
    static let unsignedShort: Int = 5123
}

enum GLTFTarget {
    static let arrayBuffer: Int = 34962
    static let elementArrayBuffer: Int = 34963
}

enum GLTFFilter {
    static let linear: Int = 9729
    static let linearMipmapLinear: Int = 9987
}

enum GLTFWrap {
    static let repeatWrap: Int = 10497
}

// MARK: - Core document

struct GLTFAsset: Codable, Equatable {
    var version: String = "2.0"
    // Fixed string (no version/date) so identical inputs produce byte-identical output.
    var generator: String = "Nimbus3D"
}

struct GLTFBufferJSON: Codable, Equatable {
    var byteLength: Int
    /// Present for `.gltf` (points at the external `.bin`); nil for `.glb` (the BIN chunk).
    var uri: String?
}

struct GLTFBufferView: Codable, Equatable {
    var buffer: Int = 0
    var byteOffset: Int
    var byteLength: Int
    var byteStride: Int?
    var target: Int?
}

struct GLTFAccessor: Codable, Equatable {
    var bufferView: Int
    var byteOffset: Int = 0
    var componentType: Int
    var count: Int
    var type: String            // "SCALAR" | "VEC2" | "VEC3"
    var min: [Double]?
    var max: [Double]?
}

struct GLTFPrimitive: Codable, Equatable {
    var attributes: [String: Int]
    var indices: Int?
    var material: Int?
    var mode: Int?              // nil == 4 (TRIANGLES)
}

struct GLTFMesh: Codable, Equatable {
    var primitives: [GLTFPrimitive]
    var name: String?
}

struct GLTFNode: Codable, Equatable {
    var mesh: Int?
    var name: String?
    var matrix: [Double]?
}

struct GLTFScene: Codable, Equatable {
    var nodes: [Int]
    var name: String?
    var extras: NimbusSceneExtras?
}

// MARK: - Materials & textures

struct GLTFTextureRef: Codable, Equatable {
    var index: Int
    var texCoord: Int?
}

struct GLTFNormalTextureRef: Codable, Equatable {
    var index: Int
    var texCoord: Int?
    var scale: Double?
}

struct GLTFOcclusionTextureRef: Codable, Equatable {
    var index: Int
    var texCoord: Int?
    var strength: Double?
}

struct GLTFPBRMetallicRoughness: Codable, Equatable {
    var baseColorFactor: [Double]?
    var baseColorTexture: GLTFTextureRef?
    var metallicFactor: Double?
    var roughnessFactor: Double?
    var metallicRoughnessTexture: GLTFTextureRef?
}

struct GLTFMaterial: Codable, Equatable {
    var name: String?
    var pbrMetallicRoughness: GLTFPBRMetallicRoughness?
    var normalTexture: GLTFNormalTextureRef?
    var occlusionTexture: GLTFOcclusionTextureRef?
    var emissiveFactor: [Double]?
    var doubleSided: Bool?
    var alphaMode: String?
    var extras: NimbusMaterialExtras?
}

struct GLTFTexture: Codable, Equatable {
    var source: Int?
    var sampler: Int?
}

struct GLTFImage: Codable, Equatable {
    /// Present for `.gltf` (relative file path); nil for `.glb`.
    var uri: String?
    /// Present for `.glb` (points into the BIN chunk); nil for `.gltf`.
    var bufferView: Int?
    var mimeType: String?
    var name: String?
}

struct GLTFSampler: Codable, Equatable {
    var magFilter: Int?
    var minFilter: Int?
    var wrapS: Int?
    var wrapT: Int?
}

// MARK: - Nimbus extras (non-standard, namespaced under `extras`)

/// Height/parallax has no core glTF representation. We reference the packed height
/// map and parallax-occlusion params under the material's `extras` so a Nimbus-aware
/// viewer can use them while standard viewers safely ignore them.
struct NimbusParallaxExtras: Codable, Equatable {
    var enabled: Bool
    var heightScale: Float
    var minSamples: Int
    var maxSamples: Int
}

struct NimbusMaterialExtras: Codable, Equatable {
    var heightTexture: Int?
    var parallax: NimbusParallaxExtras?

    enum CodingKeys: String, CodingKey {
        case heightTexture = "nimbus_heightTexture"
        case parallax = "nimbus_parallax"
    }
}

/// The equirectangular .exr environment travels beside the glTF (there is no core
/// glTF slot for it); recorded here so tooling can find it.
struct NimbusSceneExtras: Codable, Equatable {
    var environmentEXR: String?

    enum CodingKeys: String, CodingKey {
        case environmentEXR = "nimbus_environmentEXR"
    }
}

// MARK: - Root

struct GLTFDocument: Codable, Equatable {
    var asset: GLTFAsset = GLTFAsset()
    var scene: Int?
    var scenes: [GLTFScene]?
    var nodes: [GLTFNode]?
    var meshes: [GLTFMesh]?
    var accessors: [GLTFAccessor]?
    var bufferViews: [GLTFBufferView]?
    var buffers: [GLTFBufferJSON]?
    var materials: [GLTFMaterial]?
    var textures: [GLTFTexture]?
    var images: [GLTFImage]?
    var samplers: [GLTFSampler]?
}
