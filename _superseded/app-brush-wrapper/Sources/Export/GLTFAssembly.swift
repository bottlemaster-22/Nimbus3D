//
//  GLTFAssembly.swift — Export module
//
//  Turns a `MeshAsset` (+ optional `MaterialSet`) into a fully-wired glTF graph plus
//  a single binary blob, then serializes it either as a text `.gltf` (external `.bin`
//  + image files) or a self-contained binary `.glb`. This is REAL, deterministic
//  serialization: same inputs -> byte-identical output (fixed generator string,
//  sorted JSON keys, no timestamps).
//

import Foundation
import simd

// MARK: - Little-endian byte writer (glTF buffers are little-endian; iOS is arm64/LE)

enum LE {
    static func append(_ v: Float, to d: inout Data) {
        var le = v.bitPattern.littleEndian
        withUnsafeBytes(of: &le) { d.append(contentsOf: $0) }
    }
    static func append(_ v: UInt32, to d: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { d.append(contentsOf: $0) }
    }
    static func append(_ v: UInt16, to d: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { d.append(contentsOf: $0) }
    }
    static func data(_ v: UInt32) -> Data {
        var out = Data(); append(v, to: &out); return out
    }
}

/// A texture image to be emitted (as an external file for `.gltf`, or embedded for `.glb`).
struct GLTFImageSlot {
    var filename: String
    var mime: String
    var bytes: Data
}

/// Fully-assembled glTF content, container-agnostic. Serialize with `gltf()` or `glb()`.
struct GLTFAssembly {
    var accessors: [GLTFAccessor]
    var bufferViews: [GLTFBufferView]     // geometry-only; `.glb` appends image views
    var meshes: [GLTFMesh]
    var nodes: [GLTFNode]
    var scenes: [GLTFScene]
    var materials: [GLTFMaterial]?
    var textures: [GLTFTexture]?
    var samplers: [GLTFSampler]?
    var geometryBin: Data
    var imageSlots: [GLTFImageSlot]       // index i == texture source i

    // MARK: Build

    /// Builds the graph from a mesh and optional material set.
    /// - Throws: `NimbusError.exportFailed` on malformed input (mismatched buffer
    ///   counts, or materials supplied without UVs).
    static func build(mesh: MeshAsset,
                      materials: MaterialSet?,
                      sceneExtras: NimbusSceneExtras?) throws -> GLTFAssembly {
        guard !mesh.positions.isEmpty else {
            throw NimbusError.exportFailed("mesh has no vertices")
        }
        guard mesh.normals.count == mesh.positions.count else {
            throw NimbusError.exportFailed("normals count (\(mesh.normals.count)) != positions count (\(mesh.positions.count))")
        }
        guard mesh.indices.count % 3 == 0 else {
            throw NimbusError.exportFailed("index count (\(mesh.indices.count)) is not a multiple of 3")
        }
        let hasUVs = !mesh.uvs.isEmpty
        if hasUVs {
            guard mesh.uvs.count == mesh.positions.count else {
                throw NimbusError.exportFailed("uvs count (\(mesh.uvs.count)) != positions count (\(mesh.positions.count))")
            }
        }
        if materials != nil && !hasUVs {
            throw NimbusError.exportFailed("materials provided but mesh has no UVs (contract requires mesh.uvs non-empty)")
        }

        var bin = Data()
        var bufferViews: [GLTFBufferView] = []
        var accessors: [GLTFAccessor] = []

        func addBufferView(_ payload: Data, target: Int?) -> Int {
            while bin.count % 4 != 0 { bin.append(0) }   // 4-byte align
            let offset = bin.count
            bin.append(payload)
            bufferViews.append(GLTFBufferView(byteOffset: offset,
                                              byteLength: payload.count,
                                              byteStride: nil,
                                              target: target))
            return bufferViews.count - 1
        }

        // POSITION
        var posData = Data(capacity: mesh.positions.count * 12)
        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in mesh.positions {
            LE.append(p.x, to: &posData); LE.append(p.y, to: &posData); LE.append(p.z, to: &posData)
            minP = simd_min(minP, p); maxP = simd_max(maxP, p)
        }
        let posView = addBufferView(posData, target: GLTFTarget.arrayBuffer)
        let posAccessor = accessors.count
        accessors.append(GLTFAccessor(bufferView: posView,
                                      componentType: GLTFComponentType.float,
                                      count: mesh.positions.count,
                                      type: "VEC3",
                                      min: [Double(minP.x), Double(minP.y), Double(minP.z)],
                                      max: [Double(maxP.x), Double(maxP.y), Double(maxP.z)]))

        // NORMAL
        var nrmData = Data(capacity: mesh.normals.count * 12)
        for n in mesh.normals {
            LE.append(n.x, to: &nrmData); LE.append(n.y, to: &nrmData); LE.append(n.z, to: &nrmData)
        }
        let nrmView = addBufferView(nrmData, target: GLTFTarget.arrayBuffer)
        let nrmAccessor = accessors.count
        accessors.append(GLTFAccessor(bufferView: nrmView,
                                      componentType: GLTFComponentType.float,
                                      count: mesh.normals.count,
                                      type: "VEC3"))

        // TEXCOORD_0 (optional)
        var uvAccessor: Int? = nil
        if hasUVs {
            var uvData = Data(capacity: mesh.uvs.count * 8)
            for uv in mesh.uvs {
                LE.append(uv.x, to: &uvData); LE.append(uv.y, to: &uvData)
            }
            let uvView = addBufferView(uvData, target: GLTFTarget.arrayBuffer)
            uvAccessor = accessors.count
            accessors.append(GLTFAccessor(bufferView: uvView,
                                          componentType: GLTFComponentType.float,
                                          count: mesh.uvs.count,
                                          type: "VEC2"))
        }

        // Indices (optional; use UInt16 when it fits to halve size)
        var indexAccessor: Int? = nil
        if !mesh.indices.isEmpty {
            let useShort = mesh.positions.count <= Int(UInt16.max)
            var idxData = Data(capacity: mesh.indices.count * (useShort ? 2 : 4))
            if useShort {
                for i in mesh.indices { LE.append(UInt16(i), to: &idxData) }
            } else {
                for i in mesh.indices { LE.append(i, to: &idxData) }
            }
            let idxView = addBufferView(idxData, target: GLTFTarget.elementArrayBuffer)
            indexAccessor = accessors.count
            accessors.append(GLTFAccessor(bufferView: idxView,
                                          componentType: useShort ? GLTFComponentType.unsignedShort
                                                                   : GLTFComponentType.unsignedInt,
                                          count: mesh.indices.count,
                                          type: "SCALAR"))
        }
        while bin.count % 4 != 0 { bin.append(0) }

        // Materials & textures
        var imageSlots: [GLTFImageSlot] = []
        var gltfTextures: [GLTFTexture] = []
        var gltfMaterials: [GLTFMaterial]? = nil
        var samplers: [GLTFSampler]? = nil

        func addTexture(_ img: TexturePacker.EncodedImage, baseName: String) -> Int {
            let idx = imageSlots.count
            imageSlots.append(GLTFImageSlot(filename: "\(baseName).\(img.fileExtension)",
                                            mime: img.mime,
                                            bytes: img.bytes))
            gltfTextures.append(GLTFTexture(source: idx, sampler: 0))
            return idx   // texture index == image slot index (kept parallel)
        }

        var primitiveMaterial: Int? = nil
        if let mats = materials {
            samplers = [GLTFSampler(magFilter: GLTFFilter.linear,
                                    minFilter: GLTFFilter.linearMipmapLinear,
                                    wrapS: GLTFWrap.repeatWrap,
                                    wrapT: GLTFWrap.repeatWrap)]

            var pbr = GLTFPBRMetallicRoughness()

            // Base color (albedo) — required field in MaterialSet.
            if let albedo = TexturePacker.encodedImage(at: mats.albedoURL) {
                let t = addTexture(albedo, baseName: "albedo")
                pbr.baseColorTexture = GLTFTextureRef(index: t)
            } else {
                throw NimbusError.exportFailed("could not read albedo image at \(mats.albedoURL.lastPathComponent)")
            }

            // ORM: metallicRoughnessTexture (G/B) and occlusionTexture (R) share one texture.
            if let orm = TexturePacker.packORM(roughness: mats.roughnessURL,
                                               metallic: mats.metallicURL,
                                               ao: mats.ambientOcclusionURL,
                                               resolution: mats.textureResolution) {
                let t = addTexture(orm, baseName: "orm")
                pbr.metallicRoughnessTexture = GLTFTextureRef(index: t)
                pbr.metallicFactor = 1.0
                pbr.roughnessFactor = 1.0
            } else {
                // No packed maps: fall back to neutral scalar factors.
                pbr.metallicFactor = 0.0
                pbr.roughnessFactor = 0.8
            }

            var material = GLTFMaterial()
            material.name = "nimbus_" + mats.classification.materialClass.rawValue
            material.pbrMetallicRoughness = pbr
            material.doubleSided = true
            material.alphaMode = "OPAQUE"

            if let normal = mats.normalURL, let img = TexturePacker.encodedImage(at: normal) {
                let t = addTexture(img, baseName: "normal")
                material.normalTexture = GLTFNormalTextureRef(index: t, scale: 1.0)
            }
            if pbr.metallicRoughnessTexture != nil {
                material.occlusionTexture = GLTFOcclusionTextureRef(index: pbr.metallicRoughnessTexture!.index,
                                                                    strength: 1.0)
            }

            // Height + parallax live under extras (no core glTF slot).
            var extras = NimbusMaterialExtras()
            if let height = mats.heightURL, let img = TexturePacker.encodedImage(at: height) {
                extras.heightTexture = addTexture(img, baseName: "height")
            }
            let pom = mats.parallax
            if pom.enabled || extras.heightTexture != nil {
                extras.parallax = NimbusParallaxExtras(enabled: pom.enabled,
                                                       heightScale: pom.heightScale,
                                                       minSamples: pom.minSamples,
                                                       maxSamples: pom.maxSamples)
            }
            if extras.heightTexture != nil || extras.parallax != nil {
                material.extras = extras
            }

            gltfMaterials = [material]
            primitiveMaterial = 0
        }

        // Mesh / node / scene
        var attributes: [String: Int] = ["POSITION": posAccessor, "NORMAL": nrmAccessor]
        if let uv = uvAccessor { attributes["TEXCOORD_0"] = uv }
        let primitive = GLTFPrimitive(attributes: attributes,
                                      indices: indexAccessor,
                                      material: primitiveMaterial,
                                      mode: nil)
        let mesh0 = GLTFMesh(primitives: [primitive], name: "nimbus_mesh")
        let node0 = GLTFNode(mesh: 0, name: "nimbus_model", matrix: nil)
        let scene0 = GLTFScene(nodes: [0], name: "nimbus_scene", extras: sceneExtras)

        return GLTFAssembly(accessors: accessors,
                            bufferViews: bufferViews,
                            meshes: [mesh0],
                            nodes: [node0],
                            scenes: [scene0],
                            materials: gltfMaterials,
                            textures: gltfTextures.isEmpty ? nil : gltfTextures,
                            samplers: samplers,
                            geometryBin: bin,
                            imageSlots: imageSlots)
    }

    // MARK: Serialize

    private static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        // Deterministic + valid: stable key order, don't escape "/" in relative URIs.
        e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return e
    }

    /// Serializes a text `.gltf`: returns the JSON, the external `.bin`, and the list
    /// of image files to write alongside (all referenced by relative URI).
    func gltf(binFilename: String) throws -> (json: Data, bin: Data, files: [GLTFImageSlot]) {
        var doc = GLTFDocument()
        doc.scene = 0
        doc.scenes = scenes
        doc.nodes = nodes
        doc.meshes = meshes
        doc.accessors = accessors
        doc.bufferViews = bufferViews
        doc.buffers = [GLTFBufferJSON(byteLength: geometryBin.count, uri: binFilename)]
        doc.materials = materials
        doc.textures = textures
        doc.samplers = samplers
        if !imageSlots.isEmpty {
            doc.images = imageSlots.map { GLTFImage(uri: $0.filename, mimeType: $0.mime) }
        }
        let json = try GLTFAssembly.encoder().encode(doc)
        return (json, geometryBin, imageSlots)
    }

    /// Serializes a self-contained binary `.glb` (12-byte header + JSON chunk + BIN
    /// chunk, images embedded as buffer views).
    func glb() throws -> Data {
        var buffer = geometryBin
        var views = bufferViews
        var images: [GLTFImage] = []
        for slot in imageSlots {
            while buffer.count % 4 != 0 { buffer.append(0) }
            let offset = buffer.count
            buffer.append(slot.bytes)
            views.append(GLTFBufferView(byteOffset: offset,
                                        byteLength: slot.bytes.count,
                                        byteStride: nil,
                                        target: nil))
            images.append(GLTFImage(bufferView: views.count - 1, mimeType: slot.mime))
        }
        while buffer.count % 4 != 0 { buffer.append(0) }

        var doc = GLTFDocument()
        doc.scene = 0
        doc.scenes = scenes
        doc.nodes = nodes
        doc.meshes = meshes
        doc.accessors = accessors
        doc.bufferViews = views
        doc.buffers = [GLTFBufferJSON(byteLength: buffer.count, uri: nil)]
        doc.materials = materials
        doc.textures = textures
        doc.samplers = samplers
        if !images.isEmpty { doc.images = images }

        var json = try GLTFAssembly.encoder().encode(doc)
        while json.count % 4 != 0 { json.append(0x20) }   // pad JSON chunk with spaces

        let jsonChunkType: UInt32 = 0x4E4F534A   // "JSON"
        let binChunkType:  UInt32 = 0x004E4942   // "BIN\0"
        let magic:         UInt32 = 0x46546C67   // "glTF"
        let total = 12 + 8 + json.count + 8 + buffer.count

        var glb = Data(capacity: total)
        glb.append(LE.data(magic))
        glb.append(LE.data(2))                         // version
        glb.append(LE.data(UInt32(total)))
        glb.append(LE.data(UInt32(json.count)))
        glb.append(LE.data(jsonChunkType))
        glb.append(json)
        glb.append(LE.data(UInt32(buffer.count)))
        glb.append(LE.data(binChunkType))
        glb.append(buffer)
        return glb
    }
}
