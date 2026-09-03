//
//  USDZExporter.swift — Export module
//
//  Optional USDZ output via Apple's Model I/O. This is REAL code (builds an MDLMesh +
//  MDLMaterial and calls MDLAsset.export), but USDZ *writing* is not available on every
//  OS build. We gate on `MDLAsset.canExportFileExtension("usdz")` at runtime and return
//  nil (rather than faking success) when the platform can't write it — the caller treats
//  USDZ as best-effort and records the outcome in the manifest.
//

import Foundation
import simd

#if canImport(ModelIO)
import ModelIO
#endif

enum USDZExporter {

    /// Attempts to write `<destination>`. Returns the URL on success, or nil when USDZ
    /// export is unsupported on this OS build or the geometry could not be assembled.
    static func export(mesh: MeshAsset, materials: MaterialSet?, to destination: URL) -> URL? {
        #if canImport(ModelIO)
        guard MDLAsset.canExportFileExtension("usdz") else {
            // TODO(nimbus): platform cannot write USDZ via Model I/O here; if USDZ becomes
            //   a hard requirement, bundle a USD writer or produce it in a macOS/tool step.
            return nil
        }
        guard !mesh.positions.isEmpty,
              mesh.normals.count == mesh.positions.count,
              !mesh.indices.isEmpty else { return nil }

        let hasUVs = mesh.uvs.count == mesh.positions.count
        let allocator = MDLMeshBufferDataAllocator()

        // Interleaved vertex layout: position(12) + normal(12) + uv(8) = 32 bytes.
        let stride = 32
        var vertexData = Data(capacity: mesh.positions.count * stride)
        for i in 0..<mesh.positions.count {
            let p = mesh.positions[i]
            let n = mesh.normals[i]
            let uv = hasUVs ? mesh.uvs[i] : SIMD2<Float>(0, 0)
            LE.append(p.x, to: &vertexData); LE.append(p.y, to: &vertexData); LE.append(p.z, to: &vertexData)
            LE.append(n.x, to: &vertexData); LE.append(n.y, to: &vertexData); LE.append(n.z, to: &vertexData)
            LE.append(uv.x, to: &vertexData); LE.append(uv.y, to: &vertexData)
        }
        var indexData = Data(capacity: mesh.indices.count * 4)
        for idx in mesh.indices { LE.append(idx, to: &indexData) }

        let vertexBuffer = allocator.newBuffer(with: vertexData, type: .vertex)
        let indexBuffer = allocator.newBuffer(with: indexData, type: .index)

        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                      format: .float3, offset: 0, bufferIndex: 0)
        descriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                                      format: .float3, offset: 12, bufferIndex: 0)
        descriptor.attributes[2] = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                                                      format: .float2, offset: 24, bufferIndex: 0)
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: stride)

        let material = buildMaterial(materials, hasUVs: hasUVs)
        let submesh = MDLSubmesh(indexBuffer: indexBuffer,
                                 indexCount: mesh.indices.count,
                                 indexType: .uInt32,
                                 geometryType: .triangles,
                                 material: material)

        let mdlMesh = MDLMesh(vertexBuffer: vertexBuffer,
                              vertexCount: mesh.positions.count,
                              descriptor: descriptor,
                              submeshes: [submesh])

        let asset = MDLAsset(bufferAllocator: allocator)
        asset.add(mdlMesh)
        do {
            try asset.export(to: destination)
            return destination
        } catch {
            return nil
        }
        #else
        return nil
        #endif
    }

    #if canImport(ModelIO)
    private static func buildMaterial(_ materials: MaterialSet?, hasUVs: Bool) -> MDLMaterial {
        let scattering = MDLPhysicallyPlausibleScatteringFunction()
        let material = MDLMaterial(name: "nimbus", scatteringFunction: scattering)
        guard let mats = materials, hasUVs else { return material }

        func addTexture(_ url: URL?, _ semantic: MDLMaterialSemantic, _ name: String) {
            guard let url = url else { return }
            material.setProperty(MDLMaterialProperty(name: name, semantic: semantic, url: url))
        }
        addTexture(mats.albedoURL, .baseColor, "baseColor")
        addTexture(mats.normalURL, .tangentSpaceNormal, "normal")
        addTexture(mats.roughnessURL, .roughness, "roughness")
        addTexture(mats.metallicURL, .metallic, "metallic")
        addTexture(mats.ambientOcclusionURL, .ambientOcclusion, "ao")
        return material
    }
    #endif
}
