//
//  MaterialLibrary.swift
//  Nimbus3D (Materials module)
//
//  Loader for the bundled ambientCG (CC0) tileable-material sample set.
//  The library is described by MaterialLibraryManifest.json; every texture file
//  has a globally unique name so it can be looked up flat in the app bundle
//  (XcodeGen copies resources without folder references).
//
//  The Dynamic Textures design substitutes class-appropriate tileable relief
//  from this library. It does NOT extract fine relief from the splat.
//

import Foundation

// MARK: - Coarse classes

/// The six coarse classes the on-device classifier and the confirm UI work with.
/// Each maps onto one of the shared contract's `MaterialClass` cases and back.
public enum CoarseMaterial: String, Codable, Sendable, CaseIterable, Identifiable {
    case masonry
    case fabric
    case granular
    case wood
    case metal
    case tile

    public var id: String { rawValue }

    /// Contract-level class this coarse class reports as.
    public var contractClass: MaterialClass {
        switch self {
        case .masonry:  return .brick
        case .fabric:   return .fabric
        case .granular: return .ground
        case .wood:     return .wood
        case .metal:    return .metal
        case .tile:     return .ceramic
        }
    }

    /// Nearest coarse class for a contract-level class. Nil when no bundled
    /// library material is a sensible substitute (plastic, glass, unknown).
    public init?(contractClass: MaterialClass) {
        switch contractClass {
        case .brick, .stone, .concrete: self = .masonry
        case .fabric, .leather:         self = .fabric
        case .ground, .foliage:         self = .granular
        case .wood:                     self = .wood
        case .metal:                    self = .metal
        case .ceramic:                  self = .tile
        case .plastic, .glass, .unknown: return nil
        }
    }

    public var displayName: String {
        switch self {
        case .masonry:  return "Masonry"
        case .fabric:   return "Fabric"
        case .granular: return "Granular"
        case .wood:     return "Wood"
        case .metal:    return "Metal"
        case .tile:     return "Tile"
        }
    }

    /// SF Symbol used by the confirm UI.
    public var symbolName: String {
        switch self {
        case .masonry:  return "building.2.fill"
        case .fabric:   return "tshirt.fill"
        case .granular: return "circle.grid.3x3.fill"
        case .wood:     return "tree.fill"
        case .metal:    return "gearshape.fill"
        case .tile:     return "square.grid.2x2.fill"
        }
    }
}

// MARK: - Manifest types

/// One bundled tileable material (a set of PBR map file names).
public struct MaterialLibraryEntry: Codable, Sendable, Equatable, Identifiable {
    public var assetID: String
    public var displayName: String
    public var coarseLabel: CoarseMaterial
    public var sourceURL: String
    /// Edge length in meters that one texture tile covers in the real world.
    public var physicalSizeMeters: Double
    public var maps: MaterialLibraryMaps

    public var id: String { assetID }
}

/// File names (bundle resources) for each PBR map of a library entry.
public struct MaterialLibraryMaps: Codable, Sendable, Equatable {
    public var color: String
    public var normalGL: String?
    public var roughness: String?
    public var displacement: String?
    public var ambientOcclusion: String?
    public var metalness: String?
}

struct MaterialLibraryManifest: Codable {
    var version: Int
    var license: String
    var entries: [MaterialLibraryEntry]
}

// MARK: - Library

/// Loads the bundled CC0 material library and resolves entries by material class.
public final class MaterialLibrary: Sendable {

    private let entriesByCoarse: [CoarseMaterial: MaterialLibraryEntry]
    private let bundle: Bundle
    public let licenseText: String

    /// Loads MaterialLibraryManifest.json from `bundle` (the app bundle by default).
    public init(bundle: Bundle = .main) throws {
        self.bundle = bundle
        guard let manifestURL = bundle.url(forResource: "MaterialLibraryManifest", withExtension: "json") else {
            throw NimbusError.textureBuildFailed("MaterialLibraryManifest.json is missing from the app bundle")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(MaterialLibraryManifest.self, from: data)
        self.licenseText = manifest.license
        var byCoarse: [CoarseMaterial: MaterialLibraryEntry] = [:]
        for entry in manifest.entries {
            byCoarse[entry.coarseLabel] = entry
        }
        self.entriesByCoarse = byCoarse
    }

    public var allEntries: [MaterialLibraryEntry] {
        CoarseMaterial.allCases.compactMap { entriesByCoarse[$0] }
    }

    /// Library entry for a coarse class, if bundled.
    public func entry(for coarse: CoarseMaterial) -> MaterialLibraryEntry? {
        entriesByCoarse[coarse]
    }

    /// Library entry for a contract-level material class, if any bundled material
    /// is a sensible substitute. Returns nil for plastic, glass, and unknown.
    public func entry(for materialClass: MaterialClass) -> MaterialLibraryEntry? {
        guard let coarse = CoarseMaterial(contractClass: materialClass) else { return nil }
        return entriesByCoarse[coarse]
    }

    /// Resolves a manifest file name to its URL inside the bundle.
    public func url(forMapFileName fileName: String) throws -> URL {
        let name = (fileName as NSString).deletingPathExtension
        let ext = (fileName as NSString).pathExtension
        guard let url = bundle.url(forResource: name, withExtension: ext) else {
            throw NimbusError.textureBuildFailed("Library texture \(fileName) is missing from the app bundle")
        }
        return url
    }
}
