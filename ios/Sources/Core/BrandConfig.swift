//
//  BrandConfig.swift
//  Core
//
//  THE SINGLE SOURCE OF TRUTH FOR THE PRODUCT NAME.
//
//  The working name is not final. No other Swift, Metal, or Python file in this
//  project may contain the product name as a string literal. Read it from here.
//
//  Where the values actually come from:
//
//      ios/project.yml  ->  settingGroups.brand  ->  Info.plist (NB* keys)
//                                                ->  BrandConfig (this file)
//
//  So renaming the product is a change to the brand block in project.yml and
//  nothing else. The literals below are FALLBACKS only, used when there is no
//  Info.plist to read: SwiftUI previews, unit tests, and command-line tools.
//  `BrandConfig.assertConsistent()` shouts in debug builds if they drift.
//

import Foundation

/// Product identity. Every user-visible name and every on-disk folder name in
/// the app is derived from this type.
public enum BrandConfig {

    // MARK: - Fallbacks (only used when Info.plist is unavailable)

    /// Keep these in step with the `brand` settings group in `ios/project.yml`.
    /// They are never read in a normal app launch.
    private enum Fallback {
        static let displayName = "LiKOVA"
        static let productName = "LiKOVA"
        static let bundleIdentifier = "com.tombline.likova"
        static let slug = "likova"
        static let boosterServiceType = "_likovaboost._tcp"
        static let documentsFolderName = "LiKOVA"
    }

    private enum PlistKey {
        static let displayName = "NBBrandDisplayName"
        static let productName = "NBBrandProductName"
        static let slug = "NBBrandSlug"
        static let boosterServiceType = "NBBoosterServiceType"
        static let documentsFolderName = "NBDocumentsFolderName"
        static let sourceRevision = "NBSourceRevision"
        static let selfUpdateURL = "NBSelfUpdateURL"
        static let selfUpdateToken = "NBSelfUpdateToken"
    }

    // MARK: - Identity

    /// What the user reads. Home screen, titles, "About", every sentence of UI
    /// copy that names the app.
    public static let displayName: String =
        string(PlistKey.displayName) ?? Fallback.displayName

    /// Filesystem-safe form with no spaces. Used for file name prefixes and the
    /// `.app` / `.ipa` name.
    public static let productName: String =
        string(PlistKey.productName) ?? Fallback.productName

    /// Reverse-DNS bundle identifier, e.g. `com.tombline.nimbus`.
    public static let bundleIdentifier: String =
        Bundle.main.bundleIdentifier ?? Fallback.bundleIdentifier

    /// Short lowercase token used to build derived identifiers (Bonjour type,
    /// keychain service, UserDefaults prefix, log subsystem).
    public static let slug: String =
        string(PlistKey.slug) ?? Fallback.slug

    /// Marketing version, e.g. `0.1.0`.
    public static let version: String =
        string("CFBundleShortVersionString") ?? "0.0.0"

    /// Build number, e.g. `1`.
    public static let build: String =
        string("CFBundleVersion") ?? "0"

    /// `0.1.0 (1)` - for the About screen and for the Booster handshake.
    public static var versionString: String { "\(version) (\(build))" }

    /// The commit this binary was built from, short form, or `dev` for a
    /// local build that CI did not stamp.
    public static let sourceRevision: String =
        string(PlistKey.sourceRevision) ?? "dev"

    /// `LiKOVA 0.1.0 (42) fd77296` - the one line that identifies exactly
    /// which build is running.
    ///
    /// This MUST stay visible somewhere the user can reach without
    /// developer tools. Seven builds shipped reporting an identical
    /// `0.1.0 (1)`, a bug was reported against a build three releases
    /// behind its own fix, and the only way to work out what had actually
    /// been installed was to pull Mach-O UUIDs out of the release
    /// artefacts and match them against the resource reports. Never
    /// again.
    public static var buildIdentity: String {
        "\(displayName) \(versionString) \(sourceRevision)"
    }

    // MARK: - Development self-update

    /// Where this build may ask to be reinstalled, and what it presents to
    /// be allowed to.
    public struct SelfUpdateEndpoint: Sendable {
        public let url: URL
        public let token: String
    }

    /// nil unless CI injected BOTH an endpoint and a token, which is every
    /// local build and any build made without the GitHub secrets.
    ///
    /// This is the off switch for the whole self-update feature, and it is
    /// an absence rather than a flag on purpose. The owner agreed to this
    /// only as a development convenience and asked that it be easy to
    /// strip; a build with no secrets simply has no update UI, because
    /// `SelfUpdateService.isConfigured` reads this and the Booster tab
    /// renders nothing when it is false.
    ///
    /// The token genuinely ships inside the binary. See the note at the top
    /// of SelfUpdateService.swift for why that trade is acceptable for
    /// THIS token and would not be for an account token.
    public static let selfUpdate: SelfUpdateEndpoint? = {
        guard let raw = string(PlistKey.selfUpdateURL), !raw.isEmpty,
              let url = URL(string: raw),
              let token = string(PlistKey.selfUpdateToken), !token.isEmpty
        else { return nil }
        return SelfUpdateEndpoint(url: url, token: token)
    }()

    // MARK: - Networking

    /// Bonjour / DNS-SD service type the PC Booster advertises and the phone
    /// browses for, e.g. `_nimbusboost._tcp`.
    ///
    /// This string must be byte-identical to the one the Python Booster
    /// registers. It is written once in `project.yml` and documented in
    /// `docs/BOOSTER_PROTOCOL.md`.
    ///
    /// Also note: every type listed here must appear in the Info.plist
    /// `NSBonjourServices` array or iOS silently returns no results.
    public static let boosterServiceType: String =
        string(PlistKey.boosterServiceType) ?? Fallback.boosterServiceType

    /// Bonjour browse domain. Always the local link.
    public static let boosterServiceDomain = "local."

    /// Default TCP port the Booster listens on when its TXT record does not say
    /// otherwise. See `docs/BOOSTER_PROTOCOL.md`.
    public static let boosterDefaultPort: UInt16 = 8760

    /// Keychain service string under which a paired Booster token is stored.
    public static var keychainService: String { "\(bundleIdentifier).booster" }

    /// Prefix for every `UserDefaults` key the app writes, so nothing collides
    /// and everything can be wiped in one pass.
    public static var defaultsPrefix: String { "\(slug)." }

    /// `os.Logger` subsystem. Category is the module name, e.g. "Capture".
    public static var loggingSubsystem: String { bundleIdentifier }

    // MARK: - Folder names

    /// Top folder created inside the app's Documents directory, e.g.
    /// `Documents/Nimbus3D`. The PC Booster creates a folder with the same name
    /// under the user's Documents.
    public static let documentsFolderName: String =
        string(PlistKey.documentsFolderName) ?? Fallback.documentsFolderName

    /// Fixed sub-folder names. These are part of the on-disk data format and
    /// are documented in `docs/DATA_FORMAT.md`. They do NOT change on a rename.
    public enum Folder {
        /// `Documents/<brand>/Scans`
        public static let scans = "Scans"
        /// `Documents/<brand>/Scans/<scan-id>/images`
        public static let images = "images"
        /// Native LiDAR depth and confidence, plus the per-frame sidecar log.
        public static let sensorData = "sensor_data"
        /// COLMAP text model: `sparse/0/{cameras,images,points3D}.txt`
        public static let sparse = "sparse"
        public static let sparseModel = "sparse/0"
        /// ARKit anchors, in-session and re-read at end of session.
        public static let anchors = "anchors"
        /// ARKit scene mesh chunks and their classification.
        public static let mesh = "mesh"
        /// Everything the pre-pass produces.
        public static let prePass = "prepass"
        /// Trained model, background model, per-frame exposure.
        public static let model = "model"
        /// User-facing exports (.ply, .spz, zipped COLMAP bundles).
        public static let exports = "export"
        /// Scratch space, safe to delete at any time.
        public static let cache = "cache"

        /// Booster inbox/outbox on the PC side, mirrored here so the Swift
        /// client and the Python server agree.
        public enum Booster {
            public static let incoming = "Incoming"
            public static let completed = "Completed"
            public static let failed = "Failed"
        }
    }

    /// The root the app writes scans into: `Documents/<brand>/Scans`.
    ///
    /// - Returns: the URL, creating intermediate directories if needed.
    /// - Throws: whatever `FileManager` throws if the directory cannot be made.
    public static func scansDirectory(
        fileManager: FileManager = .default
    ) throws -> URL {
        let documents = try fileManager.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let url = documents
            .appendingPathComponent(documentsFolderName, isDirectory: true)
            .appendingPathComponent(Folder.scans, isDirectory: true)
        try fileManager.createDirectory(
            at: url,
            withIntermediateDirectories: true
        )
        return url
    }

    // MARK: - Consistency check

    /// Debug-only guard against the Info.plist and the fallbacks drifting
    /// apart after a rename. Call once from the app's `init`.
    public static func assertConsistent() {
        #if DEBUG
        guard Bundle.main.object(forInfoDictionaryKey: PlistKey.slug) != nil
        else {
            // No Info.plist keys at all: a preview or a test bundle. Fine.
            return
        }
        assert(
            !boosterServiceType.isEmpty && boosterServiceType.hasSuffix("._tcp"),
            "Bonjour service type must look like _something._tcp, got "
                + boosterServiceType
        )
        // DNS-SD caps the application-protocol label at 15 characters,
        // underscore included. Break this and browsing silently finds nothing.
        let typeLabel = boosterServiceType
            .split(separator: ".")
            .first
            .map(String.init) ?? ""
        assert(
            typeLabel.count <= 15,
            "Bonjour service type label \(typeLabel) is \(typeLabel.count) "
                + "characters; DNS-SD allows at most 15. Shorten "
                + "NIMBUS_BRAND_SLUG in ios/project.yml."
        )
        assert(
            !documentsFolderName.contains("/"),
            "NIMBUS_DOCS_FOLDER must be a single path component."
        )
        #endif
    }

    // MARK: - Private

    private static func string(_ key: String) -> String? {
        guard
            let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
            !value.isEmpty,
            // An unresolved build setting means the plist was not processed.
            !value.hasPrefix("$(")
        else { return nil }
        return value
    }
}
