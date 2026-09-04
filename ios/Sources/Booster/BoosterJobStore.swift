//
//  BoosterJobStore.swift
//  Booster
//
//  Small on-disk record of: which Boosters this phone has paired with (their
//  boosterID, display name, and last-known Bonjour instance name, so
//  BoosterDiscovery can recognise a device it sees on the LAN before a fresh
//  /v1/info round trip confirms it), and the history of jobs sent to them -
//  the "Results" list on the Booster tab.
//
//  Lives at Documents/<brand>/cache/booster_state.json
//  (BrandConfig.Folder.cache), so it is explicitly scratch/rebuildable data:
//  losing it only means re-pairing and losing the results *list* (the actual
//  downloaded result files, already saved under each scan's own model/
//  folder, are untouched).
//

import Foundation

public struct BoosterPairingRecord: Codable, Sendable {
    public var boosterID: String
    public var boosterName: String
    public var bonjourName: String
    public var pairedAt: Date
}

public struct BoosterJobRecord: Codable, Identifiable, Sendable {
    public var id: String { jobID }
    public var jobID: String
    public var scanID: String
    public var boosterID: String
    public var boosterName: String
    public var createdAt: Date
    public var stage: BoosterJobStage
    public var message: String
    public var resultDirectory: String?

    public init(
        jobID: String,
        scanID: String,
        boosterID: String,
        boosterName: String,
        createdAt: Date,
        stage: BoosterJobStage,
        message: String,
        resultDirectory: String?
    ) {
        self.jobID = jobID
        self.scanID = scanID
        self.boosterID = boosterID
        self.boosterName = boosterName
        self.createdAt = createdAt
        self.stage = stage
        self.message = message
        self.resultDirectory = resultDirectory
    }
}

private struct BoosterState: Codable {
    var pairings: [BoosterPairingRecord] = []
    var jobs: [BoosterJobRecord] = []
}

public final class BoosterJobStore: @unchecked Sendable {

    public static let shared = BoosterJobStore()

    private let queue = DispatchQueue(label: "booster.job-store", qos: .utility)
    private var state: BoosterState
    private let fileURL: URL?

    private init() {
        let url = Self.stateFileURL()
        self.fileURL = url
        if let url, let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder.boosterState.decode(BoosterState.self, from: data)
        {
            self.state = decoded
        } else {
            self.state = BoosterState()
        }
    }

    // MARK: - Pairings

    public func recordPairing(boosterID: String, boosterName: String, bonjourName: String) {
        queue.sync {
            state.pairings.removeAll { $0.boosterID == boosterID }
            state.pairings.append(
                BoosterPairingRecord(
                    boosterID: boosterID,
                    boosterName: boosterName,
                    bonjourName: bonjourName,
                    pairedAt: Date()
                )
            )
            persist()
        }
    }

    public func forgetPairing(boosterID: String) {
        queue.sync {
            state.pairings.removeAll { $0.boosterID == boosterID }
            persist()
        }
        BoosterKeychain.deleteToken(boosterID: boosterID)
    }

    public func boosterID(forBonjourName bonjourName: String) -> String? {
        queue.sync { state.pairings.first { $0.bonjourName == bonjourName }?.boosterID }
    }

    public func displayName(forBoosterID boosterID: String) -> String? {
        queue.sync { state.pairings.first { $0.boosterID == boosterID }?.boosterName }
    }

    public func allPairings() -> [BoosterPairingRecord] {
        queue.sync { state.pairings }
    }

    // MARK: - Jobs

    public func upsertJob(_ record: BoosterJobRecord) {
        queue.sync {
            if let index = state.jobs.firstIndex(where: { $0.jobID == record.jobID }) {
                state.jobs[index] = record
            } else {
                state.jobs.insert(record, at: 0)
            }
            persist()
        }
    }

    public func allJobs() -> [BoosterJobRecord] {
        queue.sync { state.jobs.sorted { $0.createdAt > $1.createdAt } }
    }

    public func removeJob(jobID: String) {
        queue.sync {
            state.jobs.removeAll { $0.jobID == jobID }
            persist()
        }
    }

    // MARK: - Private

    private func persist() {
        guard let fileURL, let data = try? JSONEncoder.boosterState.encode(state) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }

    private static func stateFileURL() -> URL? {
        guard
            let documents = try? FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        else { return nil }
        let cacheDirectory = documents
            .appendingPathComponent(BrandConfig.documentsFolderName, isDirectory: true)
            .appendingPathComponent(BrandConfig.Folder.cache, isDirectory: true)
        try? FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        return cacheDirectory.appendingPathComponent("booster_state.json")
    }
}

extension JSONEncoder {
    fileprivate static let boosterState: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
}

extension JSONDecoder {
    fileprivate static let boosterState: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
