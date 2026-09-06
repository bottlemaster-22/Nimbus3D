//
//  SelfUpdateService.swift
//  Booster
//
//  ASKING THE BOTTLE BROKER TO REINSTALL THIS APP, OVER THE AIR.
//
//  DEVELOPMENT ONLY, AND DELIBERATELY EASY TO REMOVE. The owner's condition
//  when he agreed to this was that it be strippable once the app is worth a
//  paid developer account: "Once it is done in development and I can see it
//  being worthy of the money to spend on the developer subscription, it should
//  be easily stripped."
//
//  So the whole feature is this one file plus one Section in BoosterTabView,
//  and it turns itself off rather than needing to be switched off. If
//  `BrandConfig.selfUpdate` is nil, which is what happens on any build where CI
//  did not inject the endpoint and token, `isConfigured` is false and the UI
//  does not render. Deleting this file and that Section removes it entirely.
//
//  ---------------------------------------------------------------------------
//  WHY THE TOKEN BEING IN THE APP IS ACCEPTABLE HERE, AND WHERE IT IS NOT
//  ---------------------------------------------------------------------------
//  The publish token lives in GitHub Actions and never leaves a server. This
//  one cannot: the phone is the thing making the request, so the token ships
//  inside the binary and anyone holding the .ipa can read it out. That is not
//  a solved problem, it is a traded one, and the trade is only acceptable
//  because of what the token can do:
//
//    * it is scoped by the broker to ONE bundle id and ONE action,
//    * the action is "reinstall the current build of LiKOVA onto the owner's
//      own pinned device", which is not a capability worth stealing,
//    * it is not the owner's Bottle account token, which the Bottle agent
//      explicitly refused to share for exactly this reason,
//    * and the whole feature is temporary.
//
//  It is NOT injected from a literal in this repo. CI passes it as a build
//  setting from a GitHub secret, so it is absent from source control and from
//  every local build. If this feature ever outlives development, the token has
//  to move behind something the phone authenticates to, not sit in the bundle.
//
//  ---------------------------------------------------------------------------
//  THE CONTRACT
//  ---------------------------------------------------------------------------
//  Written against the Bottle agent's specification. Their message names
//  docs/SELFUPDATE_CONTRACT.md as authoritative; that file is not in the
//  bottles repository yet (only BUILD_CONTRACT.md and RESIGN_CONTRACT.md are),
//  so this is built from the summary and should be re-checked against the doc
//  when it lands.
//
//      POST <endpoint>            {bundle_id, installed_build_version, force?}
//      GET  <endpoint>/status?bundle_id=...
//      Authorization: Bearer <token>
//
//  Both routes answer with the same `status` enum and a plain-English
//  `message` written for a screen. No X-Bottle-Proto header.
//
//  `updated` means INSTALLED, not "signed and hosted". The Bottle agent caught
//  and fixed a revision that reported success at the hosting step, which would
//  have made this screen lie.
//

import Foundation
import os

// MARK: - Service

/// Asks the broker to reinstall this app, and reports what happened in words
/// the owner can act on.
@MainActor
public final class SelfUpdateService: ObservableObject {

    /// The broker's status enum, shared by both routes.
    public enum Status: String, Sendable {
        /// The installed build is already the newest one. No job was started.
        case current
        /// A signing or install job is running.
        case updating
        /// INSTALLED. Not merely signed and hosted.
        case updated
        /// The job ran and did not finish. `message` says why.
        case failed
        /// Nothing in progress.
        case idle
    }

    @Published public private(set) var status: Status = .idle
    /// The broker's sentence, meant to be shown as-is. Never invented here: if
    /// the broker did not send one, this stays nil and the UI says nothing
    /// rather than guessing on its behalf.
    @Published public private(set) var message: String?
    @Published public private(set) var isWorking = false

    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem, category: "Booster.SelfUpdate"
    )

    /// The broker asked for no faster than one poll every two seconds.
    private static let pollIntervalNanoseconds: UInt64 = 2_000_000_000
    /// Stops a stuck job polling forever. Ten minutes at two seconds.
    private static let maximumPolls = 300

    public init() {}

    /// False on any build CI did not configure, which is every local build.
    /// The UI keys off this, so an unconfigured build has no update section at
    /// all rather than a button that cannot work.
    public static var isConfigured: Bool { BrandConfig.selfUpdate != nil }

    /// What this build is, as the broker wants it: the CFBundleVersion string.
    public static var installedBuildVersion: String { BrandConfig.build }

    // MARK: Driving it

    /// Ask for an update, then follow it to a terminal state.
    ///
    /// - Parameter force: only for an EXPIRED app. The broker's no-op on an
    ///   already-current build is what makes an ordinary check cheap, but it
    ///   would also refuse the 7-day certificate re-sign that an expired app
    ///   needs to rescue itself, and an expired app cannot be launched to press
    ///   this button anyway if it has already lapsed. Force does not bypass the
    ///   broker's cooldown.
    public func update(force: Bool = false) async {
        guard let config = BrandConfig.selfUpdate else {
            log.error("Self-update asked for on a build with no endpoint configured.")
            return
        }
        guard !isWorking else { return }

        isWorking = true
        message = nil
        defer { isWorking = false }

        do {
            let started = try await post(config: config, force: force)
            apply(started)
            // `current` starts no job, so there is nothing to follow. Every
            // other non-terminal answer means the broker is working.
            guard started.status == .updating else { return }
            try await follow(config: config)
        } catch {
            status = .failed
            message = Self.sentence(for: error)
            log.error("Self-update failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func follow(config: BrandConfig.SelfUpdateEndpoint) async throws {
        for _ in 0..<Self.maximumPolls {
            try await Task.sleep(nanoseconds: Self.pollIntervalNanoseconds)
            if Task.isCancelled { return }
            let reply = try await status(config: config)
            apply(reply)
            switch reply.status {
            case .updating:
                continue
            case .current, .updated, .failed, .idle:
                return
            }
        }
        status = .failed
        message = "The update is taking longer than expected. It may still finish "
            + "on its own; check again in a few minutes."
    }

    private func apply(_ reply: Reply) {
        status = reply.status
        // Only overwrite the sentence when the broker sent one. A terminal
        // state with no message keeps whatever explanation came before it.
        if let text = reply.message, !text.isEmpty { message = text }
        log.notice(
            "Self-update status \(reply.status.rawValue, privacy: .public)"
        )
    }

    // MARK: Transport

    private struct Reply {
        var status: Status
        var message: String?
    }

    private func post(
        config: BrandConfig.SelfUpdateEndpoint, force: Bool
    ) async throws -> Reply {
        var request = URLRequest(url: config.url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = [
            "bundle_id": BrandConfig.bundleIdentifier,
            "installed_build_version": Self.installedBuildVersion
        ]
        if force { body["force"] = true }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        return try await send(request)
    }

    private func status(
        config: BrandConfig.SelfUpdateEndpoint
    ) async throws -> Reply {
        var components = URLComponents(
            url: config.url.appendingPathComponent("status"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "bundle_id", value: BrandConfig.bundleIdentifier)
        ]
        guard let url = components?.url else { throw SelfUpdateError.badEndpoint }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 30
        request.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        return try await send(request)
    }

    private func send(_ request: URLRequest) async throws -> Reply {
        let (data, response) = try await URLSession.shared.data(for: request)

        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let text = (object?["message"] as? String).flatMap { $0.isEmpty ? nil : $0 }

        // A rejection carries a message too, and the broker writes those for a
        // screen, so an HTTP error is reported in ITS words rather than a
        // status code the owner cannot act on.
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw SelfUpdateError.rejected(status: http.statusCode, message: text)
        }

        guard let raw = object?["status"] as? String, let parsed = Status(rawValue: raw)
        else { throw SelfUpdateError.unreadableReply(message: text) }

        return Reply(status: parsed, message: text)
    }

    // MARK: Words

    private static func sentence(for error: Error) -> String {
        switch error {
        case SelfUpdateError.rejected(_, .some(let text)):
            return text
        case SelfUpdateError.rejected(let code, nil):
            return "The update server refused the request (code \(code))."
        case SelfUpdateError.unreadableReply(.some(let text)):
            return text
        case SelfUpdateError.unreadableReply(nil):
            return "The update server sent something this app could not read."
        case SelfUpdateError.badEndpoint:
            return "This build's update address is not usable."
        default:
            return "Could not reach the update server: \(error.localizedDescription)"
        }
    }
}

// MARK: - Errors

enum SelfUpdateError: Error {
    case badEndpoint
    case rejected(status: Int, message: String?)
    case unreadableReply(message: String?)
}
