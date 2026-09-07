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
//      POST <endpoint>   {bundle_id, installed_build_version, lan_ip, force?}
//      Authorization: Bearer <token>
//
//  The contract also offers GET <endpoint>/status, and this service does not
//  use it. It cannot: it has to be dead before the install can happen, so
//  there is nobody left to poll. The outcome is read on the next launch from
//  the build number instead.
//
//  Both routes answer with the same `status` enum and a plain-English
//  `message` written for a screen. No X-Bottle-Proto header.
//
//  `updated` means INSTALLED, not "signed and hosted". The Bottle agent caught
//  and fixed a revision that reported success at the hosting step, which would
//  have made this screen lie.
//
//  THE APP MUST QUIT ITSELF FOR THE INSTALL TO HAPPEN. iOS will not replace
//  a running application, so an install requested from inside the app it is
//  replacing simply stalls until the broker gives up, which is exactly the
//  timeout the owner hit. Bottle closes itself for the same reason. So this
//  service stops following the job and exits the process once the broker has
//  accepted it, and reports the outcome on NEXT launch instead.
//
//  `lan_ip` is REQUIRED in practice even though the contract calls it
//  optional. The installer reaches the phone through the relay bridge and has
//  no address of its own to dial, so a job without it signs successfully and
//  then fails with no_iphone_target. Sending it is the difference between an
//  install and a confusing half-success.
//

import Darwin
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

            // AND NOW GET OUT OF THE WAY.
            //
            // iOS will not replace a running application. An install driven
            // from inside the app being replaced cannot complete, so it sat
            // there until the broker timed out and reported that it could
            // not install. Polling harder would not have helped; the app's
            // continued existence WAS the failure.
            //
            // So the job is handed over and the process ends. Bottle does
            // the same thing for the same reason.
            Self.rememberPendingUpdate()
            status = .updating
            message = "Closing so the new build can be installed. Reopen "
                + "LiKOVA in a moment."
            await Self.quitForInstall()
            return
        } catch {
            status = .failed
            message = Self.sentence(for: error)
            log.error("Self-update failed: \(error.localizedDescription, privacy: .public)")
        }
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

        // Checked here rather than left to the broker to reject, because a
        // phone on cellular has no address the relay could reach and the
        // useful thing to say is why, not a 400.
        guard let lan = Self.wifiIPv4Address(), Self.isPrivateIPv4(lan) else {
            throw SelfUpdateError.noWiFiAddress
        }
        body["lan_ip"] = lan
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

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

    // MARK: - Standing aside for the installer

    private static var pendingKey: String {
        BrandConfig.defaultsPrefix + "selfupdate.pending"
    }

    /// Records that this build asked to be replaced, so the NEXT launch can
    /// say whether it worked.
    ///
    /// The build number is stored rather than a bare flag: on the next
    /// launch, a CFBundleVersion different from this one is proof the
    /// install happened, without asking the broker anything.
    private static func rememberPendingUpdate() {
        UserDefaults.standard.set(installedBuildVersion, forKey: pendingKey)
    }

    /// Ends the process so the installer can replace the bundle.
    ///
    /// `exit(0)` is a blunt instrument and Apple discourages it, because a
    /// shipping app should never terminate itself. This one is not
    /// shipping: the whole self-update feature is development-only and is
    /// stripped before any submission, which is the condition the owner set
    /// when he agreed to it. If this file ever survives into a release
    /// build, THIS is the line that fails review.
    ///
    /// The pause is so the sentence above is readable before the screen
    /// disappears. Without it the app vanishes the instant the button is
    /// pressed, which is indistinguishable from a crash, and this app has
    /// spent enough of its life being indistinguishable from a crash.
    private static func quitForInstall() async {
        try? await Task.sleep(nanoseconds: 1_800_000_000)
        UserDefaults.standard.synchronize()
        exit(0)
    }

    /// Called when the update screen appears. Reports the outcome of an
    /// update this app quit to allow.
    ///
    /// No network call is needed for the happy path. The build number that
    /// asked to be replaced was written down before quitting, so if this
    /// launch reports a DIFFERENT CFBundleVersion then the installer did its
    /// job, and saying so from local facts is more trustworthy than asking
    /// the broker to grade its own homework.
    public func resumeAfterRelaunch() {
        let defaults = UserDefaults.standard
        guard let asked = defaults.string(forKey: Self.pendingKey) else { return }
        defaults.removeObject(forKey: Self.pendingKey)

        if asked != Self.installedBuildVersion {
            status = .updated
            message = "Updated from build \(asked) to build "
                + "\(Self.installedBuildVersion)."
            // One string literal, not a concatenation: the argument is an
            // OSLogMessage, and `+` is not defined on those.
            log.notice(
                "Self-update completed: \(asked, privacy: .public) -> \(Self.installedBuildVersion, privacy: .public)"
            )
        } else {
            // Same build back again. The install did not happen, and the
            // honest reading is that it failed rather than that nothing was
            // asked for.
            status = .failed
            message = "The update did not install. This build is still "
                + "\(Self.installedBuildVersion). Check the relay at home is "
                + "powered on and try again."
            log.error(
                "Self-update did not take: still build \(Self.installedBuildVersion, privacy: .public)"
            )
        }
    }

    // MARK: - This phone's address

    /// The IPv4 address of this phone on Wi-Fi, or nil when it has none.
    ///
    /// The broker signs the build and then has to CONNECT BACK to the phone
    /// through the relay bridge, and it has no address of its own to dial.
    /// Without this the job signs and then fails with no_iphone_target,
    /// which is a success followed by a confusing failure.
    ///
    /// en0 is Wi-Fi on iPhone. Cellular is pdp_ip0 and is deliberately not
    /// accepted: a carrier address is not reachable from a machine at home,
    /// so offering it would trade a clear "you are not on Wi-Fi" for an
    /// install that fails somewhere less legible.
    static func wifiIPv4Address() -> String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return nil }
        defer { freeifaddrs(head) }

        for interface in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(interface.pointee.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            guard let address = interface.pointee.ifa_addr,
                  address.pointee.sa_family == UInt8(AF_INET),
                  String(cString: interface.pointee.ifa_name) == "en0"
            else { continue }

            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                address, socklen_t(address.pointee.sa_len),
                &host, socklen_t(host.count),
                nil, 0, NI_NUMERICHOST
            )
            if result == 0 { return String(cString: host) }
        }
        return nil
    }

    /// True for the private ranges the broker accepts: 10/8, 172.16-31 and
    /// 192.168/16.
    ///
    /// PARSED DIGIT BY DIGIT, NOT MATCHED WITH A REGULAR EXPRESSION, and
    /// that is a deliberate choice rather than a style preference. The
    /// Bottle agent shipped this same validator as a regex, had its
    /// backslashes eaten when the file was written, and ended up with a
    /// checker that rejected every valid address including the phone's own.
    /// The same escaping hazard has bitten this repository repeatedly today
    /// in its patch scripts. A validator that can be silently corrupted by
    /// the act of writing it is worth avoiding when plain arithmetic does
    /// the same job and cannot be.
    static func isPrivateIPv4(_ text: String) -> Bool {
        let parts = text.split(
            separator: ".", omittingEmptySubsequences: false
        )
        guard parts.count == 4 else { return false }

        var octets: [Int] = []
        for part in parts {
            guard !part.isEmpty, part.count <= 3,
                  part.allSatisfy({ $0.isASCII && $0.isNumber }),
                  let value = Int(part), value <= 255
            else { return false }
            octets.append(value)
        }

        switch (octets[0], octets[1]) {
        case (10, _): return true
        case (172, 16...31): return true
        case (192, 168): return true
        default: return false
        }
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
        case SelfUpdateError.noWiFiAddress:
            return "This iPhone is not on Wi-Fi, so the machine at home has no "
                + "way to reach it. Join the same network and try again."
        default:
            return "Could not reach the update server: \(error.localizedDescription)"
        }
    }
}

// MARK: - Errors

enum SelfUpdateError: Error {
    case badEndpoint
    case noWiFiAddress
    case rejected(status: Int, message: String?)
    case unreadableReply(message: String?)
}
