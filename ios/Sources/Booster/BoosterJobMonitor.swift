//
//  BoosterJobMonitor.swift
//  Booster
//
//  Live progress for one Booster job over a WebSocket
//  (ws://host:port/v1/jobs/{jobID}/stream), with automatic reconnect and a
//  polling fallback (GET /v1/jobs/{jobID}) for the (expected to be rare) case
//  where the WebSocket cannot be established at all, e.g. a restrictive
//  router. Either path funnels into the same `events` stream so the UI never
//  needs to know which transport is actually in use.
//

import Foundation

actor BoosterJobMonitor {

    private let address: BoosterEndpointAddress
    private let jobID: String
    private var webSocketTask: URLSessionWebSocketTask?
    private var isCancelled = false

    init(address: BoosterEndpointAddress, jobID: String) {
        self.address = address
        self.jobID = jobID
    }

    /// An async stream of progress events. Finishes naturally once a
    /// terminal stage (ready / failed / cancelled) arrives, or the caller
    /// calls `stop()`.
    func events() -> AsyncStream<BoosterProgressEvent> {
        AsyncStream { continuation in
            let task = Task {
                await self.run(continuation: continuation)
            }
            continuation.onTermination = { [weak self] _ in
                Task { await self?.stop() }
                task.cancel()
            }
        }
    }

    func stop() {
        isCancelled = true
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
    }

    // MARK: - Private

    private func run(continuation: AsyncStream<BoosterProgressEvent>.Continuation) async {
        for delay in [0] + BoosterAPI.progressReconnectDelays {
            if isCancelled { break }
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            let finishedCleanly = await streamOnce(continuation: continuation)
            if finishedCleanly || isCancelled { break }
        }

        if !isCancelled {
            // The WebSocket never worked across every retry delay. Fall back
            // to polling so the user still sees progress, just less
            // smoothly.
            await pollUntilTerminal(continuation: continuation)
        }
        continuation.finish()
    }

    /// Returns true if the job reached a terminal stage (i.e. there is no
    /// reason to reconnect), false if the socket just dropped and a retry is
    /// worth attempting.
    private func streamOnce(
        continuation: AsyncStream<BoosterProgressEvent>.Continuation
    ) async -> Bool {
        guard let url = address.url(path: BoosterAPI.jobStream(jobID)) else { return true }
        var wsURL = URLComponents(url: url, resolvingAgainstBaseURL: false)
        wsURL?.scheme = "ws"
        guard let finalURL = wsURL?.url else { return true }

        var request = URLRequest(url: finalURL)
        if let token = address.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.setValue(BoosterAPI.apiVersion, forHTTPHeaderField: "X-Nimbus-Api-Version")

        let task = BoosterHTTP.session.webSocketTask(with: request)
        webSocketTask = task
        task.resume()

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        while !isCancelled {
            do {
                let message = try await task.receive()
                guard
                    let data = Self.data(from: message),
                    let event = try? decoder.decode(BoosterProgressEvent.self, from: data)
                else { continue }
                continuation.yield(event)
                if Self.isTerminal(event.stage) {
                    task.cancel(with: .normalClosure, reason: nil)
                    return true
                }
            } catch {
                // Socket dropped. Signal "not terminal" so the caller retries.
                return false
            }
        }
        return true
    }

    private func pollUntilTerminal(
        continuation: AsyncStream<BoosterProgressEvent>.Continuation
    ) async {
        while !isCancelled {
            do {
                let status: BoosterJobStatusResponse = try await BoosterHTTP.get(
                    address,
                    path: BoosterAPI.job(jobID)
                )
                let event = BoosterProgressEvent(
                    stage: status.stage,
                    fractionComplete: status.fractionComplete,
                    message: status.message,
                    timestamp: Date()
                )
                continuation.yield(event)
                if Self.isTerminal(status.stage) { return }
            } catch {
                // Keep polling; a transient network blip should not end the
                // whole job-monitoring session.
            }
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }
    }

    private static func isTerminal(_ stage: BoosterJobStage) -> Bool {
        switch stage {
        case .ready, .failed, .cancelled: return true
        case .queued, .receiving, .verifying, .training, .exporting: return false
        }
    }

    private static func data(from message: URLSessionWebSocketTask.Message) -> Data? {
        switch message {
        case .data(let data): return data
        case .string(let string): return Data(string.utf8)
        @unknown default: return nil
        }
    }
}
