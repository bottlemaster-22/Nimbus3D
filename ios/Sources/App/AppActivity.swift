//
//  AppActivity.swift
//
//  Build 422: whether the app is in the background, readable from any thread.
//
//  iOS refuses GPU work from a backgrounded app, so a long training run used
//  to fail outright if the owner switched apps or the phone locked. The
//  training loop reads this flag before each step, waits while it is set and
//  carries on where it was when the app comes back. A few seconds of
//  background time are asked for so the step already on the GPU can finish.
//

import Foundation
import UIKit

enum AppActivity {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var background = false
    private nonisolated(unsafe) static var observers: [NSObjectProtocol] = []
    private nonisolated(unsafe) static var task: UIBackgroundTaskIdentifier = .invalid

    static var isInBackground: Bool {
        lock.lock()
        defer { lock.unlock() }
        return background
    }

    private static func set(_ value: Bool) {
        lock.lock()
        background = value
        lock.unlock()
    }

    /// Called once at launch, on the main thread.
    @MainActor
    static func start() {
        guard observers.isEmpty else { return }
        let center = NotificationCenter.default
        observers.append(center.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
        ) { _ in
            AppActivity.set(true)
            MainActor.assumeIsolated {
                AppActivity.endTask()
                AppActivity.task = UIApplication.shared.beginBackgroundTask {
                    MainActor.assumeIsolated { AppActivity.endTask() }
                }
            }
        })
        observers.append(center.addObserver(
            forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main
        ) { _ in
            AppActivity.set(false)
            MainActor.assumeIsolated { AppActivity.endTask() }
        })
    }

    @MainActor
    private static func endTask() {
        guard task != .invalid else { return }
        UIApplication.shared.endBackgroundTask(task)
        task = .invalid
    }
}
