//
//  CaptureLog.swift
//  Capture
//
//  One logger for the module, so a capture can be read back out of Console
//  without every file inventing its own subsystem string.
//

import Foundation
import os

/// `os.Logger` for this module. Subsystem comes from `BrandConfig` so a
/// product rename does not orphan the logs.
enum CaptureLog {
    static let session = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Session"
    )
    static let writer = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Writer"
    )
    static let coverage = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Coverage"
    )
    static let exposure = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Exposure"
    )
    static let guidance = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Guidance"
    )
    static let renderer = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Capture.Renderer"
    )
}
