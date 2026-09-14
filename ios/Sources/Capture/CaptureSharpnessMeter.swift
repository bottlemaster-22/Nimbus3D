//
//  CaptureSharpnessMeter.swift
//  Capture
//
//  Variance of the Laplacian, on the camera's own luma plane, normalised
//  against the running maximum for this session.
//
//  Two design notes worth keeping:
//
//  1. It runs on the Y plane of the ARKit pixel buffer directly. Converting to
//     RGB first would cost a full-frame pass and change nothing: sharpness is
//     a luminance property.
//
//  2. It is normalised against a session running maximum rather than an
//     absolute threshold, because the absolute variance of a textured brick
//     wall and a white painted wall differ by an order of magnitude while
//     both being perfectly in focus. `FrameQC.sharpness` is documented as
//     exactly this: 0...1 against the running maximum for this session.
//

import CoreVideo
import Foundation

/// Measures per-frame sharpness from the luma plane of an ARKit frame.
///
/// Not thread-safe by design: it is owned by, and only ever touched from, the
/// capture pipeline's serial queue.
final class CaptureSharpnessMeter {

    /// Sampling stride through the luma plane. At 1920x1440 a stride of 4
    /// leaves ~172k samples, which is far more than enough for a variance and
    /// a quarter of the memory traffic of the full plane.
    private static let stride = 4

    /// The running maximum decays slowly so that one exceptionally detailed
    /// frame early in a walk does not permanently flatten every later score.
    /// 0.999 per frame is a half-life of about 690 frames, roughly a minute.
    private static let maximumDecay: Float = 0.999

    private var runningMaximum: Float = 0

    /// Raw variance of the Laplacian for the most recent frame, before
    /// normalisation. Kept for the log; the QC record stores the normalised
    /// value because that is what the contract specifies.
    private(set) var lastRawVariance: Float = 0

    /// Computes normalised sharpness in 0...1 for `pixelBuffer`.
    ///
    /// - Returns: 0 when the buffer cannot be read, which is honest: an
    ///   unreadable frame is not a sharp one.
    func sharpness(of pixelBuffer: CVPixelBuffer) -> Float {
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 1 else { return 0 }
        guard
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess
        else { return 0 }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard
            let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        else { return 0 }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        guard width > 8, height > 8 else { return 0 }

        let luma = base.assumingMemoryBound(to: UInt8.self)
        let step = Self.stride

        // Four-neighbour Laplacian: 4*c - up - down - left - right. Sampled on
        // the strided grid, so the kernel spans `step` pixels; that is a
        // deliberate low-pass, and it is what makes the measure insensitive to
        // sensor noise while still collapsing when the frame smears.
        var sum: Double = 0
        var sumOfSquares: Double = 0
        var count: Int = 0

        var y = step
        while y < height - step {
            let rowCenter = luma + y * bytesPerRow
            let rowUp = luma + (y - step) * bytesPerRow
            let rowDown = luma + (y + step) * bytesPerRow
            var x = step
            while x < width - step {
                let center = Int(rowCenter[x])
                let laplacian =
                    4 * center
                    - Int(rowUp[x])
                    - Int(rowDown[x])
                    - Int(rowCenter[x - step])
                    - Int(rowCenter[x + step])
                let value = Double(laplacian)
                sum += value
                sumOfSquares += value * value
                count += 1
                x += step
            }
            y += step
        }

        guard count > 0 else { return 0 }
        let mean = sum / Double(count)
        let variance = Swift.max(0, sumOfSquares / Double(count) - mean * mean)
        let raw = Float(variance)
        lastRawVariance = raw

        runningMaximum = Swift.max(runningMaximum * Self.maximumDecay, raw)
        guard runningMaximum > 0 else { return 0 }
        return Swift.max(0, Swift.min(1, raw / runningMaximum))
    }

    /// Forgets the running maximum. Called when a new session starts so one
    /// scan's normalisation cannot leak into the next.
    func reset() {
        runningMaximum = 0
        lastRawVariance = 0
    }
}
