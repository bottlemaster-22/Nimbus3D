//
//  CapturePixelBuffer.swift
//  Capture
//
//  ONE deep copy of the camera image, taken while the `ARFrame` is still
//  alive, so every later stage can work off the ARKit delivery thread.
//
//  WHY A COPY AND NOT THE ARFRAME. `ARFrame.capturedImage` is only valid for
//  as long as ARKit's frame object is retained, and ARKit's own guidance is
//  that holding frames stalls the session - the pool is small and a retained
//  frame is a frame the tracker cannot reuse. Encoding a 1920x1440 JPEG takes
//  something like 10-15 ms, which is most of a 60 Hz delivery slot, so it
//  cannot happen on the delegate thread either. A 4 MB plane-wise memcpy is
//  about a millisecond, and it is the honest way out: copy fast, release the
//  frame, encode later.
//
//  `@unchecked Sendable` is load-bearing and deliberate. `CVPixelBuffer` is a
//  CoreFoundation type with no `Sendable` conformance, but the buffer this
//  struct wraps was created by this struct, is never handed to anything that
//  writes to it, and is passed to exactly one serial queue. That is the
//  invariant the `@unchecked` is asserting; do not widen it.
//

import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import simd

/// A private copy of one camera frame, plus the two things the pipeline needs
/// from it: JPEG bytes and point colours.
struct CapturePixelBuffer: @unchecked Sendable {

    let buffer: CVPixelBuffer
    let width: Int
    let height: Int

    /// Deep-copies `source` into a buffer this struct owns.
    ///
    /// - Returns: nil when the copy could not be made, which the caller must
    ///   treat as "this frame was not written", never as "this frame was
    ///   written without an image".
    init?(copying source: CVPixelBuffer) {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }

        // IOSurface backing is required for CoreImage and Metal to accept the
        // buffer without a further copy of their own.
        let attributes: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
        ]

        var copy: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            sourceWidth,
            sourceHeight,
            format,
            attributes as CFDictionary,
            &copy
        )
        guard status == kCVReturnSuccess, let destination = copy else {
            let message = "CVPixelBufferCreate failed with \(status); "
                    + "this frame is dropped rather than written without pixels."
            CaptureLog.writer.error("\(message, privacy: .public)")
            return nil
        }

        guard
            CVPixelBufferLockBaseAddress(source, .readOnly) == kCVReturnSuccess
        else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        guard CVPixelBufferLockBaseAddress(destination, []) == kCVReturnSuccess
        else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(destination, []) }

        let planeCount = CVPixelBufferGetPlaneCount(source)
        if planeCount == 0 {
            guard
                let from = CVPixelBufferGetBaseAddress(source),
                let to = CVPixelBufferGetBaseAddress(destination)
            else { return nil }
            let fromStride = CVPixelBufferGetBytesPerRow(source)
            let toStride = CVPixelBufferGetBytesPerRow(destination)
            let rowBytes = Swift.min(fromStride, toStride)
            for row in 0..<sourceHeight {
                memcpy(to + row * toStride, from + row * fromStride, rowBytes)
            }
        } else {
            guard planeCount == CVPixelBufferGetPlaneCount(destination) else {
                return nil
            }
            for plane in 0..<planeCount {
                guard
                    let from = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                    let to = CVPixelBufferGetBaseAddressOfPlane(destination, plane)
                else { return nil }
                let fromStride = CVPixelBufferGetBytesPerRowOfPlane(source, plane)
                let toStride = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
                let rows = CVPixelBufferGetHeightOfPlane(source, plane)
                let rowBytes = Swift.min(fromStride, toStride)
                for row in 0..<rows {
                    memcpy(to + row * toStride, from + row * fromStride, rowBytes)
                }
            }
        }

        self.buffer = destination
        self.width = sourceWidth
        self.height = sourceHeight
    }

    // MARK: - JPEG

    /// One `CIContext` for the whole module. Creating one per frame allocates
    /// a fresh Metal command queue and a shader cache every time, which is the
    /// classic way to make a capture loop stutter. `CIContext` is documented as
    /// safe to use from multiple threads.
    private static let ciContext = CIContext(options: [.cacheIntermediates: false])

    private static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)
        ?? CGColorSpaceCreateDeviceRGB()

    /// JPEG bytes at `docs/DATA_FORMAT.md` section 5's quality, sRGB, pixels in
    /// the same orientation the intrinsics describe - no EXIF orientation tag,
    /// because a reader that honours it and a reader that ignores it would then
    /// disagree about which way up the scan is.
    func jpegData(quality: CGFloat = CaptureTuning.jpegQuality) -> Data? {
        let image = CIImage(cvPixelBuffer: buffer)
        return Self.ciContext.jpegRepresentation(
            of: image,
            colorSpace: Self.sRGB,
            options: [
                CIImageRepresentationOption(
                    rawValue: kCGImageDestinationLossyCompressionQuality as String
                ): quality
            ]
        )
    }

    // MARK: - Colour sampling
    //
    // Used for the `R G B` columns of `sparse/0/points3D.txt`. Lock once, take
    // tens of thousands of samples, unlock: locking per sample would cost more
    // than the conversion.

    /// Locks the buffer for the batch of `rgb(atNormalizedX:y:)` calls that
    /// follows. Every `beginSampling` must be paired with an `endSampling`.
    @discardableResult
    func beginSampling() -> Bool {
        CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
    }

    func endSampling() {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }

    /// sRGB colour at a normalised image coordinate, 0...1 with the origin at
    /// the top-left of the image the intrinsics describe.
    ///
    /// Must be called between `beginSampling()` and `endSampling()`.
    func rgb(atNormalizedX u: Float, y v: Float) -> SIMD3<UInt8> {
        let clampedU = Swift.max(0, Swift.min(0.999_999, u))
        let clampedV = Swift.max(0, Swift.min(0.999_999, v))
        let x = Int(clampedU * Float(width))
        let y = Int(clampedV * Float(height))

        if CVPixelBufferGetPlaneCount(buffer) >= 2 {
            return biplanarRGB(x: x, y: y)
        }
        return interleavedRGB(x: x, y: y)
    }

    // MARK: - Private colour paths

    private func biplanarRGB(x: Int, y: Int) -> SIMD3<UInt8> {
        guard
            let lumaBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
            let chromaBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)
        else { return SIMD3<UInt8>(0, 0, 0) }

        let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let chromaWidth = CVPixelBufferGetWidthOfPlane(buffer, 1)
        let chromaHeight = CVPixelBufferGetHeightOfPlane(buffer, 1)

        let luma = lumaBase.assumingMemoryBound(to: UInt8.self)
        let chroma = chromaBase.assumingMemoryBound(to: UInt8.self)

        // 4:2:0, so the chroma plane is half resolution in both axes.
        let cx = Swift.min(chromaWidth - 1, x / 2)
        let cy = Swift.min(chromaHeight - 1, y / 2)

        var yValue = Float(luma[y * lumaStride + x])
        var cb = Float(chroma[cy * chromaStride + cx * 2])
        var cr = Float(chroma[cy * chromaStride + cx * 2 + 1])

        // Video range packs luma into 16...235 and chroma into 16...240; full
        // range uses the whole byte. Getting this wrong shows up as washed-out
        // or crushed point colours, which is exactly the kind of quiet error
        // nobody chases down later.
        let format = CVPixelBufferGetPixelFormatType(buffer)
        if format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            yValue = (yValue - 16) * (255.0 / 219.0)
            cb = (cb - 128) * (255.0 / 224.0) + 128
            cr = (cr - 128) * (255.0 / 224.0) + 128
        }

        let cbOffset = cb - 128
        let crOffset = cr - 128

        let r: Float
        let g: Float
        let b: Float
        if isRec709 {
            r = yValue + 1.574_80 * crOffset
            g = yValue - 0.187_324 * cbOffset - 0.468_124 * crOffset
            b = yValue + 1.855_60 * cbOffset
        } else {
            r = yValue + 1.402 * crOffset
            g = yValue - 0.344_136 * cbOffset - 0.714_136 * crOffset
            b = yValue + 1.772 * cbOffset
        }

        return SIMD3<UInt8>(
            UInt8(Swift.max(0, Swift.min(255, r))),
            UInt8(Swift.max(0, Swift.min(255, g))),
            UInt8(Swift.max(0, Swift.min(255, b)))
        )
    }

    private func interleavedRGB(x: Int, y: Int) -> SIMD3<UInt8> {
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            return SIMD3<UInt8>(0, 0, 0)
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let offset = y * stride + x * 4
        let format = CVPixelBufferGetPixelFormatType(buffer)
        if format == kCVPixelFormatType_32BGRA {
            return SIMD3<UInt8>(pixels[offset + 2], pixels[offset + 1], pixels[offset])
        }
        // 32RGBA and anything else four-byte: take the first three channels.
        return SIMD3<UInt8>(pixels[offset], pixels[offset + 1], pixels[offset + 2])
    }

    /// Which YCbCr matrix the buffer was tagged with. ARKit tags its frames, so
    /// this is read rather than assumed.
    private var isRec709: Bool {
        let attachment = CVBufferGetAttachment(
            buffer,
            kCVImageBufferYCbCrMatrixKey,
            nil
        )?.takeUnretainedValue()
        guard let value = attachment else { return false }
        return CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_709_2)
    }
}
