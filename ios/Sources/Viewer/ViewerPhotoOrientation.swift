//
//  ViewerPhotoOrientation.swift
//  Viewer
//
//  TURNING A CAPTURED PHOTO THE RIGHT WAY UP, FOR DISPLAY ONLY.
//
//  The JPEGs a scan writes are the raw sensor buffer, which is LANDSCAPE
//  however the phone was held, and they deliberately carry no EXIF orientation
//  tag (see `CapturePixelBuffer`): the pixels are stored in exactly the
//  orientation the intrinsics describe, which is what makes the COLMAP files,
//  the trainer and every export agree with each other. That is right and must
//  stay right.
//
//  It does mean that anything showing one of those photos has to turn it
//  itself, or a scan shot in portrait appears on its side. The number of
//  quarter turns is `CaptureSettings.imageQuarterTurnsClockwiseToUpright` where
//  a scan records it and `ViewerPoseMath.uprightQuarterTurns(of:)` where it
//  does not, it is the SAME number the preview camera is rolled by with
//  `Pose.rolledForDisplay(quarterTurnsClockwise:)`, and this is where it is
//  applied to the picture.
//
//  Nothing in this file writes to a file or copies a pixel: setting a UIImage's
//  orientation is a flag on the wrapper, and UIKit and SwiftUI both honour it
//  when they draw.
//

import UIKit

enum ViewerPhoto {

    /// The same photo, tagged so that it DRAWS turned that many quarter turns
    /// clockwise from the way its pixels are stored.
    ///
    /// A photo that already carries an orientation of its own is left exactly
    /// as it is: the app's own captures never do, so an image that does came
    /// from somewhere else and already knows which way up it goes.
    static func upright(_ image: UIImage, quarterTurnsClockwise turns: Int) -> UIImage {
        let steps = ((turns % 4) + 4) % 4
        guard steps != 0,
              image.imageOrientation == .up,
              let pixels = image.cgImage
        else { return image }

        let orientation: UIImage.Orientation
        switch steps {
        case 1:
            // A quarter turn clockwise. This is the same tag iOS itself puts
            // on a photo taken with the phone held upright in portrait.
            orientation = .right
        case 2:
            orientation = .down
        default:
            orientation = .left
        }
        return UIImage(cgImage: pixels, scale: image.scale, orientation: orientation)
    }
}
