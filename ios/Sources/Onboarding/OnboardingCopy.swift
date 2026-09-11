//
//  OnboardingCopy.swift
//  Onboarding
//
//  Every sentence the first-run screens show, in one file.
//
//  ---------------------------------------------------------------------------
//  THE RULES THIS FILE FOLLOWS
//  ---------------------------------------------------------------------------
//
//  1. Plain language. No jargon that is not immediately defined in the same
//     sentence. "The laser scanner (LiDAR)", never "LiDAR" on its own the
//     first time. "Memory", not "RAM". "Graphics chip", not "GPU".
//
//  2. Specific to THIS phone. Every negative sentence names the actual thing
//     that is missing, with the real number in it where there is one. Never
//     "insufficient resources", never "your device is not supported". If the
//     copy cannot name the reason, the reason was not measured, and that is a
//     bug in the probe rather than something to paper over here.
//
//  3. Warm, and never falsely hopeful. Telling someone their iPhone will never
//     do this is kinder than implying an update might fix it. Where a re-check
//     genuinely could change the answer (the phone is warm, another app has
//     the camera, iOS is being odd), the copy says so and the screen offers
//     the button. Where it could not, neither does.
//
//  4. The product is never named as a literal. `BrandConfig.displayName`.
//
//  5. Nothing here decides anything. `DeviceCompatibilityProbe` reaches the
//     verdict; this file only says it out loud. If a sentence here would need
//     to make a judgement to be written, the judgement belongs in the probe.
//

import Foundation

/// The words. See the file header for the rules they follow.
///
/// Internal rather than public because several of these take
/// `DeviceCompatibilityProbe.ReasonContext`, which is internal. Everything is
/// in one app target, so internal reaches every screen that needs it.
enum OnboardingCopy {

    // MARK: - Shorthand

    /// Saves writing `DeviceCompatibilityProbe.ReasonContext` a dozen times.
    typealias Context = DeviceCompatibilityProbe.ReasonContext

    private typealias Thresholds = DeviceCompatibilityProbe.Thresholds

    /// What we call the scanner the first time it appears in any sentence.
    /// Always the plain word first, the jargon in brackets after it.
    private static let scanner = "laser scanner (LiDAR)"

    // MARK: - The incompatible screen

    /// The apology, first line on the incompatible screen.
    static var incompatibleApology: String {
        "We are sorry. \(BrandConfig.displayName) cannot work on this iPhone."
    }

    /// The owner's exact line, verbatim. Second on the incompatible screen.
    static let tryingOurBest =
        "We are trying our best to support as many devices as possible."

    /// The big bold heading. A constant so it cannot drift.
    static let whyIncompatibleHeading = "Why is my device incompatible?"

    /// The answer to that heading when everything else has failed to produce a
    /// specific one. Reaching this is a bug, but the user still gets a true
    /// sentence rather than a blank space.
    static let lastResortReason =
        "This app measures the real distance to everything around you with a "
        + "laser scanner (LiDAR) that is built into some iPhone models. This "
        + "device does not have one, and there is no way to work that "
        + "measurement out from the camera picture alone."

    /// The honest, device-specific answer to "Why is my device incompatible?".
    ///
    /// The order of the checks below is the order of certainty: the most
    /// specific thing we actually measured wins. Each branch names the real
    /// cause on this phone, and each ends by saying what, if anything, the
    /// person can do about it.
    static func incompatibleReason(_ context: Context) -> String {
        let ctx = context

        // --- The Simulator. Not a phone, and not the user's fault.
        if ctx.model.isSimulator || ctx.lidar.isSimulator {
            return "This is the iPhone Simulator running on a Mac. The "
                + "Simulator has no cameras and no \(scanner), so there is "
                + "nothing here for \(BrandConfig.displayName) to measure. On "
                + "a real iPhone with the scanner, this screen does not "
                + "appear."
        }

        // --- No graphics chip. Everything this app does is drawn on it.
        if !ctx.metal.hasDevice {
            return "\(BrandConfig.displayName) does all of its work on the "
                + "graphics chip, and iOS did not hand this app one. On a real "
                + "iPhone that should never happen, so it is worth restarting "
                + "the phone and checking again. If it keeps happening after a "
                + "restart, the app cannot run here."
        }

        // --- The device has the scanner but ARKit is not offering it.
        //     A real and recoverable situation, and the one case where the
        //     honest answer is "this is probably temporary".
        if ctx.catalogDisagreesWithARKit {
            let name = ctx.model.marketingName ?? "This iPhone"
            return "\(name) does have the \(scanner), but iOS is not offering "
                + "it to \(BrandConfig.displayName) at the moment, so the app "
                + "has nothing to measure with. That is usually temporary. "
                + "Close anything else that is using the camera, restart the "
                + "phone, and tap Check again. If it still fails after a "
                + "restart, the scanner itself may need looking at."
        }

        // --- ARKit cannot track this device's movement at all. That is a
        //     chip-age limit, and it is worth saying plainly because it is a
        //     different problem from a missing scanner.
        if !ctx.lidar.supportsWorldTracking {
            var text = "\(BrandConfig.displayName) needs to know exactly where "
                + "the phone is as you walk around, which iOS only provides on "
                + "an A12 chip or newer."
            if let chip = ctx.model.chip?.name {
                text += " This device has the \(chip) chip."
            }
            text += " Without that, there is no way to tell where each "
                + "measurement was taken from, so the measurements cannot be "
                + "assembled into one scan."
            if ctx.model.catalogHasLiDAR == false {
                text += " This model does not have the \(scanner) either."
            }
            return text
        }

        // --- An iPad running an iPhone-only build.
        if ctx.model.isPad {
            let name = ctx.model.marketingName ?? "this iPad"
            var text = "\(BrandConfig.displayName) is built for iPhone, and "
                + "iOS is running it on \(name) in iPhone compatibility mode. "
                + "Here, iOS did not offer the app any depth measurements from "
                + "a \(scanner), and those measurements are the whole basis of "
                + "a scan."
            if ctx.model.catalogHasLiDAR == true {
                text += " Your iPad does have the scanner, so this is a limit "
                    + "of how the app is being run rather than of your "
                    + "hardware. An iPad version is not built yet."
            }
            return text
        }

        // --- A named iPhone that Apple does not fit a scanner to. The most
        //     common case by far, and the one that has to land kindly.
        if ctx.model.catalogHasLiDAR == false, let name = ctx.model.marketingName {
            return "\(name) does not have the \(scanner). "
                + "\(BrandConfig.displayName) fires that laser many times a "
                + "second and measures the real distance to every surface "
                + "around you, then builds the 3D model out of those "
                + "measurements. A camera on its own cannot produce them, and "
                + "guessing them from the picture is exactly what makes other "
                + "scanning apps bend walls and drift. On iPhone, Apple fits "
                + "the scanner only to the Pro and Pro Max models, starting "
                + "with iPhone 12 Pro."
        }

        // --- An iPhone we could not name, but that ARKit says has no depth.
        //     The identifier is included because it is the one fact a support
        //     conversation can start from.
        if ctx.model.isPhone {
            let name = ctx.model.marketingName ?? "this iPhone"
            return "iOS reports no \(scanner) on \(name) "
                + "(model \(ctx.model.identifier)). "
                + "\(BrandConfig.displayName) measures the real distance to "
                + "every surface with that laser and builds the 3D model from "
                + "those measurements, so on this phone it has nothing to work "
                + "from. On iPhone, Apple fits the scanner only to the Pro and "
                + "Pro Max models, starting with iPhone 12 Pro."
        }

        return lastResortReason
    }

    /// Whether tapping "Check again" could honestly change the verdict.
    ///
    /// A missing scanner is missing forever, and offering a button that cannot
    /// help is a small cruelty. These three situations are genuinely
    /// recoverable, so these three get the button.
    static func recheckCouldHelp(_ context: Context) -> Bool {
        context.catalogDisagreesWithARKit
            || !context.metal.hasDevice
            || context.model.isSimulator
            || context.lidar.isSimulator
    }

    // MARK: - Feature-list details

    /// The `detail` on the scanner row when the scanner is not there. Shorter
    /// than `incompatibleReason`, because it sits in a list rather than under
    /// a heading, and the full explanation is already on the same screen.
    static func noScannerFeatureDetail(_ context: Context) -> String {
        let ctx = context
        if ctx.model.isSimulator || ctx.lidar.isSimulator {
            return "The Simulator has no cameras and no scanner."
        }
        if ctx.catalogDisagreesWithARKit {
            return "This iPhone has the scanner, but iOS is not offering it to "
                + "the app right now. A restart usually fixes that."
        }
        if !ctx.lidar.supportsWorldTracking {
            return "iOS is not offering this app any of the scanning "
                + "measurements on this device."
        }
        if let name = ctx.model.marketingName {
            return "\(name) does not have the \(scanner). On iPhone it is "
                + "fitted to the Pro and Pro Max models only, from iPhone 12 "
                + "Pro onwards."
        }
        return "iOS reports no \(scanner) on this device "
            + "(model \(ctx.model.identifier))."
    }

    /// The `detail` on the on-device training row, in every tier.
    ///
    /// Always returns a sentence, including when the feature is available:
    /// people want to know what "build the 3D model on this phone" means
    /// before they want to know why it is unavailable.
    static func trainingFeatureDetail(tier: DeviceTier, context: Context) -> String {
        let ctx = context
        switch tier {
        case .full:
            var text = "The whole job happens on this phone. Nothing is "
                + "uploaded anywhere, and there is nothing to pay for."
            if ctx.performance.reducedByLowPowerMode {
                text += " Low Power Mode is on at the moment, which will make "
                    + "it noticeably slower until you turn it off."
            }
            if ctx.performance.reducedByHeat {
                text += " Your phone is warm right now, so it will start "
                    + "slower than usual and speed up as it cools."
            }
            return text

        case .limited:
            let why = limitedTierReasons(ctx)
            var text = "This phone can build the model, but at a smaller size "
                + "than the newest iPhones."
            if !why.isEmpty {
                text = "This phone can build the model, but at a smaller size, "
                    + "because \(OnboardingFormat.sentenceList(why))."
            }
            text += " A single object or one room will finish. A whole house "
                + "is better handed to a computer on your Wi-Fi, which is free "
                + "and completely optional."
            return text

        case .incompatible:
            return "There is no scan to build from on this device."
        }
    }

    /// The `detail` on the full-detail row. Only ever shown when that row is
    /// unavailable, so it must name the one thing that is holding it back.
    static func highDetailFeatureDetail(tier: DeviceTier, context: Context) -> String {
        let ctx = context

        if tier == .incompatible {
            return "There is nothing to build, at any level of detail, without "
                + "a scanner."
        }

        if tier == .limited {
            let why = limitedTierReasons(ctx)
            if why.isEmpty {
                return "The app will build the model at a size it is confident "
                    + "this phone can finish, rather than starting something "
                    + "that runs out of memory half way through. The PC helper "
                    + "removes that limit."
            }
            return "Full detail is held back because "
                + "\(OnboardingFormat.sentenceList(why)). The app will build "
                + "the model at a size it is confident this phone can finish, "
                + "rather than starting something that runs out of memory half "
                + "way through. The PC helper removes that limit."
        }

        // Full tier, so the block is either sustained performance or the
        // memory iOS is willing to give the app right now.
        if ctx.performance.reducedByLowPowerMode || ctx.performance.reducedByHeat {
            var causes: [String] = []
            if ctx.performance.reducedByLowPowerMode {
                causes.append("Low Power Mode is on")
            }
            if ctx.performance.reducedByHeat {
                causes.append(
                    "your phone is \(OnboardingFormat.thermal(ctx.performance.thermalLevel).lowercased()) "
                    + "at the moment"
                )
            }
            return "\(OnboardingFormat.sentenceList(causes).capitalisedFirst), "
                + "so the app is holding back on the largest models for now. "
                + "This is temporary. Tap Check again once it has changed."
        }

        if ctx.performance.baseClass < 2 {
            let chip = ctx.model.chip?.name ?? "this chip"
            return "Full detail is reserved for the chips that can hold their "
                + "speed for minutes at a time without overheating, and \(chip) "
                + "is not one of them. Everything else still works."
        }

        if !ctx.memory.availableIsEstimated,
            ctx.memory.availableBytes < Thresholds.highDetailMinimumAvailableMemoryBytes
        {
            return "iOS is currently letting this app use "
                + "\(OnboardingFormat.bytes(ctx.memory.availableBytes)) of "
                + "memory, and building at full detail needs about "
                + "\(OnboardingFormat.bytes(Thresholds.highDetailMinimumAvailableMemoryBytes)). "
                + "Closing a few other apps and tapping Check again usually "
                + "frees up enough."
        }

        return "The app could not confirm there is enough free memory for the "
            + "largest models right now, so it will build at a size it is "
            + "certain it can finish. Tap Check again after closing other apps."
    }

    /// The `detail` on the storage row when free space is below the
    /// comfortable threshold. Two bands, because "a bit tight" and "this will
    /// fail" deserve different sentences.
    static func storageDetail(freeBytes: Int64) -> String {
        let free = OnboardingFormat.bytes(freeBytes)

        if freeBytes < Thresholds.minimumUsefulScanDiskBytes {
            return "You have \(free) free, which is not enough for a scan of "
                + "any size. A few minutes of scanning writes a gigabyte or "
                + "more of photos and measurements. Free up some space first, "
                + "or the scan will stop part way through and you will have "
                + "walked the room for nothing."
        }

        return "You have \(free) free, which is enough for one object or one "
            + "small room. A whole house wants about "
            + "\(OnboardingFormat.bytes(Thresholds.comfortableScanDiskBytes)) "
            + "or more, because every second of scanning writes photos plus "
            + "the scanner's own measurements."
    }

    // MARK: - Why a device landed in the limited tier

    /// The measured reasons this phone is `.limited` rather than `.full`, as
    /// clauses that read after the word "because".
    ///
    /// Mirrors `DeviceCompatibilityProbe.tier(...)` exactly. If a threshold
    /// moves there and not here, the screen starts lying, so both read the
    /// same `Thresholds` constants and neither hard-codes a number.
    static func limitedTierReasons(_ context: Context) -> [String] {
        let ctx = context
        var reasons: [String] = []

        if let generation = ctx.chipGeneration {
            if generation < Thresholds.fullTierMinimumChipGeneration {
                let chip = ctx.model.chip?.name ?? "A\(generation)"
                reasons.append(
                    "the \(chip) chip is a few generations behind what "
                    + "full-speed building needs"
                )
            }
        } else {
            reasons.append(
                "the app could not work out which chip this device has, so it "
                + "is starting cautiously"
            )
        }

        if let family = ctx.metal.appleFamilyOrdinal {
            if family < Thresholds.fullTierMinimumMetalAppleFamily {
                reasons.append(
                    "this graphics chip is older than the app's building step "
                    + "was designed for"
                )
            }
        } else {
            reasons.append("the graphics chip did not report what it can do")
        }

        if ctx.memory.totalBytes < Thresholds.fullTierMinimumTotalMemoryBytes {
            reasons.append(
                "this device has \(OnboardingFormat.memory(ctx.memory.totalBytes)) "
                + "of memory"
            )
        }

        if !ctx.memory.availableIsEstimated,
            ctx.memory.availableBytes < Thresholds.fullTierMinimumAvailableMemoryBytes
        {
            reasons.append(
                "iOS is only letting this app use "
                + "\(OnboardingFormat.bytes(ctx.memory.availableBytes)) of "
                + "memory right now"
            )
        }

        return reasons
    }

    // MARK: - Tier headlines

    /// The headline on the "what this iPhone can do" page.
    static func tierHeadline(_ tier: DeviceTier) -> String {
        switch tier {
        case .full: return "This iPhone can do everything"
        case .limited: return "This iPhone can do most of it"
        case .incompatible: return "This iPhone cannot run the app"
        }
    }

    /// One or two sentences under that headline, naming what is actually
    /// different about this phone rather than congratulating it.
    static func tierSummary(_ tier: DeviceTier, context: Context?) -> String {
        switch tier {
        case .full:
            return "It has the \(scanner), and a chip fast enough to build the "
                + "finished 3D model here on the phone. Nothing you scan "
                + "leaves this device."

        case .limited:
            // No context means a screen is drawing from a bare report with no
            // measurements behind it. The sentence stays true, it just cannot
            // name the specific reason, so the generic branch below runs.
            let why: [String]
            if let context = context {
                why = limitedTierReasons(context)
            } else {
                why = []
            }
            var text = "It has the \(scanner), so scanning, checking and "
                + "previewing all work properly."
            if why.isEmpty {
                text += " Building the finished 3D model on the phone will be "
                    + "slower and smaller than on the newest iPhones."
            } else {
                text += " Building the finished 3D model on the phone will be "
                    + "slower and smaller, because "
                    + "\(OnboardingFormat.sentenceList(why))."
            }
            text += " For anything bigger than a room you can hand the work to "
                + "a computer on your Wi-Fi instead. That is free, optional, "
                + "and never touches the internet."
            return text

        case .incompatible:
            return incompatibleApology
        }
    }

    // MARK: - Section headings on the feature list

    static let worksHeading = "What works on this iPhone"
    static let doesNotWorkHeading = "What does not work on this iPhone"
    static let technicalDetailsHeading = "Technical details"

    static let featureListNote =
        "This list comes from what your phone actually reported a moment ago, "
        + "not from its model name."

    // MARK: - Welcome page

    static var welcomeTitle: String {
        "Welcome to \(BrandConfig.displayName)"
    }

    static let welcomeBody =
        "Point your phone at something and walk around it. The app measures "
        + "the real distance to every surface with the laser scanner built "
        + "into your iPhone, takes photos as it goes, and turns the two into a "
        + "3D model you can walk around, share, or open in Blender."

    static let welcomePoints: [(symbol: String, title: String, body: String)] = [
        (
            "figure.walk",
            "It tells you where to go",
            "The app paints the room as you cover it and says out loud, and "
            + "through the phone's vibration, when to walk around something, "
            + "get closer, or slow down."
        ),
        (
            "wifi.slash",
            "Nothing leaves your phone",
            "There is no account, no cloud, and no subscription. If you choose "
            + "to hand a big scan to a computer, it happens over your own "
            + "Wi-Fi and goes no further."
        ),
        (
            "ruler",
            "It measures instead of guessing",
            "Most scanning apps work out shape from the camera picture, which "
            + "is why walls bend. This one uses the laser's real distances, so "
            + "a straight wall stays straight."
        ),
    ]

    // MARK: - Good-scan tips page

    static let tipsTitle = "How to get a good scan"

    static let tipsBody =
        "You do not have to remember any of this. The app guides you while you "
        + "scan. It is here so the guidance makes sense when it starts talking."

    static let tips: [(symbol: String, title: String, body: String)] = [
        (
            "arrow.triangle.2.circlepath",
            "Walk all the way around",
            "A surface seen from one angle only is a guess. The app shades "
            + "everything you have seen from enough angles, and asks you to go "
            + "back to anything it has not."
        ),
        (
            "tortoise",
            "Move slowly and smoothly",
            "A blurred photo cannot be sharpened later. The app shows a live "
            + "blur meter and asks you to slow down before it becomes a "
            + "problem, rather than telling you afterwards."
        ),
        (
            "arrow.left.and.right",
            "Get within arm's reach for detail",
            "The laser is accurate to about five metres and is at its best up "
            + "close. If you want a detailed object, scan it from close by, "
            + "then step back for the room around it."
        ),
        (
            "sun.max",
            "Windows and mirrors need care",
            "Glass does not reflect the laser, so the app knows it cannot "
            + "measure through it. It will ask you to stand back and it will "
            + "briefly darken a few photos so the bright view outside is not "
            + "pure white. That is deliberate."
        ),
    ]

    // MARK: - Buttons

    static let continueButton = "Continue"
    static let backButton = "Back"
    static let startButton = "Start scanning"
    static let recheckButton = "Check again"
    static let recheckingLabel = "Checking..."

    static let recheckNoteRecoverable =
        "If you have just restarted the phone, closed another camera app, or "
        + "let it cool down, check again."

    static let recheckNoteGeneral =
        "This check is quick and does not use the camera. Run it again after "
        + "closing other apps, turning off Low Power Mode, or letting the "
        + "phone cool down."
}

// MARK: - Rebuilding the reason context from saved findings

extension DeviceCompatibilityFindings {

    /// The bag of measurements the copy functions read, reassembled from a set
    /// of findings.
    ///
    /// `DeviceCompatibilityProbe` builds one of these during a check and
    /// throws it away afterwards; the findings keep every field it held, so a
    /// screen drawing a saved report can rebuild it exactly rather than
    /// falling back to generic wording. Every field is copied straight across,
    /// so this cannot introduce a difference between what the probe decided
    /// and what a screen later explains.
    var reasonContext: DeviceCompatibilityProbe.ReasonContext {
        DeviceCompatibilityProbe.ReasonContext(
            model: model,
            lidar: lidar,
            metal: metal,
            memory: memory,
            performance: performance,
            chipGeneration: effectiveChipGeneration,
            catalogDisagreesWithARKit: catalogDisagreesWithARKit
        )
    }
}

// MARK: - Small string helper

extension String {
    /// Upper-cases the first character only, leaving the rest alone, so a
    /// clause built for the middle of a sentence can start one.
    ///
    /// `capitalized` would title-case every word and turn "Low Power Mode is
    /// on" into "Low Power Mode Is On".
    var capitalisedFirst: String {
        guard let first = first else { return self }
        return String(first).uppercased() + dropFirst()
    }
}
