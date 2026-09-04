//
//  CaptureGuidanceEngine.swift
//  Capture
//
//  AUDIO AND HAPTICS AS THE PRIMARY GUIDANCE CHANNEL (F9).
//
//  This is not a decoration on top of the HUD. While someone is scanning a
//  room they are looking THROUGH the phone at the wall, walking backwards
//  around a sofa, holding the phone at arm's length above their head to get
//  the ceiling. They are not reading a status line. Sound and vibration reach
//  them in all of those positions and text does not, so the rule in this
//  module is: anything worth showing is worth saying, and the screen is the
//  redundant copy.
//
//  Three channels, deliberately different in character so they never sound
//  like one another:
//
//    TICKS      a short, dry haptic tap plus a click, repeating while motion
//               blur is in the red. Rate is fixed, intensity tracks the blur.
//               This is the "you are moving too fast" channel and it needs no
//               interpretation at all - it is the sound of the scan being
//               damaged, and it stops the instant you slow down.
//    SPEECH     one plain sentence, at most one every four seconds, only after
//               the condition has held for over a second. "Walk around it a
//               bit more." Not jargon, not a code, not a beep the user has to
//               learn.
//    CHIMES     a rising two-note figure when coverage crosses the done line;
//               a falling one when tracking is lost. Two sounds, both
//               unmistakable, neither used for anything else.
//
//  ON THE TONE GENERATOR. There are no audio assets in this project, and a
//  bundled sound file would be one more thing to keep in step with a rename.
//  `AVAudioSourceNode` synthesises the tones directly, which is exact, tiny,
//  and has no licensing question attached.
//
//  ON THE RENDER-THREAD PARAMETERS. The render block runs on Core Audio's
//  real-time thread, where taking a lock or allocating is how you get a
//  dropout. Frequency and amplitude therefore live in a small manually
//  allocated buffer written from the main actor and read from the render
//  thread without synchronisation. That is a deliberate benign race on two
//  `Float`s: a torn read of an audio parameter is inaudible, and it is the
//  standard way this is done. Do not "fix" it with a lock.
//

import AVFoundation
import CoreHaptics
import Foundation

/// Speaks, chimes and taps.
///
/// `@MainActor` because it owns an `AVAudioEngine`, a `CHHapticEngine` and a
/// speech synthesiser, all of which want a stable owning thread.
@MainActor
final class CaptureGuidanceEngine {

    /// What the user may switch off. Both default on: the whole point is that
    /// guidance reaches someone who is not looking at the screen.
    var isSoundEnabled = true
    var isHapticsEnabled = true

    /// The sentence currently worth showing, mirrored into
    /// `CaptureLiveState.guidanceHint`.
    private(set) var currentHint: String?

    // MARK: - Audio

    private let audioEngine = AVAudioEngine()
    private var toneNode: AVAudioSourceNode?
    private let synthesizer = AVSpeechSynthesizer()

    /// [0] frequency in Hz, [1] amplitude 0...1, [2] phase. Written from the
    /// main actor, read from the audio render thread. See the note at the top
    /// of this file before changing how this is accessed.
    private let toneParameters = UnsafeMutablePointer<Float>.allocate(capacity: 3)

    private var sampleRate: Double = 44_100

    // MARK: - Haptics

    private var hapticEngine: CHHapticEngine?
    private var hapticsAvailable = false

    // MARK: - Pacing

    private var lastSpokenAt: TimeInterval = 0
    private var candidateHint: String?
    private var candidateSince: TimeInterval = 0
    private var lastTickAt: TimeInterval = 0
    private var announcedDone = false
    private var lastTrackingWasNormal = true

    init() {
        toneParameters[0] = 660
        toneParameters[1] = 0
        toneParameters[2] = 0
    }

    deinit {
        toneParameters.deallocate()
    }

    // MARK: - Lifecycle

    func start() {
        configureAudioSession()
        startAudioEngine()
        startHaptics()
        reset()
    }

    func stop() {
        toneParameters[1] = 0
        synthesizer.stopSpeaking(at: .immediate)
        if audioEngine.isRunning { audioEngine.stop() }
        if let toneNode {
            audioEngine.detach(toneNode)
            self.toneNode = nil
        }
        hapticEngine?.stop()
        hapticEngine = nil
        hapticsAvailable = false
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }

    func reset() {
        currentHint = nil
        candidateHint = nil
        candidateSince = 0
        lastSpokenAt = 0
        lastTickAt = 0
        announcedDone = false
        lastTrackingWasNormal = true
    }

    // MARK: - The one entry point

    /// Re-evaluates what, if anything, to tell the user.
    ///
    /// Called at the HUD's rate, not at frame rate. Everything inside is
    /// priority-ordered and debounced, so calling it more often changes
    /// nothing except how quickly a genuinely new condition is noticed.
    func update(
        blurPixels: Float,
        trackingQuality: TrackingQuality,
        worstChannel: CaptureCoverageChannel?,
        windowHint: String?,
        coverageFraction: Float,
        isDone: Bool,
        now: TimeInterval
    ) {
        // 1. Tracking loss beats everything. Nothing being written is worth
        //    saying about a scan that has stopped knowing where it is.
        if trackingQuality != .normal {
            if lastTrackingWasNormal {
                playChime(rising: false)
                lastTrackingWasNormal = false
            }
            setHint(Self.trackingHint(for: trackingQuality), now: now, urgent: true)
            return
        }
        if !lastTrackingWasNormal {
            lastTrackingWasNormal = true
        }

        // 2. Blur ticks. Continuous, not debounced: this is feedback on what
        //    the user's hands are doing right now, and a delay would make it
        //    feel like it belonged to some earlier movement.
        if blurPixels >= CaptureTuning.blurRedPixels {
            if now - lastTickAt >= CaptureTuning.guidanceBlurTickIntervalSeconds {
                lastTickAt = now
                let intensity = Swift.min(
                    1,
                    blurPixels / (CaptureTuning.blurRedPixels * 2)
                )
                playTap(intensity: intensity)
                playBlip(frequency: 1_100, seconds: 0.05, amplitude: 0.18)
            }
            setHint("Slow down, the picture is smearing", now: now, urgent: true)
            return
        }

        // 3. Coverage done, announced exactly once.
        if isDone, !announcedDone {
            announcedDone = true
            playChime(rising: true)
            speak("That is enough. You can stop whenever you like.", now: now)
            currentHint = "Coverage looks good"
            return
        }

        // 4. Window mode.
        if let windowHint {
            setHint(windowHint, now: now, urgent: false)
            return
        }

        // 5. The coverage channel with the largest deficit, and its own fix.
        if let worstChannel, !isDone {
            setHint(worstChannel.fixHint, now: now, urgent: false)
            return
        }

        // 6. Nothing needs saying. Say nothing; do not fill the silence with
        //    a percentage the user can already see.
        _ = coverageFraction
        currentHint = nil
        candidateHint = nil
    }

    // MARK: - Hints

    /// Debounces, then speaks. A hint must stay true for over a second before
    /// it is worth interrupting for, and no two sentences arrive closer
    /// together than four seconds. Urgent hints show immediately but still
    /// respect the speech interval, so a fast pan cannot turn the app into a
    /// stream of chatter.
    private func setHint(_ text: String, now: TimeInterval, urgent: Bool) {
        if urgent {
            currentHint = text
        }

        if candidateHint != text {
            candidateHint = text
            candidateSince = now
            if !urgent { return }
        }

        guard now - candidateSince >= CaptureTuning.guidanceHintDebounceSeconds
        else { return }

        currentHint = text

        // One sentence every four seconds at most, whether it is the same
        // sentence again or a different one. A hint repeated sooner is
        // nagging; a new hint sooner is chatter.
        guard now - lastSpokenAt >= CaptureTuning.guidanceSpeechMinIntervalSeconds
        else { return }

        speak(text, now: now)
    }

    private static func trackingHint(for quality: TrackingQuality) -> String {
        switch quality {
        case .limitedExcessiveMotion:
            return "Moving too fast to keep track. Slow right down."
        case .limitedInsufficientFeatures:
            return "Not enough detail here. Point at something with pattern on it."
        case .limitedRelocalizing:
            return "Finding the room again. Go back to where you just were."
        case .limitedInitializing:
            return "Starting up. Move the phone gently side to side."
        case .notAvailable:
            return "Tracking has stopped."
        case .normal:
            return ""
        }
    }

    // MARK: - Speech

    private func speak(_ text: String, now: TimeInterval) {
        lastSpokenAt = now
        guard isSoundEnabled, !text.isEmpty else { return }
        // A new sentence replaces an old one rather than queueing behind it:
        // guidance about what the user was doing four seconds ago is worse
        // than silence.
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.96
        utterance.volume = 0.9
        utterance.postUtteranceDelay = 0
        synthesizer.speak(utterance)
    }

    // MARK: - Tones

    private func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            // `.playback` with `.mixWithOthers`: this app never records audio
            // (there is deliberately no NSMicrophoneUsageDescription), and it
            // has no business silencing the user's music while they scan.
            try session.setCategory(
                .playback,
                mode: .default,
                options: [.mixWithOthers, .duckOthers]
            )
            try session.setActive(true)
            sampleRate = session.sampleRate > 0 ? session.sampleRate : 44_100
        } catch {
            let message = "Audio session refused to configure: "
                    + "\(error.localizedDescription). Guidance "
                    + "falls back to haptics and the on-screen hint."
            CaptureLog.guidance.error("\(message, privacy: .public)")
        }
    }

    private func startAudioEngine() {
        guard toneNode == nil else { return }
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: sampleRate,
                channels: 1
            )
        else { return }

        let parameters = toneParameters
        let rate = Float(sampleRate)
        let node = AVAudioSourceNode(format: format) {
            _, _, frameCount, audioBufferList -> OSStatus in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let frequency = parameters[0]
            let amplitude = parameters[1]
            var phase = parameters[2]
            let increment = 2 * Float.pi * frequency / rate

            for frame in 0..<Int(frameCount) {
                let value = sin(phase) * amplitude
                phase += increment
                if phase > 2 * Float.pi { phase -= 2 * Float.pi }
                for buffer in buffers {
                    let samples = buffer.mData?.assumingMemoryBound(to: Float.self)
                    samples?[frame] = value
                }
            }
            parameters[2] = phase
            return noErr
        }

        audioEngine.attach(node)
        audioEngine.connect(node, to: audioEngine.mainMixerNode, format: format)
        toneNode = node

        do {
            try audioEngine.start()
        } catch {
            let message = "Audio engine would not start: "
                    + "\(error.localizedDescription)"
            CaptureLog.guidance.error("\(message, privacy: .public)")
        }
    }

    /// A short tone. Amplitude is stepped rather than ramped because at 50 ms
    /// the attack is inaudible and a ramp would need a timer per blip.
    private func playBlip(frequency: Float, seconds: Double, amplitude: Float) {
        guard isSoundEnabled, audioEngine.isRunning else { return }
        toneParameters[0] = frequency
        toneParameters[1] = amplitude
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            self?.toneParameters[1] = 0
        }
    }

    /// Two notes. Rising means "you are done"; falling means "tracking is
    /// gone". Nothing else in the app uses either figure.
    private func playChime(rising: Bool) {
        guard isSoundEnabled else { return }
        let first: Float = rising ? 660 : 880
        let second: Float = rising ? 990 : 550
        playBlip(frequency: first, seconds: 0.12, amplitude: 0.22)
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 130_000_000)
            self?.playBlip(frequency: second, seconds: 0.16, amplitude: 0.22)
        }
    }

    // MARK: - Haptics

    private func startHaptics() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            let message = "This device has no haptic engine; guidance is sound and "
                    + "on-screen only."
            CaptureLog.guidance.notice("\(message, privacy: .public)")
            hapticsAvailable = false
            return
        }
        do {
            let engine = try CHHapticEngine()
            engine.playsHapticsOnly = true
            engine.isAutoShutdownEnabled = true
            engine.stoppedHandler = { reason in
                CaptureLog.guidance.notice(
                    "Haptic engine stopped: \(reason.rawValue, privacy: .public)"
                )
            }
            engine.resetHandler = { [weak engine] in
                try? engine?.start()
            }
            try engine.start()
            hapticEngine = engine
            hapticsAvailable = true
        } catch {
            let message = "Haptic engine would not start: "
                    + "\(error.localizedDescription)"
            CaptureLog.guidance.error("\(message, privacy: .public)")
            hapticsAvailable = false
        }
    }

    /// One dry tap. Sharpness high so it reads as a click rather than a thud -
    /// a thud feels like an error, and moving slightly too fast is not one.
    private func playTap(intensity: Float) {
        guard isHapticsEnabled, hapticsAvailable, let hapticEngine else { return }
        let event = CHHapticEvent(
            eventType: .hapticTransient,
            parameters: [
                CHHapticEventParameter(
                    parameterID: .hapticIntensity,
                    value: Swift.max(0.2, Swift.min(1, intensity))
                ),
                CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.85),
            ],
            relativeTime: 0
        )
        do {
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try hapticEngine.makePlayer(with: pattern)
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            CaptureLog.guidance.debug(
                "Haptic tap failed: \(error.localizedDescription, privacy: .public)"
            )
        }
    }
}
