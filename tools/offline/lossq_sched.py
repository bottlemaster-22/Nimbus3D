"""Every schedule the objective depends on, evaluated at the iterations that
matter, transcribed line by line from the Swift.

Sources:
  TwoScaleTrustField.depthLossScale(iteration:of:floor:)      Smart/TwoScaleTrustField.swift:794
  MetalSplatTrainer.cameraUniforms  (SH gate, frequency blur) Trainer/MetalSplatTrainer.swift:2259-2272
  MetalSplatTrainer.adamUniforms    (all learning rates)      Trainer/MetalSplatTrainer.swift:2304-2350
  MetalSplatTrainer.regularizerUniforms (binarize ramp)       Trainer/MetalSplatTrainer.swift:2276-2296
  MetalSplatTrainer.evaluateHeldOut passes iteration:1 total:1  Trainer/MetalSplatTrainer.swift:2812-2820
"""
import math

TOTAL = 30000                 # budgetAsRun.iterations
RAN   = 4000                  # iterationsCompleted
SH_COEFFS = 4                 # shDegree 1 -> (1+1)^2 = 4

# TrainerTuning defaults, TrainerSupport.swift
frequencyBlurStartVariance = 4.0
frequencyBlurEndFraction   = 0.2
shFullyEnabledFraction     = 0.35
warmupFraction             = 0.15
densifyStartMaxIterations  = 300
densifyEndFraction         = 0.5
positionLRInitialScaled    = 0.00016
positionLRFinalScaled      = 0.0000016
lateLRFraction             = 0.1
opacityLR, scaleLR, rotationLR, shDCLR = 0.05, 0.005, 0.001, 0.0025
shDCLRMultiplier, shRestLRDivisor = 4.0, 20.0
binarizeLastFraction       = 0.0
depthScheduleFloor         = 0.05      # SmartCore.swift:175
filter2DVariance           = 0.25      # MetalSplatTrainer.swift:2255

# model.json bounds -> longest edge
BOUNDS_MIN = (-2.1402798, -1.7696334, -2.845373)
BOUNDS_MAX = ( 2.9781783,  1.4613773,  2.6855052)
sceneExtent = max(max(b - a for a, b in zip(BOUNDS_MIN, BOUNDS_MAX)), 0.5)


def exp_lerp(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return math.exp(math.log(a) * (1 - t) + math.log(b) * t)


def depth_scale(i, total, floor):
    i = max(0, min(i, total))
    strong_end = min(int(total * 0.35), 7000)
    if i <= strong_end:
        return 1.0
    span = max(total - strong_end, 1)
    t = min(max((i - strong_end) / span, 0.0), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(t * math.pi))


def freq_blur(i, total):
    f = i / max(total, 1)
    if f < frequencyBlurEndFraction and frequencyBlurEndFraction > 0:
        return frequencyBlurStartVariance * (1 - f / frequencyBlurEndFraction)
    return 0.0


def active_sh(i, total):
    f = i / max(total, 1)
    gate = min(max(f / max(shFullyEnabledFraction, 1e-3), 0.0), 1.0)
    return max(min(1 + int((SH_COEFFS - 1) * gate), SH_COEFFS), 1)


print('scene extent (model.json bounds longest edge) = %.4f m' % sceneExtent)
print('budget as run: %d iterations, %d completed\n' % (TOTAL, RAN))

hdr = ('iter', 'depthScale', 'freqBlurVar', 'lowPass px2', 'activeSH', 'lrMean',
       'lrScale', 'lrOpacity', 'lrSHDC', 'binarize', 'poseDeltas')
print('%6s %10s %11s %11s %8s %10s %9s %9s %9s %8s %10s' % hdr)
for i in (0, 500, 1000, 2000, 2500, 3000, 3500, 4000, 6000, 7000, 10500, 15000, 30000):
    t = i / TOTAL
    lrDecay = exp_lerp(1.0, max(lateLRFraction, 1e-4), t)
    lrMean = exp_lerp(positionLRInitialScaled * sceneExtent,
                      positionLRFinalScaled * sceneExtent, t)
    fb = freq_blur(i, TOTAL)
    mark = ' <-' if i in (2000, 4000) else ''
    print('%6d %10.4f %11.4f %11.4f %8d %10.3e %9.3e %9.3e %9.3e %8.3f %10s%s'
          % (i, depth_scale(i, TOTAL, depthScheduleFloor), fb,
             filter2DVariance + fb, active_sh(i, TOTAL),
             lrMean, scaleLR * lrDecay, opacityLR * lrDecay,
             shDCLR * shDCLRMultiplier * lrDecay,
             0.0, 'frozen' if t < warmupFraction else 'free', mark))

print('\nWHAT THE HELD-OUT EVALUATION USES (evaluateHeldOut passes iteration:1, totalIterations:1)')
print('  fraction              = 1.0')
print('  frequencyBlurVariance = %.4f   -> lowPass = %.4f px^2'
      % (freq_blur(1, 1), filter2DVariance + freq_blur(1, 1)))
print('  activeSHCoeffCount    = %d of %d' % (active_sh(1, 1), SH_COEFFS))
print('\nWHAT TRAINING USED AT THE SAME MOMENT')
for i in (2000, 4000):
    print('  iteration %5d: lowPass = %.4f px^2, activeSHCoeffCount = %d of %d'
          % (i, filter2DVariance + freq_blur(i, TOTAL), active_sh(i, TOTAL), SH_COEFFS))

print('\nSCHEDULE MILESTONES THE RUN NEVER REACHED (stopped at %d of %d)' % (RAN, TOTAL))
for name, it in (('frequency blur reaches zero', int(TOTAL * frequencyBlurEndFraction)),
                 ('all SH coefficients active', int(TOTAL * shFullyEnabledFraction)),
                 ('camera pose deltas unfreeze', int(TOTAL * warmupFraction)),
                 ('depth taper begins', min(int(TOTAL * 0.35), 7000)),
                 ('densification window closes', int(TOTAL * densifyEndFraction))):
    print('  %-32s iteration %6d   %s' % (name, it, 'REACHED' if it <= RAN else 'NOT REACHED'))

print('\nSH COEFFICIENT ACTIVATION ITERATIONS (of %d)' % SH_COEFFS)
prev = 0
for i in range(0, TOTAL + 1):
    a = active_sh(i, TOTAL)
    if a != prev:
        print('  coefficient index %d first evaluated at iteration %6d %s'
              % (a - 1, i, '' if i <= RAN else '  <- never, run ended at %d' % RAN))
        prev = a
