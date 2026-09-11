# Refutation ledger: every refuted finding, re-read by hand

Owner's rule (2026-09-11): nothing is refuted or dropped without my own check against the code or the data. Agent verdicts are hypotheses. Verdict here = what I concluded after reading the finding, the refutation, and the code or measurements it rests on.

Legend: **REOPEN** = the refutation is wrong or only corrected magnitudes; work it. **OPEN-LOW** = real, small or unpriced. **CLOSED(evidence)** = refutation holds and a measurement shows it. **CLOSED(shipped)** = the idea shipped in another form. **CLOSED(rule)** = conflicts with the owner's seed floor.

Sources: `scratchpad/refuted_all.json` (A00-A72), `refuted_missed.json` (M00-M49), `refuted_extra.json` (X00-X99).

## A00-A19 (read 2026-09-11)

| id | finding | verdict | what I checked |
|---|---|---|---|
| A00 | 13-bit depth key | CLOSED(shipped) | 252 shipped a 12-bit LOG key, 6 passes; offline 48.81 dB vs exact order |
| A01 | per-keyframe sort reuse / coherent tour | OPEN-LOW | literal proposal: 120 x ~500k instances x 8 B does not fit; the coherent-tour variant is unpriced |
| A02 | gpuScan is a drain, indirect dispatch | CLOSED(evidence) | 254's encodeStep = 0.1 s/run; submit latency ~0.25 ms x 4000 = ~1 s is the whole prize |
| A03 | shard the seeding sample loop across cores | **REOPEN** | the refutation's "rearrangement fails 5/5" was about GPU changes; CPU parallelism paid 3 times since (carving 3.09->1.05, trust 4.37->3.38, timeOffset 1.08->0.38). Seeding 1.68 s. Needs an order-exact merge: PLY order drives the trainer's index-stride thinning |
| A04 | half the population never gets a gradient | CLOSED(shipped) | exact alpha cull in trainer_preprocess already skips sub-minAlpha splats |
| A05 | densifier unmeasured, add a timer | CLOSED(evidence) | timings.densify exists; 1.4 s on 264 |
| A06 | stage colours as half in threadgroup memory | OPEN-LOW | my earlier 'closed' leaned on the tgIndex result, which is confounded (see A46) |
| A07 | ranking of where time sits | informational | superseded by per-stage clocks (252-258) |
| A08 | seeder targets 5.5x the kept seeds | CLOSED(rule) | also changes seed SIZE via spacing = sqrt(area/target); owner's floor is 800k seeds |
| A09 | census.py omits earlyStopEval | CLOSED(shipped) | census.py prints earlyStopEval and an explicitly lower-bound "untimed" line |
| A10 | splat plateau makes 2000-iter estimate high | informational | |
| A11 | pose graph leaves ~3x the median splat | **REOPEN** | refutation corrected magnitudes only: pose share of revisit disagreement up to 2.09 cm = 8.5 render px against a 3.7 px median splat. maxIterations 300 already shipped and changed cost 0.08% ("stalled, not starved"). Registration quality is an unworked quality ceiling |
| A12 | splat size grows with evidence | OPEN-LOW | on world size r falls +0.32 -> +0.22, still positive, consistent with view inconsistency (see A11) |
| A13 | keyframe blur, pick the sharpest in window | CLOSED(evidence) | 264 did exactly this: held-out 20.47 -> 19.73, SSIM 0.6668 -> 0.6414, reverted in 266. Sharper needs MORE keyframes, not a thinner spread |
| A14 | 720x540 and SH 1 hardcoded; train/eval mismatch | OPEN-LOW | SH degree 1 is fully active from ~iteration 467 at 4000 iters, so the SH half is moot; the eval renders at iteration 1/1 (no frequency blur) while training blurs, a real but small mismatch in early-stop scores |
| A15 | delete the least-contributing half | CLOSED(evidence) | random-50% control costs 0.499 dB vs 0.332: the readout, not redundancy |
| A16 | growth exhausts the cap at 600 | CLOSED(shipped) | relocation revived in 252 (25,935 moves on 256) |
| A17 | lower the cap | CLOSED(evidence) | as A15; plus it halves seeding |
| A18 | ceiling is eval-view registration | CLOSED(evidence) | local photos are frames 0-15, zero overlap with held-out; full shift correction +0.8 dB |
| A19 | sharper model penalised by misalignment | CLOSED(evidence) | best blur on the real model worth +0.23-0.30 dB, a tenth of the claim |

## A20-A41 (read 2026-09-11)

| id | finding | verdict | what I checked |
|---|---|---|---|
| A20 | held-out exposure clamp binds asymmetrically | CLOSED(evidence) | measured on frames 0-15, zero overlap with held-out; early stopping has never fired, so the decision it would move does not exist |
| A21 | schedules indexed to 30,000 | CLOSED(shipped) | re-indexed to 4,000 in 244 |
| A22 | SSIM carries 74% of gradient mass | CLOSED(evidence) | L1 mass is constant by construction (w*sign/3); SSIM share tracks render error, no stable lambda target |
| A23 | depth Huber outweighs photometric 6-20x | OPEN-LOW | "81x past the knee" used the model's own depth spread, not a residual; the real residual needs device depth files. See A68 (5.5-8x at a supervised pixel) |
| A24 | Adam sign coherence 0.50 | OPEN-LOW | refuter did not run it; measures opacity only, extrapolated to scale |
| A25 | disc prior sets the shape of a third of the model | **REOPEN (experiment)** | fair null leaves a 3.44x excess; forensics measured the prior dominating 98.7% of discs. The proposed null test, discPriorWeight 0, has NEVER been run |
| A26 | exposure model cannot absorb the variation | **REOPEN** | the refutation's "exposure constant" used frames 0-15 only; across the capture exposureDurationSeconds spans 0.0100-0.0167 s = 0.74 stops (A39's own data). And A38 measured the per-frame correction reaching gain 1.001-1.002: effectively inert |
| A27 | misregistration 1.8 px, not the ceiling | CLOSED (re-measured, clamped) | build 266, frames 0-15 (submap 0), lossq_align_cv.py, reproduced by the checker's own run: sub-pixel shift median 2.05 px (0.227 deg); whole-image gain +0.437 dB in-sample, +0.25 cross-validated left/right, +0.20 top/bottom; not one rigid shift (quadrants disagree by a median 4.7 px). Registration inside the anchored submap is worth about +0.25 dB on untrained views. No photos near an owner boundary, so the tear (below) is not in this number. Re-run as a CONTROL on every registration build |
| A27t | the frame 0-15 misregistration is a camera-to-IMU timing error | REFUTED (measured) | reg_timing.py, 13 untrained frames: best dt +12.2 ms, R^2 0.170, leave-one-out 0.078, at the p95 of shuffled flows; removing it RAISES the median shift 2.37 to 3.38 px. The calibrated 1.3 ms offset stands. 13 frames, one scan |
| TEAR | one rigid correction per submap tears the camera path at owner boundaries | OPEN (device A/B) | reg_boundaries.py, cross-checked from prepass_result submap corrections: all 18 boundaries, median 1.18 deg / 3.4 cm (13 px), worst 2.08 deg / 23.9 cm (67 px) between frames a third of a second apart, against 0.06 deg / 0.16 cm inside a submap. Fix: interpolate the correction in time between bracketing submaps. Needs its own build |
| VEIL | splats within 0.5 m of the camera | CLOSED(evidence) | an ARTEFACT of the offline tools: none of their three Jacobian copies had the device's tangent clamp (250+). Unclamped: +2.2 dB mean from deleting them. Clamped, as the phone renders: +0.000 dB on all 16 frames. The phone has no veil |
| A28 | radix saving small | CLOSED(shipped) | 6 passes shipped in 252 |
| A29 | byte-traffic comment undercounts 39% | CLOSED(evidence) | peak vs mean; mean 786,967 within 3% of the comment |
| A30 | 8-bit radix digit cannot fit | OPEN-LOW | a SIMD-rank design removes tgScratch; sort is now a small share of gpuStep |
| A31 | SH degree 2+3 bands carry signal | OPEN (experiment) | the refuter's per-coefficient shuffle null is itself biased (recovery #10); degree 2 costs time |
| A32 | SH schedule throttles view dependence | CLOSED(evidence) | the shader gates per band: degree 1 fully on from 11.7% of the run |
| A33 | seeds overshoot the 2D target 2.96x | CLOSED(evidence) | box-counting D = 2.31, never 2.0 |
| A34 | smear band tracks the pose residual | CLOSED(evidence) | the run used MEASURED sigma 9.3 mm; B = +/-2.35 sigma |
| A35 | 2x too many seeds | CLOSED(rule) | also build 182's measured regression |
| A36 | split the seeding time | OPEN-LOW (cheap) | stage clocks now split trust/edges/glass off; the remaining 1.68 s is not split between sample loop, weight sort and the 60 MB PLY write. Sub-timers decide A03 |
| A37 | collapse the seed pitch | CLOSED(rule) | |
| A38 | exposure correction is a fast-converging nuisance | **REOPEN (with A26)** | the refuter's OWN simulation shows it is not converging but inert: 1.9% of its clamp width after the 28 updates a frame gets |
| A39 | QC exposureJumpEV watches the wrong signal | CLOSED(evidence) | free-running AE: duration compensates brightness; proposal worse |
| A40 | protocol explains the 172/182 gaps | CLOSED(evidence) | reachable exposure worth 0.00 dB |
| A41 | 12 finiteness guards | CLOSED(shipped) | two roots per pair since 254 |

## A42-A61 (read 2026-09-11)

| id | finding | verdict | what I checked |
|---|---|---|---|
| A42 | instruction ledger, one atomic = 3.7 slots | CLOSED(evidence) | the 11.05/3.03 ms time base is typed into bwdcheck.py, not measured; k is a residual |
| A43 | SIMD-group reduction of the 12 atomics | CLOSED(evidence) | build 92 shipped it: 6% SLOWER |
| A44 | length(dLdMean2D) to an L1 norm | OPEN-LOW | changes the AbsGS statistic (densification ranking), unpriced |
| A45 | accumulate p, q and rebuild dL/dmean2D once per splat | **REOPEN (cheap)** | refuter confirmed the algebra and that d.conic and r.conic are bit-identical; 4 instructions per contributing pair (~69 M pairs/iter). 254 showed the backward pays for fewer instructions (gpuStep 55.0 to 49.0 across that batch) |
| A46 | tgIndex costs a third of resident threadgroups | OPEN (unmeasured) | NOT settled. 260 (tgIndex removed) gpuStep 52.2 s; 266 (restored) 52.4 s; 256 49.0 s. Run-to-run gpuStep noise is about +/-3 s and the keyframe set moves it 6 s (264: 46.4). One run each cannot resolve it |
| A47 | fold the stats accumulators into the grad record | CLOSED(shipped) | shipped with the preprocess_backward fold |
| A48 | writer/reader inventory | informational | |
| A49 | compulsory stats traffic 2.64 MB | CLOSED(evidence) | preprocess already dirties a superset of those lines |
| A50 | false sharing on stats lines | CLOSED(evidence) | the whole threadgroup is on one j at a time; co-residency is not contention |
| A51 | self-clear splatGrad2D | CLOSED(shipped) | 250 |
| A52 | relocation structurally dead | CLOSED(shipped) | 252; 20-26k relocations a run since |
| A53 | shrink supply 8.6x short | OPEN | direction confirmed; p50 is 7.43 mm on 264 against 3.39: the size gap is still THE quality gap |
| A54 | splitShareOfGrowth 1.0 + splitShrink 2.0 | **REOPEN (experiment)** | refuter showed the sim's answer swings -42%/+47% on an unmeasured parameter: that is uncertainty, not refutation. Direction (wider size distribution) is what the model lacks. Needs a device A/B |
| A55 | what the simulator did not measure | informational | |
| A56 | zero headroom from the seeder | CLOSED(shipped) | fillFraction 0.5 |
| A57 | 18.5% of the cap on faint splats | CLOSED(shipped) | relocation now consumes faint splats as donors; prunedLowOpacity 2,299 fell to 69-118 |
| A58 | no tangent clamp | CLOSED(shipped) | 250 (and its backward in 252) |
| A59 | SH band 1 is 4.8x weaker than Scaniverse's | **REOPEN (experiment)** | refuter CONFIRMED the 4.82x rms ratio and only withdrew the cone numbers. Rest coefficients learn at DC/20 (reference 3DGS setting, tuned for 30k iterations); at 4,000 they are barely trained |
| A60 | relocation stops at 500 | CLOSED(shipped) | |
| A61 | Scaniverse ships decompressed SPZ | OPEN-LOW (export) | 427k degree-3 splats in 106 MB expanded; an SPZ export is a feature, not speed |

## A62-A72 (read 2026-09-11)

| id | finding | verdict | what I checked |
|---|---|---|---|
| A62 | the pre-pass computes image detail and throws it away | OPEN (idea) | 5 free bits in the edge byte confirmed; the indexing claim was wrong (different loop), fixable. Size following image detail is what the size gap needs; see A64 |
| A63 | native gradient predicts unavoidable error across views | OPEN-LOW | local frames span 9.4 cm of baseline, so "across views" is untested |
| A64 | photographs demand an 18x size range | OPEN (with A62) | 18x is partly censoring; but our spread is 4.5-5.0x on 256-264 against Scaniverse's 8.9x, and seed size still follows LiDAR spacing, not the image |
| A65 | 720x540 floors splat size at 3.3 mm | CLOSED(evidence) | criterion-dependent (28% at the filter's own sigma); our p50 is 7.4 mm, far above any floor |
| A66 | splats 1.6x too large for their spacing | OPEN-LOW | the seed ratio is 1.04, not 0.5 (arbitrary sample per cell); the growth between seed and export is real and unlocated |
| A67 | error model checked on a real render | CLOSED(evidence) | the 9% agreement came from a hand-picked subset; 57% apart over all 16 frames |
| A68 | depth 5.5-8x photometric at a supervised pixel | OPEN-LOW | the script prints 8.70/5.95/7.30; authority is unmeasured |
| A69 | disc target should be 2.6 | OPEN (experiment, with A25) | post-hoc isotropy costs up to 1.8 dB, but a perturbation of a trained model is not a retrain; the retrain is unmeasured. Test the prior OFF first |
| A70 | huberDeltaMeters unreachable | CLOSED(evidence) | the fallback is live when `trust` itself is nil |
| A71 | depth is a fog 130x the Huber knee | CLOSED(evidence) + VEIL | the "fog" is 216 off-axis splats with camera z < 0.5 m and 80-420 px footprints composited first: the unclamped EWA problem that 250's tangent clamp fixed. The veil re-test must use a CLAMPED projection |
| A72 | F4 transition term divides by span | CLOSED(evidence) | the /span is the chain rule; the 1e-4 floor is unreachable |

## M00-M24 (refuted_missed.json, read 2026-09-11)

| id | finding | verdict | what I checked |
|---|---|---|---|
| M00 | supervision's per-sample trust lookups: 5 locks + 12 refcounts per sample | **REOPEN (speed)** | the same pattern paid in the trust build and background warm-up. It was off the critical path while the GPU was slow; on 264 the loop WAITED 4.2 s for supervision (1.5 s on 250) because gpuBusy fell to 50.9 s against a 42.5 s worker |
| M01 | depth-sample prefix copy | CLOSED(evidence) | sampleCount == depthSamples.count is an invariant; ArraySlice shares the buffer |
| M02 | image cache 0% hit; luma built and never read by supervision | **REOPEN (with M00)** | 0% hit is inherent to round-robin; the dead luma pass per decode is real work on the same worker that now stalls the loop |
| M03 | SH gradient waste per coefficient | CLOSED(evidence) | band-level gate |
| M04 | 24-bit key | CLOSED(shipped) | 252 |
| M05 | filter3D stale after a resolution drop | CLOSED-LOW | budgetReductions [] on this device |
| M06 | opacity-scaled tile footprint | CLOSED(evidence) | the exact alpha cull already covers it |
| M07 | per-pixel background loop | CLOSED(shipped) | background on the GPU since 252 |
| M08 | finiteness guards | CLOSED(shipped) | 254 |
| M09 | skip Adam for zero-gradient splats | CLOSED(shipped) | sparse gate; CPU compaction already runs |
| M10 | SH / Adam state in half | CLOSED(evidence) | fp16 ULP at |sh0|~2 is 1.3 steps: colour freeze |
| M11 | ASTC ground truth; cache depth samples | CLOSED(evidence) | compression biases the loss target; the depth-sample cache costs 189 MB |
| M12 | incremental sort reuse | CLOSED(evidence) | shuffled keyframes, tile in the high bits |
| M13 | hot/cold draw split; 8x8 tiles | CLOSED(evidence) | sector-granular fetch; 8x8 triples instances |
| M14 | opacity resets | CLOSED(evidence) | carving is the measured version; resets erase the trust-driven seed opacities |
| M15 | merge buffers A and B | CLOSED(evidence) | encodeStep 0.1 s: ~1 s total prize |
| M16 | writeArray element loop | CLOSED(shipped) | NOTE THE REFUTATION WAS WRONG: it said 0.5%; the memcpy fix measured 126 s to 101 s of training |
| M17 | split and narrow TrainerSplatDraw | CLOSED(evidence) | field accounting and a mean2D fixed-point overflow |
| M18 | 45% instance cut from alpha; 12/12 key | CLOSED(evidence / shipped) | units error (logit read as alpha); the key shipped as LOG depth, which answers its clamp objection |
| M19 | fold the clears into kernels | CLOSED(shipped) | NOTE THE REFUTATION WAS WRONG: "identical byte traffic" - the self-clears took gpuScan 6.7 s to 3.8 s |
| M20 | test-then-exp | CLOSED(shipped) | pad0 cutoff |
| M21 | tile-list depth, batches == 1 | CLOSED(evidence) | the second sweep measured batches mean 1.46 |
| M22 | denominators from a 238 ms iteration | CLOSED(obsolete) | |
| M23 | 2DGS / surfels | CLOSED(evidence) | lower NVS quality, more primitives for texture |
| M24 | ShorterSplatting scale reset | CLOSED(evidence) | the 3D filter turns a global shrink into mass deletion |
| M25 | indirect dispatch over a compacted active list | CLOSED(evidence) | SIMD-group masking: a compacted gather touches the same lines |
| M26 | EDGS / PocketGS / Mini-Splatting2 | CLOSED(evidence) | dense LiDAR init already; PocketGS confirms the architecture |
| M27 | imageblocks and tile shading | CLOSED(evidence) | SSIM's 11-tap window crosses tiles; traffic is small |
| M28 | coherence, Morton order, sort-free | CLOSED(evidence) | second sweep: Morton worth <0.2 s |
| M29 | transmittance cutoff 1e-3 | CLOSED(rule) | biases the supervised T_final: a quality trade |
| M30 | stochastic survivor sampling | CLOSED(evidence) | targets a dead trim path; misattributed MCMC formula |
| M31 | batched SH rest update | CLOSED(evidence) | approximation, below the noise floor |
| M32 | half tgConicOpacity | CLOSED(evidence) | measured 2 dB loss (build 164); A19 has no 8 KB cliff |
| M33 | two sorts / per-tile bucket sort | CLOSED(evidence) | TileGS nets 1.07x on a 4090; breaks the stable scatter |
| M34 | SAD-GS multi-split | CLOSED(evidence) | buys iterations with 54% more population |
| M35 | desk-scan governor and memory | OPEN-LOW | about a different scan whose census is gone; the room shows budgetReductions [] |
| M36 | did pre-pass ICP confirm anything | **REOPEN (with A11)** | answered by today's census: 132 of 600 converged, poses moved 7.5 cm median, so loop closure WORKS; but the 600 cap discards 97.6% of 24,916 candidates. More confirmed revisits is the direct lever on the registration ceiling. Revisits is 0.89 s and parallelisable like carving |
| M37 | five seeding census fields | CLOSED(answered) | sigma 12.76 mm measured, trust cut 0.230 above the 0.20 floor, thinning 5.55x |
| M38 | relocation donor fraction per pass | CLOSED(answered) | 256-264 census: relocated 20-26k, donorsAvailable ~ relocated every pass |
| M39 | does loop closure work | CLOSED(answered) | 132 ICP pairs converged; its 36% prediction used 81.8 deg FoV where the device reports 69.6 |
| M40 | keyframe prefix bias | CLOSED(shipped) | census records the keyframe span since 266 |
| M41 | memory gate | CLOSED(answered) | budgetReductions [] on the room scan |
| M42 | effective splat cap | CLOSED(answered) | 300,000 |
| M43 | ICP, coverage labels, stride-4 Laplacian | OPEN-LOW (capture) | ICP answered; the sharpness-meter ratio is 18.8-45.6x within a scene with a 45 s half-life (second sweep), a capture-side issue |
| M44 | background transmittance | CLOSED(answered) | mean T 0.0011 in trained views: viewer-only |
| M45 | does the 600 ICP cap bind | **REOPEN (with A11)** | binds 41.5x: 24,916 geometric candidates, 600 tried, 132 confirmed |
| M46 | seeding census fields | CLOSED(answered) | |
| M47 | rotation clause, governor rungs | OPEN (keyframes) | rotation admitted 104 of 120 keyframes (second sweep): that, not translation, is what exhausts the budget early. The fix for sharper AND wider coverage is more keyframes, not a different pick |
| M48 | ground-truth staging per build | OPEN-LOW (with M00) | writeArray is memcpy now; the per-build 4.67 MB allocate + deinterleave stays on the worker the loop now waits for |
| M49 | per-pixel ray in backgroundImage | CLOSED(shipped) | background on the GPU since 252; backgroundImage deleted |

## X00-X99 (refuted_extra.json, read 2026-09-11)

These are not research refutations. 29 are "skipped" notes from applying earlier patches and 71 are patch-review "problems" (comment accuracy, CI gates, the build-92 SIMD patch). Read in full for the ones with substance:

| id | note | verdict | what I checked |
|---|---|---|---|
| X13-X14 | capture keyframe rate 0.30 s | CLOSED(history) | now 0.45 s (journal, builds 216-224) |
| X22 | supervision build() autorelease pool | CLOSED(evidence) | about pooling, not the per-sample locks (M00, fixed separately) |
| X29 | PLY reader guard divides by a runtime value | CLOSED(shipped) | PLYCodec.swift:300 already reads `totalValues <= reader.remaining / 4`; the ascii branch is `(remaining + 1) / 2`. |
| X50-X57 | capture patch comments (carver's two appetites, uncited disk figures, short-scan supervision loss, bracket cadence) | OPEN-LOW (capture) | documentation debts on the capture path; X52's point stands: a scan under ~40 s at 3 frames/s gets fewer than 120 views |
| X79-X81, X83 | memory governor sizes a cut off the slice allocation; a cut can reallocate LARGER; thermal path unclamped | OPEN-LOW (dormant) | budgetReductions is [] on every room run, so none of this has fired; the thermal-path bidirectional resize was explicitly left as a separate ticket and is still open |
| X85-X99 | SIMD-sum backward patch | CLOSED(evidence) | shipped in 92, measured 6% slower, reverted |
| rest | skipped / comment / CI notes | CLOSED(history) | |


## CORRECTION, 2026-09-11

Every offline RENDER measurement from build 250 to today used an unclamped projection and understated the model by about 2 dB on the 16 photographed frames (clamped 16.1-19.7 dB against unclamped 14.0-18.1). All three copies (project.py, raster.py, detail.py; plus cacheline.py) now apply the clamp behind `project.TANGENT_CLAMP`. Findings built on offline renders since 250, including the forensics shape and isotropy numbers, should be re-run before they are trusted.


## CORRECTION 2, 2026-09-11

I concluded from 256 -> 260 that dropping tgIndex made gpuStep 6.5% slower and reverted it. 266 restored it and measured 52.4 s, the same as 260 without it. The single-run comparison was inside the noise. Speed conclusions of about 3 s or less from one device run each are not evidence.


## Keyframe selector (2026-09-11, build 274)

| id | finding | verdict | what I checked |
|---|---|---|---|
| KF-DUP | the sharpness look-ahead re-picks frames and leaks held-out frames into training | **CONFIRMED, fixed in 274** | I ran tools/offline/kf_exact.py myself: it reproduces held_out_frames.json and the 0..439 span exactly, and prints 29 duplicate slots of 120. I read selectKeyframes: after the window picks a frame ahead, the loop continues from the gate frame, and the next frames pass the turn gate measured against the pick, so they pick it again. splitHeldOut takes every tenth entry of the list, so a back-to-back duplicate lands one copy in each set: 6 of 12 held-out frames were trained. Every held-out score from 244 to 272 is flattered. Fix: look-ahead 0 (the pre-244 greedy), 120 distinct frames, predicted held-out [17, 61, 109, 155, 199, 231, 272, 314, 352, 392, 425, 460], span 0..485 |
| KF-MORE | more keyframes with today's code | OPEN (priced) | kf_price.py: above ~128 distinct frames the authority and edge caches (128 each) thrash; at 228+ keyframes the planner switches to 19-slice training. Raise both caps before going past 128 |
