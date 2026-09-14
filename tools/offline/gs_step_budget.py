"""gpuStep budget for build 250, from MEASURED populations, line-granular bytes.

Everything below is DRAM-side bytes per iteration, counted at 64-byte-line
granularity (a 4-byte field touched in every 32- or 64-byte record costs the
whole line), which is what the old _gs_traffic.py did not do.

Populations (all measured offline on the build-250 model, see
gs_step_populations.py, refute_bwd_simd.py, chk_simdreduce.py):
  N  299,888 splats                       (census)
  D  32,505 drawn per frame, mean          (868 poses; 10.8% of N)
  I  252,095 tile instances per frame, mean (offline max 528,066 = 1.059 x census peak)
  P  contributing (pixel, splat) pairs     (backward FULL BODY, see argv / defaults)
  f  contributing lanes per hot (SIMD group, splat) slot

Calibration: buffer A (gpuScan) is pure streaming work with no rasteriser in
it, measured 6.6746 s / 4000 = 1.669 ms. Its line-granular bytes / 1.669 ms is
the effective bandwidth the streaming kernels of buffer B are priced at.

Usage: python gs_step_budget.py [P_millions] [f]
"""
import sys

N, D, I, PX = 299_888, 32_505, 252_095, 720 * 540
S = 49_152                     # depth samples per frame (census: 196,608,000 / 4000)
SHF = 12                       # shDegree 1 -> 4 coeffs x 3 floats
MB = 1e6
P = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 18.0e6
f = float(sys.argv[2]) if len(sys.argv) > 2 else 17.0

GPU_SCAN_MS = 6.674550458381418 / 4000 * 1000
GPU_STEP_MS = 55.03132116666893 / 4000 * 1000

# ---------------------------------------------------------------- buffer A
A = [
    ('blit clearPerIteration: splatGrad N*48',        N * 48),
    ('blit clearPerIteration: shGrad N*12*4',         N * SHF * 4),
    ('blit clearPerIteration: 3 per-pixel planes',    3 * PX * 4),
    ('reset_visibility (4B store in every 32B stats)', N * 32),
    ('background (bgColor write)',                    PX * 12 + 72 * 1024),
    ('preprocess: splats read N*48',                  N * 48),
    ('preprocess: stats.filter3D read N*32',          N * 32),
    ('preprocess: tilesTouched write N*4',            N * 4),
    ('preprocess: draws[gid].opacity=0 dirties N*64', N * 64),
    ('preprocess: drawn rows (sh, raster, stats)',    D * (SHF * 4 + 40 + 32)),
    ('exclusive scan over N',                         N * 4 * 4),
]
bytesA = sum(b for _, b in A)
bw = bytesA / (GPU_SCAN_MS * 1e-3)          # bytes / s

# ---------------------------------------------------------------- buffer B
B = [
    ('duplicate_keys',            N * 4 + D * 40 + D * 64 + I * 8),
    ('radix sort, 8 passes',      8 * (I * 4 + I * 8 + I * 8)),
    ('tile_ranges (+fill)',       I * 8 + 1530 * 8),
    ('rasterize_forward (DRAM)',  I * 4 + D * 40 + PX * 28),
    ('loss_photometric',          PX * 72),
    ('ssim_prepare',              PX * 20),
    ('blur h+v, 5 planes',        PX * 80),
    ('ssim_stats (5 in, 3 out)',  PX * 32),
    ('blur h+v, 3 planes',        PX * 48),
    ('ssim_backward',             PX * 44),
    ('loss_finalize',             PX * 52),
    ('loss_depth (scattered)',    S * 32 + 4 * (S // 5) * 64),
    ('rasterize_backward (DRAM)', I * 4 + D * 40 + PX * 44 + D * 64 * 2),
    ('preprocess_backward',       N * 4 + D * (128 + 48 + 64 + 48 + 32 * 2 + 48 * 2 + 48 * 2)),
    ('regularizer (ALL N)',       N * (48 + 32 + 48 * 2)),
    ('adam_splat (N flag + D)',   N * 32 + D * (48 * 2 + 48 + 48 * 4 + 32)),
    ('adam_sh (N flag + D)',      N * 32 + D * (SHF * 4 * 6)),
]
print('populations: N %d  D %d  I %d  px %d  P %.1fM  f %.1f' % (N, D, I, PX, P / 1e6, f))
print()
print('BUFFER A (gpuScan, measured %.3f ms/iter)' % GPU_SCAN_MS)
for n, b in A:
    print('  %-50s %7.2f MB  %5.3f ms' % (n, b / MB, b / bw * 1e3))
print('  %-50s %7.2f MB  -> effective %.1f GB/s' % ('TOTAL', bytesA / MB, bw / 1e9))
print()
print('BUFFER B (gpuStep, measured %.3f ms/iter), streaming kernels priced at %.1f GB/s'
      % (GPU_STEP_MS, bw / 1e9))
stream_ms = 0.0
for n, b in B:
    ms = b / bw * 1e3
    tag = ''
    if 'rasterize' in n:
        tag = '   (DRAM floor only; compute-bound, see residual)'
    else:
        stream_ms += ms
    print('  %-50s %7.2f MB  %5.3f ms%s' % (n, b / MB, ms, tag))
print('  streaming kernels total (excl. both rasterisers)       %5.3f ms' % stream_ms)
resid = GPU_STEP_MS - stream_ms
print('  RESIDUAL for forward + backward rasterisers + dispatch  %5.3f ms  (%.0f%% of gpuStep)'
      % (resid, 100 * resid / GPU_STEP_MS))
print('  split by the measured 3.65x backward/forward ratio (11.05/3.03 ms, last on-device split):')
print('    forward  %.2f ms   backward  %.2f ms' % (resid / 4.65, resid * 3.65 / 4.65))
print()
hot = P / f
print('BACKWARD ATOMICS per iteration')
print('  contributing pairs              %6.1f M' % (P / 1e6))
print('  lane-atomics today (x11.x)      %6.1f M   (9 grads + absGrad2D + visAccum, +unknownAccum on UNKNOWN px)' % (P * 11 / 1e6))
print('  hot (SIMD group, splat) slots   %6.2f M   = SIMD atomic INSTRUCTIONS today / 11' % (hot / 1e6))
print('  after simd reduction            %6.2f M atomics (%.1f%% survive)' % (hot * 11 / 1e6, 100 / f))
print('  reduction cost, 11 x simd_sum   %6.0f M SIMD issues (~10 per simd_sum)' % (hot * 11 * 10 / 1e6))
print('  reduction cost, reduce-scatter  %6.0f M SIMD issues (~16 shuffles + 16 adds + 4 selects)' % (hot * 36 / 1e6))
print()
print('RUN SECONDS per 1 ms/iter saved: 4.0 s (4,000 iterations)')
