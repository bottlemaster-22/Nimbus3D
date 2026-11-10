"""Stage 1: the disc/effective-rank prior's dL/dlogScale on the REAL model,
and the Huber delta against the REAL noise model. No render needed.

Everything here is exact arithmetic on measured inputs:
  * scales from model.ply (299,209 splats, build 182)
  * the kernel's own formula from TrainerShaders.metal trainer_regularizer
  * SmartDepthNoiseModel defaults from SmartCore.swift
"""
import io, json, os, sys
import numpy as np
import project as P

DISC_W = 0.001          # SmartLossSettings.discPriorWeight  (was 0.01)
DISC_W_OLD = 0.01
TARGET = 2.0            # discTargetEffectiveRank
EDGE_TARGET = 1.0

col, count = P.load_ply(os.path.join(P.D, 'model', 'model.ply'))
logs = np.clip(np.stack([col['scale_0'], col['scale_1'], col['scale_2']], 1).astype(np.float64), -12, 3)
scale = np.exp(logs)
print('splats %d' % count)

s_sorted = np.sort(scale, axis=1)          # ascending: s0 <= s1 <= s2
smin, smid, smax = s_sorted[:,0], s_sorted[:,1], s_sorted[:,2]
print('\n--- SHAPE, measured on model.ply ---')
for name, v in (('largest axis mm', smax*1000), ('middle axis mm', smid*1000),
                ('smallest axis mm', smin*1000)):
    print('  %-18s p10 %8.3f  p50 %8.3f  p90 %8.3f' %
          (name, np.percentile(v,10), np.percentile(v,50), np.percentile(v,90)))
ar = smax/np.maximum(smin,1e-12)
print('  aspect max/min      p10 %8.3f  p50 %8.3f  p90 %8.3f' %
      (np.percentile(ar,10), np.percentile(ar,50), np.percentile(ar,90)))
ar2 = smid/np.maximum(smin,1e-12)
print('  aspect mid/min      p10 %8.3f  p50 %8.3f  p90 %8.3f' %
      (np.percentile(ar2,10), np.percentile(ar2,50), np.percentile(ar2,90)))
print('  p90/p10 of largest axis  %.3f' % (np.percentile(smax,90)/np.percentile(smax,10)))

# --- the kernel, verbatim -------------------------------------------------
lam = scale*scale
ssum = np.maximum(lam.sum(1, keepdims=True), 1e-20)
p = lam/ssum
logP = np.log(np.maximum(p, 1e-20))
H = -(p*logP).sum(1)
rank = np.exp(H)
print('\n--- EFFECTIVE RANK, measured ---')
for q in (1,10,25,50,75,90,99):
    print('  p%-3d %.4f' % (q, np.percentile(rank,q)))
print('  mean %.4f   frac rank > 2 : %.4f' % (rank.mean(), (rank>2).mean()))

def prior_grad(w, target):
    residual = rank - target
    k = -4.0*w*residual*rank
    return k[:,None]*p*(logP + H[:,None])

for w, target, label in ((DISC_W, TARGET, 'disc w=0.001 target=2'),
                         (DISC_W_OLD, TARGET, 'disc w=0.01  target=2'),
                         (DISC_W, EDGE_TARGET, 'disc w=0.001 target=1 (edge splats)')):
    g = prior_grad(w, target)
    mag = np.abs(g)
    nrm = np.linalg.norm(g, axis=1)
    print('\n--- PRIOR dL/dlogScale : %s ---' % label)
    print('  per-component |g|   p10 %.3e  p50 %.3e  p90 %.3e  max %.3e' %
          (np.percentile(mag,10), np.percentile(mag,50), np.percentile(mag,90), mag.max()))
    print('  3-vector norm       p10 %.3e  p50 %.3e  p90 %.3e' %
          (np.percentile(nrm,10), np.percentile(nrm,50), np.percentile(nrm,90)))
    # which axis does it push, and which way?
    order = np.argsort(scale, axis=1)
    gs = np.take_along_axis(g, order, axis=1)
    print('  on SMALLEST axis: median %+.3e   frac pushing it UP  %.3f'
          % (np.median(gs[:,0]), (gs[:,0] < 0).mean()))
    print('  on LARGEST  axis: median %+.3e   frac pushing it DOWN %.3f'
          % (np.median(gs[:,2]), (gs[:,2] > 0).mean()))

np.save(os.path.join(os.path.dirname(__file__), '_prior_grad.npy'), prior_grad(DISC_W, TARGET))
np.save(os.path.join(os.path.dirname(__file__), '_rank.npy'), rank)

# --- Huber delta vs the noise model ---------------------------------------
print('\n=== HUBER DELTA vs SmartDepthNoiseModel (SmartCore.swift defaults) ===')
sigma0, kr, spot, maxinc = 0.004, 0.0015, 0.004, 80.0
def sigma_model(z, cos_i):
    zz = np.maximum(z, 0.05)
    rng = sigma0 + kr*zz*zz
    mincos = np.cos(np.radians(maxinc))
    c = np.maximum(np.abs(cos_i), mincos)
    tan = np.sqrt(np.maximum(0,1-c*c))/c
    return np.sqrt(rng*rng + (spot*zz*tan)**2)
def fallback_delta(z):
    return np.maximum(0.02*np.maximum(z/2.0,1.0), 0.004)
print('  z(m)  incid   sigma_model(mm)  fallback delta(mm)  fallback/sigma')
for z in (0.5,1.0,1.275,2.0,3.0,4.5):
    for inc in (0.0,45.0,70.0):
        s = sigma_model(np.array([z]), np.array([np.cos(np.radians(inc))]))[0]
        d = fallback_delta(np.array([z]))[0]
        print('  %5.2f  %4.0fdeg  %10.2f       %10.2f        %8.2f'
              % (z, inc, s*1000, d*1000, d/s))
print('\n  trust weight 1/(1+(sigma/0.02)^2) at those sigmas:')
for z in (0.5,1.275,3.0,4.5):
    s = sigma_model(np.array([z]), np.array([1.0]))[0]
    print('    z=%.3f normal incidence: sigma %.2f mm  ->  trustWeight %.4f' % (z, s*1000, 1/(1+(s/0.02)**2)))
