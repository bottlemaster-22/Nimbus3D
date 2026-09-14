"""Refutation harness for the 'prepassspeed' findings.

Everything here is arithmetic on real census fields or box counting on the
real trained model. Nothing is estimated.
"""
import json, math, os
import numpy as np
from project import load_ply, D

P = os.path.join(D, 'prepass')
cen = json.load(open(os.path.join(P, 'census.json')))
res = json.load(open(os.path.join(P, 'prepass_result.json')))
S = cen['seeding']

print('=== 1. the pitch formula, reproduced exactly ===')
area   = S['measuredSurfaceAreaSquareMeters']
target = S['targetSplatCount']
built  = S['gaussiansBuilt']
sp_cen = S['spacingMeters']
sp_calc = math.sqrt(area / target)
print(f'area                 {area}')
print(f'target               {target}')
print(f'spacing recomputed   {sp_calc:.9f}')
print(f'spacing in census    {sp_cen:.9f}   rel err {abs(sp_calc-sp_cen)/sp_cen:.2e}')
print(f'clamped?             {S["spacingWasClamped"]}  (min 0.010 max 0.120)')

print()
print('=== 2. where does area 65.65 come from? ===')
occ = res['occupancy']
print(f'carve  surfaceCellCount {occ["surfaceCellCount"]}  voxel {occ["voxelSizeMeters"]}')
print(f'  carve cells * voxel^2 = {occ["surfaceCellCount"]*occ["voxelSizeMeters"]**2:.3f} m^2'
      f'   <-- NOT 65.65')
# PrePassPipeline.swift:839 uses `survey`, PrePassSurveyor Settings.voxelSizeMeters = 0.10
for v in (0.10, 0.05, 0.03):
    print(f'  implied survey cell count at voxel {v:.2f} m : {area/v**2:.1f}')

print()
print('=== 3. finding 2 arithmetic: built vs area/h^2 ===')
flat = area / sp_cen**2
print(f'area/h^2   {flat:.2f}   (= target by construction)')
print(f'built      {built}')
print(f'built/target  {built/target:.5f}')

print()
print('=== 4. finding 3 arithmetic: solve seeds = (area/h^2)*max(1,B/h) ===')
B = built / flat * sp_cen
print(f'B = {B*1000:.3f} mm')
# physics prior, SmartDepthNoiseModel.default, SmartCore.swift 597-628
def sigma(z, cosi, s0=0.004, k=0.0015, spot=0.004, maxdeg=80.0):
    zz = max(z, 0.05)
    rt = s0 + k*zz*zz
    mc = math.cos(math.radians(maxdeg))
    c = max(abs(cosi), mc)
    tt = math.sqrt(max(0.0, 1-c*c))/c
    it = spot*zz*tt
    return math.sqrt(rt*rt + it*it)
z = res['qcCard']['medianCameraToSurfaceMeters']
print(f'median standoff       {z:.7f} m')
print(f'physics sigma  cos=1  {sigma(z,1.0)*1000:.3f} mm   (the finding used this)')
print(f'physics sigma  80 deg {sigma(z,math.cos(math.radians(80)))*1000:.3f} mm')
meas = S['medianSigmaMeters']
print(f'MEASURED median sigma {meas*1000:.3f} mm  '
      f'(sigmaIsMeasured={S["sigmaIsMeasured"]}, '
      f'{S["gaussiansWithMeasuredSigma"]}/{built} seeds carry it)')
print(f'B / physics_sigma  = {B/sigma(z,1.0):.3f}')
print(f'B / MEASURED sigma = {B/meas:.3f}    <-- i.e. B is +/- {B/meas/2:.2f} sigma')
pg = cen['poseGraph']
print(f'pose residual median {pg["finalResidualMedianCentimeters"]} cm, '
      f'exit {pg["exitReason"]}')
print(f'B / pose residual  = {B/(pg["finalResidualMedianCentimeters"]/100):.3f}')

print()
print('=== 5. finding 7 arithmetic: invert for h ===')
for N in (150_000, 300_000):
    h = (area*B/N)**(1/3)
    print(f'N={N}: h = {h*1000:.3f} mm  -> seed radius 0.5h = {h*500:.3f} mm '
          f'(today {sp_cen*500:.3f} mm, x{h/sp_cen:.3f})')

print()
print('=== 6. seed radius today, and what the proposals do to it ===')
fx_rgb = json.load(open(os.path.join(D,'capture_bundle.json')))['intrinsics']['fx']
print(f'rgb fx {fx_rgb}')
for name, h in (('today', sp_cen), ('finding 4 (target 150k)', math.sqrt(area/150_000)),
                ('finding 7 (h*=26.77mm)', (area*B/150_000)**(1/3))):
    print(f'{name:28s} spacing {h*1000:8.3f} mm  radius {h*500:7.3f} mm  '
          f'x{h/sp_cen:.3f} vs today')
