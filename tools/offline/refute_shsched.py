"""Reproduce the trainer's REAL SH activation schedule, then push it through
the SHADER's actual gate (which is per-BAND, not per-coefficient)."""
import math
ITERS=3000; FRAC=0.35
def sched(nCoeff):
    print(f"\n--- shCoefficientCount = {nCoeff} (degree {int(math.isqrt(nCoeff))-1}) ---")
    prev=None; band_on={}
    for it in range(ITERS+1):
        f=it/ITERS
        gate=min(max(f/FRAC,0.0),1.0)
        active=1+int((nCoeff-1)*gate)
        active=max(min(active,nCoeff),1)
        if active!=prev:
            print(f"  iter {it:5d} ({100*f:5.1f}%): activeSHCoeffCount -> {active}")
            prev=active
        # SHADER gates (TrainerShaders.metal trainer_evalSH / backward):
        #   deg1 block:  activeCount > 1  (enables ALL THREE deg1 coeffs at once)
        #   deg2 block:  activeCount > 4  (enables ALL FIVE deg2 coeffs at once)
        for nm,cond in [("deg1 band (all 3 coeffs)",active>1 and nCoeff>1),
                        ("deg2 band (all 5 coeffs)",active>4 and nCoeff>4)]:
            if cond and nm not in band_on:
                band_on[nm]=it
    print("  SHADER-VISIBLE band activation:")
    for nm,it in band_on.items():
        print(f"    {nm}: first active at iter {it} = {100*it/ITERS:.1f}% of run")
sched(4)   # shipped: degree 1
sched(9)   # degree 2, the already-implemented option

print("\n=== LR TRAJECTORY (lateLRFraction=0.1, shDCLRMultiplier=4 are the")
print("    CURRENT defaults in TrainerSupport.swift lines 440/456) ===")
shDCLR=0.0025; mult=4.0; div=20.0; late=0.1
def expLerp(a,b,t):
    t=min(max(t,0.0),1.0)
    return math.exp(math.log(a)*(1-t)+math.log(b)*t)
tot=0.0; act=0.0
for it in range(ITERS):
    t=it/ITERS
    dec=expLerp(1.0,late,t)
    lr=shDCLR*mult*dec/div
    tot+=lr
    if 1+int(3*min(max(t/FRAC,0.0),1.0))>1: act+=lr
print(f"  lrSHRest at iter 0    = {shDCLR*mult/div:.6f}")
print(f"  lrSHRest at iter 350  = {shDCLR*mult*expLerp(1,late,350/ITERS)/div:.6f}"
      f"  (deg1 band switches ON here; {100*expLerp(1,late,350/ITERS):.1f}% of peak)")
print(f"  lrSHRest at iter 1050 = {shDCLR*mult*expLerp(1,late,1050/ITERS)/div:.6f}"
      f"  ({100*expLerp(1,late,1050/ITERS):.1f}% of peak) <- finding's claimed switch-on")
print(f"  lrSHRest at iter 2999 = {shDCLR*mult*expLerp(1,late,2999/ITERS)/div:.6f}")
print(f"  integrated LR the deg1 band ACTUALLY receives / full-run integral"
      f" = {act/tot:.4f}")
