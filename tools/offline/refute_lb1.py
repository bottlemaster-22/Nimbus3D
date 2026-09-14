"""INDEPENDENT re-derivation of the disc prior. Nothing reused from lossbal_*."""
import numpy as np, os, io, json
import project as P

# ---- the kernel's loss, transcribed straight from TrainerShaders.metal 2568-2586
def disc_loss(logScale, w, target):
    ls = np.clip(logScale, -12.0, 3.0)
    scale = np.exp(ls)
    lam = scale*scale
    ssum = np.maximum(lam.sum(-1), 1e-20)
    p = lam/ssum[..., None]
    logP = np.log(np.maximum(p, 1e-20))
    H = -(p*logP).sum(-1)
    rank = np.exp(H)
    residual = rank - target
    return w*0.5*residual*residual, rank, p, logP, H

def kernel_grad(logScale, w, target, factor):
    _, rank, p, logP, H = disc_loss(logScale, w, target)
    residual = rank - target
    k = factor*w*residual*rank
    return k[..., None]*p*(logP + H[..., None])

col, n = P.load_ply(os.path.join(P.D,'model','model.ply'))
logs = np.stack([col['scale_0'],col['scale_1'],col['scale_2']],1).astype(np.float64)
print('model splats', n)

rng = np.random.default_rng(0)
sub = rng.choice(n, 3000, replace=False)
L = logs[sub].copy()
W, T = 0.001, 2.0

for factor in (-4.0, -2.0):
    for h in (1e-6, 1e-5, 1e-4):
        num = np.zeros_like(L)
        for j in range(3):
            Lp = L.copy(); Lp[:, j] += h
            Lm = L.copy(); Lm[:, j] -= h
            num[:, j] = (disc_loss(Lp, W, T)[0] - disc_loss(Lm, W, T)[0])/(2*h)
        ana = kernel_grad(L, W, T, factor)
        keep = np.abs(num) > 1e-14
        r = ana[keep]/num[keep]
        rel = np.abs(ana[keep]-num[keep])/np.abs(num[keep])
        print('factor %+.1f  h=%.0e  used %d/%d  ratio p1 %.6f p50 %.6f p99 %.6f  medrel %.3e'
              % (factor, h, keep.sum(), L.size, *np.percentile(r,[1,50,99]), np.median(rel)))
