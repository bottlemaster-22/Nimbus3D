"""How well do splats lie ON their surface? For each splat, the local surface
plane is fitted to its 24 nearest splat centres (PCA); the splat's own normal
is its flattest axis. Reports the angle between the two on planar patches."""
import sys, numpy as np
from scipy.spatial import cKDTree
import project as P
import render_view as RV
col, n = P.load_ply(sys.argv[1])
m = RV.prepare(col, n)
x = m['mean']; nrm = m['normal']; sc = np.sort(m['scale'], 1)
tree = cKDTree(x)
d, idx = tree.query(x, k=24)
nb = x[idx] - x[idx].mean(1, keepdims=True)
cov = np.einsum('nki,nkj->nij', nb, nb) / 24
w, v = np.linalg.eigh(cov)
plane_n = v[:, :, 0]
planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
cosang = np.abs(np.sum(plane_n * nrm, 1))
ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
flat = sc[:, 0] / np.maximum(sc[:, 1], 1e-12) < 0.3
sel = (planarity > 0.5) & flat
print('splats', len(x), ' on clearly planar patches & flat:', sel.sum())
print('angle between splat and its surface: median %.1f deg, p75 %.1f, share over 30 deg %.0f%%, over 60 deg %.0f%%' % (
    np.median(ang[sel]), np.percentile(ang[sel], 75), 100 * (ang[sel] > 30).mean(), 100 * (ang[sel] > 60).mean()))
print('random orientation would give median 60.0 deg')
ratio = sc[:, 2] / np.maximum(sc[:, 1], 1e-12)
print('needles (long/middle > 3): %.1f%%,  > 5: %.1f%%' % (100 * (ratio > 3).mean(), 100 * (ratio > 5).mean()))
print('opacity: median %.2f, share under 0.3: %.0f%%' % (np.median(m['opacity']), 100 * (m['opacity'] < 0.3).mean()))
nn = d[:, 1]
print('floaters (nearest neighbour > 3 cm): %d' % (nn > 0.03).sum())
