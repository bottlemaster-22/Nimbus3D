"""sizesim2 hard-codes SCENE_EXTENT_M = 7.6996, the longest edge of the CARVING
bounds in prepass/census.json. The trainer reads
`bundle.sceneBounds?.longestEdgeMeters ?? slice.bounds.longestEdgeMeters`
(MetalSplatTrainer.swift:909-911), and TrainerSupport.swift:641 records the
measured consequence: "splitScaleFraction 0.01 put the bar at 5.39 cm", which
pins sceneExtent for THIS scan at 5.39 m, not 7.70 m.

That matters because splitThresholdScale = sceneExtent * splitScaleFraction is
the FIRST of the three ORed split tests (TrainerDensifier.swift:543-545), i.e.
a second router that runs regardless of splitShareOfGrowth.
  7.6996 m -> bar 30.80 mm   (what the simulator used)
  5.3900 m -> bar 21.56 mm   (what the source says)
Build 182's exported model has p90 24.832 mm, so the second bar is inside the
population and the first is not.
"""
import json
import numpy as np
import sizesim2 as S

CAL = json.load(open('optcal.json'))
big = np.load('ours_largest_mm.npy')
for e in (7.6996, 5.39):
    bar = e * 0.004 * 1000
    print('extent %.4f m -> splitThresholdScale %6.2f mm ; share of the REAL '
          'build-182 model above it: %6.2f%%' % (e, bar, 100.0 * (big > bar).mean()))

def head(**kw):
    b = dict(seeds=150000, seed_multiplier=1.0, screen_split_fraction=0.0,
             opt_drift=CAL['opt_drift'], opt_diffuse=CAL['opt_diffuse'])
    b.update(kw); return S.Cfg().clone(**b)

print()
for extent in (7.6996, 5.39):
    S.SCENE_EXTENT_M = extent
    for name, kw in (('head (share 0.2)', dict()),
                     ('share 1.0', dict(split_share=1.0)),
                     ('share 1.0 + shrink 2.0', dict(split_share=1.0, split_shrink=2.0))):
        s, t, h = S.run(head(**kw))
        st = S.stats(s)
        print('extent %.4f  %-24s p10 %7.3f p50 %7.3f p90 %8.3f  spread %8.3fx  '
              'sd %.4f  split %7d clone %7d'
              % (extent, name, st['p10'], st['p50'], st['p90'], st['spread'],
                 st['sd'], t['split'], t['clone']))
    print()
