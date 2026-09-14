import sv_load, numpy as np
SCR=r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
c,_=sv_load.sv(); sent=np.load(SCR+r"\sv_sentinel.npy"); F=~sent
o,_=sv_load.ours()

def alph(col,m=None):
    v=1/(1+np.exp(-np.clip(col['opacity'].astype(np.float64),-60,60)))
    return v if m is None else v[m]

print('=== OPACITY (alpha domain) ===')
for tag,a in [('SCANIVERSE fg',alph(c,F)),('SCANIVERSE dome',alph(c,sent)),('OURS',alph(o))]:
    h,_=np.histogram(a,bins=np.linspace(0,1,11))
    print(' %-16s ' % tag, ' '.join('%5.1f%%'%(100*x/len(a)) for x in h))
    print('   %-14s  p1 %.4f p5 %.4f p50 %.4f p95 %.4f p99 %.4f | frac<0.05 %.2f%% <0.1 %.2f%% >0.9 %.2f%% >0.99 %.3f%%'
          % ('', *np.percentile(a,[1,5,50,95,99]), 100*np.mean(a<0.05),100*np.mean(a<0.1),100*np.mean(a>0.9),100*np.mean(a>0.99)))
print(' bins:              0-.1  .1-.2 .2-.3 .3-.4 .4-.5 .5-.6 .6-.7 .7-.8 .8-.9 .9-1')
print()
print('=== SH BANDS ===')
# INRIA layout: f_rest is channel-major, 15 coeffs per channel for deg3
def sh_rms(col, ncoef, m=None):
    R=np.stack([col['f_rest_%d'%i] for i in range(ncoef*3)],1).astype(np.float64)
    if m is not None: R=R[m]
    R=R.reshape(len(R),3,ncoef)          # [n, channel, coeff]
    dc=np.stack([col['f_dc_0'],col['f_dc_1'],col['f_dc_2']],1).astype(np.float64)
    if m is not None: dc=dc[m]
    return R, dc
R,dc = sh_sv = sh_rms(c,15,F)
print(' SCANIVERSE  DC rms %.4f  (per-channel %s)' % (np.sqrt((dc**2).mean()), np.round(np.sqrt((dc**2).mean(0)),4)))
bands={1:range(0,3),2:range(3,8),3:range(8,15)}
for b,idx in bands.items():
    v=R[:,:,list(idx)]
    print('   band %d (%2d coef/ch): rms %.5f  mean|.| %.5f  frac exactly 0 %.2f%%  p99|.| %.4f  max|.| %.4f'
          % (b,len(list(idx)),np.sqrt((v**2).mean()),np.abs(v).mean(),100*np.mean(v==0),np.percentile(np.abs(v),99),np.abs(v).max()))
    # energy relative to DC
print('   ratio band-energy/DC-energy: b1 %.4f  b2 %.5f  b3 %.5f'
      % tuple(np.sqrt((R[:,:,list(bands[b])]**2).sum(axis=(1,2)).mean()/ (dc**2).sum(1).mean()) for b in [1,2,3]))
allz = (R==0).all(axis=(1,2))
b23z = (R[:,:,3:]==0).all(axis=(1,2))
print('   frac splats with ALL SH-rest zero: %.2f%% ; with bands 2+3 all zero: %.2f%%' % (100*allz.mean(),100*b23z.mean()))
print('   per-splat band3 rms: p50 %.5f p90 %.5f p99 %.5f' % tuple(np.percentile(np.sqrt((R[:,:,8:]**2).mean(axis=(1,2))),[50,90,99])))
print()
Ro,dco = sh_rms(o,3)
print(' OURS        DC rms %.4f  (per-channel %s)' % (np.sqrt((dco**2).mean()), np.round(np.sqrt((dco**2).mean(0)),4)))
print('   band 1 ( 3 coef/ch): rms %.5f  mean|.| %.5f  p99|.| %.4f  max|.| %.4f' % (np.sqrt((Ro**2).mean()),np.abs(Ro).mean(),np.percentile(np.abs(Ro),99),np.abs(Ro).max()))
print('   ratio band1-energy/DC-energy: %.4f' % np.sqrt((Ro**2).sum(1).sum(1).mean()/(dco**2).sum(1).mean()))
print()
print(' Scaniverse band1 vs ours band1 (same 3 coef/ch):  SV rms %.5f   OURS rms %.5f   ratio %.2fx'
      % (np.sqrt((R[:,:,0:3]**2).mean()), np.sqrt((Ro**2).mean()), np.sqrt((R[:,:,0:3]**2).mean())/np.sqrt((Ro**2).mean())))
