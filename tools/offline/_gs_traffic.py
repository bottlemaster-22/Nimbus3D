N=299888; D=56721; PX=720*540; TILES=45*34; I=334953; IPEAK=498745
S=49152; SHF=12  # shDegree 1 -> 4 coeffs -> 12 floats
MB=1e6
def mb(b): return b/MB
rows=[]
def R(name,dispatch,grid,b):
    rows.append((name,dispatch,grid,b))

# ---- buffer A (gpuScan, 1.669 ms/iter) ----
R('blit clearPerIteration (8 fills)','blit','33.45 MB', N*48 + N*SHF*4 + PX*4*3 + 36)
R('reset_visibility','1D',f'{N}', N*32*0+N*4*2)  # rmw one uint in 32B record
R('background','1D',f'{PX}', 72*1024 + PX*12)
R('preprocess','1D',f'{N}', N*48 + N*32 + N*4 + N*64*0 + N*4 + D*(48+64+40) + D*32)
R('exclusive_scan(N) x3','block','293+1 tg', N*4*2 + N*4)
# ---- buffer B (gpuStep, 13.758 ms/iter) ----
R('duplicate_keys','1D',f'{N}', N*4 + D*(40+4) + I*8)
R('radix_sort 8 x (hist+3 scan+scatter)','block','328 tg', 8*(I*4 + I*8 + I*8 + 5248*4*3))
R('tile_ranges (+fill)','1D',f'{I}', TILES*8 + I*4)
R('rasterize_forward','tile',f'{TILES} tg x256', I*4 + I*40 + PX*28)
R('loss_photometric','1D',f'{PX}', PX*12+PX*4+PX*12+PX*12 + PX*12+PX*12+PX*8)
R('ssim_prepare','1D',f'{PX}', PX*8 + PX*12)
R('blur_h + blur_v (5 planes)','2D','720x540', 4*PX*5*4)
R('ssim_stats','1D',f'{PX}', PX*5*4 + PX*5*4)
R('blur_h + blur_v (3 planes)','2D','720x540', 4*PX*3*4)
R('ssim_backward','1D',f'{PX}', PX*12 + PX*8 + PX*24)
R('loss_finalize','1D',f'{PX}', PX*12+PX*12+PX*4+PX*12 + PX*12)
R('loss_depth','1D',f'{S}', S*32 + S*8 + S*12)
R('rasterize_backward','tile',f'{TILES} tg x256', I*4 + I*40 + PX*(4+4+12+4+4+12+4))
R('preprocess_backward','1D',f'{N}', N*4 + D*(64+48+64+SHF*4) + D*(64+48*2+SHF*4*2))
R('regularizer','1D',f'{N}', N*48 + N*32 + N*48*2)
R('adam_splat','1D',f'{N}', N*32 + D*(48+48+48+48) + D*(48+48+48))
R('adam_sh','1D',f'{N}', N*32 + D*(SHF*4*4) + D*(SHF*4*3))
tot=sum(r[3] for r in rows if r[0] not in ())
totA=sum(r[3] for r in rows[:5]); totB=tot-totA
print('%-42s %-6s %-14s %9s %6s' % ('kernel','shape','grid','MB/iter','%stepB'))
for n,dsp,g,b in rows:
    print('%-42s %-6s %-14s %9.2f %6.1f%%' % (n,dsp,g,mb(b),100*b/totB))
print('%-42s %-6s %-14s %9.2f' % ('TOTAL buffer A','','',mb(totA)))
print('%-42s %-6s %-14s %9.2f' % ('TOTAL buffer B','','',mb(totB)))
print()
print('buffer A measured 1.669 ms/iter -> %.1f GB/s effective' % (totA/1.669e-3/1e9))
print('buffer B measured 13.758 ms/iter -> %.1f GB/s effective' % (totB/13.758e-3/1e9))
print('whole step   %.1f GB/s' % (tot/(1.669e-3+13.758e-3)/1e9))
