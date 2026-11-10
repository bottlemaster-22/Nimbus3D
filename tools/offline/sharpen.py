"""THE DECLINE MECHANISM.
Under a fixed registration error d, what model sharpness maximises PSNR?
Measured on the real frames: PSNR( blur(image, sigma), shift(image, d) ).
A model sharper than the pose error supports scores WORSE, and every extra
iteration sharpens the model."""
import json,io,os,numpy as np
from scipy.ndimage import gaussian_filter
import project as P, render_full as RF
D=P.D; INC=RF.INC
census=json.load(io.open(os.path.join(D,'model','train_census.json'),encoding='utf-8'))
sl=census['slices'][0]; rw,rh=sl['renderWidth'],sl['renderHeight']
bundle=json.load(io.open(os.path.join(D,'capture_bundle.json'),encoding='utf-8'))
have=set(os.listdir(os.path.join(INC,'images')))
names=[os.path.basename(f['imagePath']) for f in bundle['frames'] if os.path.basename(f['imagePath']) in have]
imgs=[RF.ground_truth(n,rw,rh) for n in names]
SIG=[0,0.5,1.0,1.5,2.0,3.0,4.0]
print('rows = registration error d (px), cols = blur applied to the PERFECT model (sigma px)')
print('     d |' + ''.join('%7.1f'%s for s in SIG) + '   | best sigma   gain over sharp')
for d in [0,2,3,4,5,6,8]:
    row=[]
    for s in SIG:
        ps=[]
        for im in imgs:
            a=gaussian_filter(im,(s,s,0)) if s>0 else im
            ps.append(RF.psnr(a[20:-20,20:-20],np.roll(im,d,axis=1)[20:-20,20:-20]))
        row.append(np.mean(ps))
    row=np.array(row); j=int(np.argmax(row))
    print('%6d |'%d + ''.join('%7.2f'%v for v in row) + '   |   %4.1f        %+.2f dB'%(SIG[j],row[j]-row[0]))
