"""Shared loader for the Scaniverse PLY and ours. Caches to npz in scratchpad."""
import os, numpy as np

SCRATCH = r"C:\Users\Undea\AppData\Local\Temp\claude\C--Users-Undea-Documents-TOMBLINE\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad"
SV_PLY  = r"C:\Users\Undea\Downloads\Four Marks.ply"
OUR_PLY = r"C:\Users\Undea\Documents\LiKOVA\Scans\diagnostics\scan_20260906_164840\model\model.ply"

def read_ply(path):
    f = open(path, 'rb')
    header = b''
    while not header.endswith(b'end_header\n'):
        header += f.read(1)
    names, count = [], 0
    for line in header.decode('ascii', 'replace').split('\n'):
        p = line.split()
        if not p: continue
        if p[0] == 'element' and p[1] == 'vertex': count = int(p[2])
        elif p[0] == 'property' and p[1] == 'float': names.append(p[2])
    raw = f.read(count * len(names) * 4)
    f.close()
    data = np.frombuffer(raw, dtype='<f4').reshape(count, len(names))
    return {n: data[:, i] for i, n in enumerate(names)}, count, names

def cached(tag, path):
    npz = os.path.join(SCRATCH, tag + '.npz')
    if os.path.exists(npz):
        z = np.load(npz)
        return {k: z[k] for k in z.files}, int(z['x'].shape[0])
    col, n, names = read_ply(path)
    os.makedirs(SCRATCH, exist_ok=True)
    np.savez(npz, **col)
    return col, n

def sv():  return cached('sv', SV_PLY)
def ours(): return cached('ours', OUR_PLY)
