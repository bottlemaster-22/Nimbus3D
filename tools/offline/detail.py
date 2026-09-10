"""Shared helpers for the "can seeds be sized from image detail" measurement.

Everything here is measured off the real capture:
  images/*.jpg                 the 16 photographs that shipped with the scan
  model/model.ply              build 182's trained model
  prepass/prepass_result.json  refined poses
  capture_bundle.json          intrinsics + depthWidth/depthHeight
"""
import io, json, os, sys
import numpy as np
import cv2

import project as P

D = P.D
IMG = r"C:\Users\Undea\Documents\LiKOVA\Scans\Incoming\scan_20260906_164840\images"
SCRATCH = (r"C:\Users\Undea\AppData\Local\Temp\claude"
           r"\C--Users-Undea-Documents-TOMBLINE"
           r"\a90bf6bf-3c58-445f-bbbe-b309e7c3439f\scratchpad")

C0 = 0.28209479177387814
C1 = 0.48860251190291990


def bundle():
    return json.load(io.open(os.path.join(D, 'capture_bundle.json'),
                             encoding='utf-8'))


def frames_with_images():
    """(frame index, absolute jpg path) for every frame whose image is on disk."""
    b = bundle()
    have = set(os.listdir(IMG))
    out = []
    for f in b['frames']:
        name = os.path.basename(f['imagePath'])
        if name in have:
            out.append((int(f['index']), os.path.join(IMG, name)))
    return out


def luma_of(path, w=None, h=None):
    """Rec.709 luma of the sRGB-encoded JPEG, 0..1, exactly SmartImage's."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if w is not None and (bgr.shape[1] != w or bgr.shape[0] != h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    f = bgr.astype(np.float32) / 255.0
    # cv2 is BGR
    return 0.2126 * f[:, :, 2] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 0], f[:, :, ::-1]


def geometry_full(col, count, R, t, fx, fy, cx, cy):
    """Screen position and the UNFILTERED 2D covariance, at whatever
    resolution fx/cx describe. Returns mx, my, z, sa, sb, sc (no 2D filter)."""
    mean = np.stack([col['x'], -col['y'], -col['z']], axis=1).astype(np.float64)
    cam = mean @ R.T + t
    z = cam[:, 2]
    inv_z = 1.0 / np.where(np.abs(z) > 1e-9, z, 1e-9)
    mx = fx * cam[:, 0] * inv_z + cx
    my = fy * cam[:, 1] * inv_z + cy

    scale = np.exp(np.clip(np.stack(
        [col['scale_0'], col['scale_1'], col['scale_2']], axis=1
    ).astype(np.float64), -12, 3))
    q = np.stack([col['rot_1'], -col['rot_2'], -col['rot_3'], col['rot_0']],
                 axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((count, 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    Rm[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
    Rm[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
    Rm[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
    Rm[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    Rm[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
    Rm[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
    Rm[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
    Rm[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    M = Rm * scale[:, None, :]
    sigma_cam = R @ (M @ np.transpose(M, (0, 2, 1))) @ R.T
    j00 = fx * inv_z
    j11 = fy * inv_z
    # Device tangent clamp (build 250+): 1.3x the frame half-angle.
    if P.TANGENT_CLAMP:
        _lx, _ly = 1.3 * cx / fx, 1.3 * cy / fy
        _zz = cam[:, 2]
        _txc = np.clip(cam[:, 0] / np.where(np.abs(_zz) > 1e-9, _zz, 1e-9), -_lx, _lx) * _zz
        _tyc = np.clip(cam[:, 1] / np.where(np.abs(_zz) > 1e-9, _zz, 1e-9), -_ly, _ly) * _zz
    else:
        _txc, _tyc = cam[:, 0], cam[:, 1]
    j02 = -fx * _txc * inv_z * inv_z
    j12 = -fy * _tyc * inv_z * inv_z
    s00, s01, s02 = sigma_cam[:, 0, 0], sigma_cam[:, 0, 1], sigma_cam[:, 0, 2]
    s11, s12, s22 = sigma_cam[:, 1, 1], sigma_cam[:, 1, 2], sigma_cam[:, 2, 2]
    a0 = j00 * s00 + j02 * s02
    a1 = j00 * s01 + j02 * s12
    a2 = j00 * s02 + j02 * s22
    b1 = j11 * s11 + j12 * s12
    b2 = j11 * s12 + j12 * s22
    sa = a0 * j00 + a2 * j02
    sb = a1 * j11 + a2 * j12
    sc = b1 * j11 + b2 * j12
    return mx, my, z, sa, sb, sc, cam


def bilinear(img, x, y):
    """Sample a 2D float image at float coordinates (pixel centres at +0.5)."""
    h, w = img.shape
    xf = np.clip(x - 0.5, 0, w - 1.0001)
    yf = np.clip(y - 0.5, 0, h - 1.0001)
    x0 = xf.astype(np.int64); y0 = yf.astype(np.int64)
    ax = xf - x0; ay = yf - y0
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    return (img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x1] * ax * (1 - ay)
            + img[y1, x0] * (1 - ax) * ay + img[y1, x1] * ax * ay)


def scale_space(luma, sigmas):
    """Blurred luma and blurred luma^2 at each sigma, for footprint variance."""
    l2 = luma * luma
    out = []
    for s in sigmas:
        k = int(2 * round(3 * s) + 1)
        m = cv2.GaussianBlur(luma, (k, k), s, borderType=cv2.BORDER_REFLECT)
        m2 = cv2.GaussianBlur(l2, (k, k), s, borderType=cv2.BORDER_REFLECT)
        out.append((m, np.maximum(m2 - m * m, 0.0)))
    return out


def native_sobel(luma_full, nw, nh):
    """The gradient the pre-pass ALREADY computes: Sobel on the native
    256x192 luma, x0.125 for the kernel gain, exactly classifyFrame's."""
    small = cv2.resize(luma_full, (nw, nh), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(gx * gx + gy * gy) * 0.125, small
