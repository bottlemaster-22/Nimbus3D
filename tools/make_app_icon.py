"""The LiKOVA app icon, rendered as actual Gaussian splats.

Not a drawing of splats. Every mark on this icon is a real anisotropic 2D
Gaussian, evaluated as exp(-0.5 * d^T Sigma^-1 d) and alpha composited
back-to-front, which is the same maths trainer_rasterize_forward runs on the
phone. The icon is a tiny render from the app's own renderer.

THE MARK: a capture orbit. Splats sit on a circle tilted away from the viewer,
each one oriented along the path, so the ring is built from the lens-shaped
discs the trainer actually produces rather than drawn as an outline. The near
half comes toward you larger, brighter and warmer; the far half recedes,
smaller and cooler. What was scanned is the space they enclose.

WHY IT SURVIVES BLACK AND WHITE, which is how the owner's phone shows it:
depth is carried by LUMINANCE, not hue. The near arc is near-white, the far arc
is dark indigo against a charcoal ground. Desaturate it and the ring still
reads, because the light-to-dark sweep around the orbit is the whole design.
The colour is there for everyone else.
"""
import math
import numpy as np
from PIL import Image

SS = 2                      # supersample factor
N = 1024 * SS
OUT = ('C:/Users/Undea/Documents/TOMBLINE/Nimbus3D/ios/Resources/'
       'AppIcon.png')

# --------------------------------------------------------------- the ground
# Not pure black: the far splats are dark, and they need something to sit on.
# A soft radial lift behind the ring reads as the object it enclosed.
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
cx = cy = (N - 1) / 2.0
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (N * 0.5)

top = np.array([0.055, 0.058, 0.078], np.float32)
bottom = np.array([0.021, 0.022, 0.031], np.float32)
img = (bottom[None, None, :]
       + (top - bottom)[None, None, :] * (1.0 - yy / N)[:, :, None])
glow = np.exp(-(r / 0.42) ** 2)[:, :, None].astype(np.float32)
img = img + glow * np.array([0.045, 0.062, 0.10], np.float32)[None, None, :]


def splat(image, cxp, cyp, sx, sy, angle, colour, opacity):
    """One anisotropic 2D Gaussian, composited over `image`.

    sx is the sigma along the splat's own long axis, sy across it. The conic
    is built the way TrainerSplatDraw builds it: R diag(s^2) R^T, inverted.
    """
    ca, sa = math.cos(angle), math.sin(angle)
    ax = np.array([ca, sa], np.float32)      # long axis
    ay = np.array([-sa, ca], np.float32)     # short axis
    inv = (np.outer(ax, ax) / (sx * sx)) + (np.outer(ay, ay) / (sy * sy))

    reach = 3.0 * max(sx, sy)
    x0 = max(0, int(cxp - reach)); x1 = min(N, int(cxp + reach) + 1)
    y0 = max(0, int(cyp - reach)); y1 = min(N, int(cyp + reach) + 1)
    if x1 <= x0 or y1 <= y0:
        return

    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = gx - cxp
    dy = gy - cyp
    power = -0.5 * (inv[0, 0] * dx * dx
                    + 2.0 * inv[0, 1] * dx * dy
                    + inv[1, 1] * dy * dy)
    alpha = (opacity * np.exp(power))[:, :, None]
    patch = image[y0:y1, x0:x1, :]
    image[y0:y1, x0:x1, :] = patch * (1.0 - alpha) + colour[None, None, :] * alpha


# --------------------------------------------------------------- the iris
# SIX LENS-SHAPED GAUSSIANS IN ROTATIONAL SYMMETRY, like the blades of an
# aperture, each one rotated off its own radius so the whole figure turns.
#
# A ring of tangential splats was tried first and merged into a smooth band:
# put enough soft ellipses along a circle and you have drawn a circle, and the
# thing that makes them splats disappears. Blades keep their identity because
# they overlap at an ANGLE to each other rather than end to end, so every
# crossing shows two distinct lenses.
#
# The light comes from the upper left, which is what carries the figure in
# black and white: opposite blades differ in luminance, not just hue, so
# desaturating leaves a lit form rather than a flat rosette.
BLADES = 6
ORBIT = 0.170 * N          # how far each blade sits from the middle
LEAN = math.radians(73.0)  # how far it is turned off its own radius

WARM = np.array([0.97, 0.98, 1.00], np.float32)
COOL = np.array([0.42, 0.72, 0.99], np.float32)
DEEP = np.array([0.31, 0.47, 0.88], np.float32)

LIGHT = math.radians(215.0)   # where the light is coming from

marks = []
for i in range(BLADES):
    a = 2.0 * math.pi * i / BLADES - math.radians(90.0)
    px = cx + ORBIT * math.cos(a)
    py = cy + ORBIT * math.sin(a)
    angle = a + LEAN

    # 1 facing the light, 0 facing away. This is the whole black and white
    # story: the blades are not all the same brightness.
    lit = 0.5 + 0.5 * math.cos(a - LIGHT)
    if lit > 0.5:
        colour = COOL + (WARM - COOL) * ((lit - 0.5) / 0.5) ** 1.2
    else:
        colour = DEEP + (COOL - DEEP) * (lit / 0.5) ** 0.85

    long_sigma = (0.076 + 0.008 * lit) * N
    short_sigma = (0.0255 + 0.0030 * lit) * N
    opacity = 0.88 + 0.11 * lit
    # Drawn dimmest first so the lit blades sit in front, which is also the
    # order a depth sort would give if the light were the camera.
    marks.append((lit, px, py, long_sigma, short_sigma, angle,
                  colour.astype(np.float32), opacity))

marks.sort(key=lambda m: m[0])
for _, px, py, ls, ss_, ang, col, op in marks:
    splat(img, px, py, ls, ss_, ang, col, op)

# ------------------------------------------------------------ the seed point
# The point every blade is turning around. Round, so it reads as a different
# kind of thing from the oriented discs rather than another one of them.
splat(img, cx, cy, 0.026 * N, 0.026 * N, 0.0,
      np.array([1.0, 1.0, 1.0], np.float32), 0.90)

# ------------------------------------------------------------------ finish
img = np.clip(img, 0.0, 1.0)
img = np.power(img, 1.0 / 1.06)          # a touch of lift in the shadows
out = Image.fromarray((img * 255.0 + 0.5).astype(np.uint8), 'RGB')
out = out.resize((1024, 1024), Image.LANCZOS)
out.save(OUT)

grey = out.convert('L').convert('RGB')
grey.save(OUT.replace('AppIcon.png', 'AppIcon-mono-check.png'))
print('wrote', OUT, out.size)
