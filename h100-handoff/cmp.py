"""Compare runner outputs: python cmp.py <dir> <ref_dir>. PNG images or .npy video frames."""

import glob
import math
import sys

import numpy as np
from PIL import Image


def load(d):
    npy = sorted(glob.glob(d + "/*.npy"))
    if npy:
        arr = np.load(npy[0])
        return arr.astype(np.float64) * (255.0 if arr.dtype.kind == "f" else 1.0)
    png = sorted(glob.glob(d + "/*.png"))
    return np.asarray(Image.open(png[0]).convert("RGB")).astype(np.float64)


a, b = load(sys.argv[1]), load(sys.argv[2])
assert a.shape == b.shape, (a.shape, b.shape)
diff = np.abs(a - b)
mse = (diff**2).mean()
psnr = "identical" if mse == 0 else f"{10 * math.log10(255**2 / mse):.2f} dB"
print(
    f"{sys.argv[1].rstrip('/').split('/')[-1]} vs {sys.argv[2].rstrip('/').split('/')[-1]}: {psnr}, max diff {diff.max():.0f}, shape {a.shape}"
)
