"""Kaggle GPU kernel: train the Sen1Floods11 U-Nets.

Sen1Floods11 is pulled from its public GCS bucket inside the kernel, so the
kernel needs internet enabled. Each kernel folder holds its own copy of this
file with RUNS set, and the pre-event chips are used when attached as input.
"""

import glob
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

RUNS = ["post_hand_weak"]

for pkg in ("rasterio", "gcsfs"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

code = Path(glob.glob("/kaggle/input/**/train_unet.py", recursive=True)[0]).parent
work = Path("/kaggle/working")
for f in ("unet.py", "train_unet.py"):
    shutil.copy(code / f, work / f)
pre = sorted({str(Path(p).parent) for p in glob.glob("/kaggle/input/**/*_S1PreHand.tif", recursive=True)})
args = [sys.executable, "train_unet.py", "--data", "/tmp/sen1floods11", "--out", str(work / "models"), "--runs", *RUNS]
if pre:
    args += ["--pre-dir", pre[0]]
subprocess.run(args, check=True, cwd=work)
