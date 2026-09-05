#!/usr/bin/env python
"""
qc_meshes.py
============

Report outlier subjects in a set of segmented head meshes, before building a
basis from them.

A single badly segmented or misaligned subject can dominate the PCA. In the bcas
build one subject out of 180 was rotated 20 degrees out of the CTF frame and by
itself carried about 11% of the total shape variance, where 1/180 = 0.56% would
be expected.

A subject is flagged when its scalp sits further from the group mean than
--rms-factor times the MEDIAN distance, or when aligning it to the mean needs
more than --max-rotation degrees.

The median is used rather than a z-score on purpose. A z-score is measured
against the spread of the same set it is judging, so a couple of bad subjects
inflate it and hide themselves, and removing them then promotes the next
subjects into being outliers. The median barely moves either way.

A large rotation means the fiducials landed badly; near 90 degrees means
nasion/LPA/RPA were swapped.

Usage:
    python qc_meshes.py --input-dir /path/to/subjects [--subdir ctf]
                        [--variant normal] [--rms-factor 2.5] [--max-rotation 10]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from tri_io import load_tri  # noqa: E402

SHELLS = ["scalp", "skull", "csf", "cortex"]
TEMPLATES = {
    "normal": {t: "bnd4_1922_corrected_%s.tri" % t for t in SHELLS},
    "hartmut": dict({t: "bnd4_1922_corrected_%s.tri" % t for t in SHELLS},
                    scalp="bnd4_1922_hartmut_corrected_scalp.tri"),
}


def rotation_to(source, target):
    """Degrees of rotation that best aligns source onto target (Kabsch)."""
    s = source - source.mean(0)
    t = target - target.mean(0)
    u, _, vt = np.linalg.svd(s.T @ t)
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--subdir", default="ctf")
    ap.add_argument("--variant", choices=["normal", "hartmut"], default="normal")
    ap.add_argument("--rms-factor", type=float, default=2.5,
                    help="flag a subject whose RMS to the group mean exceeds "
                         "this many times the median RMS (default 2.5)")
    ap.add_argument("--max-rotation", type=float, default=10.0,
                    help="flag a subject needing more than this many degrees "
                         "to align to the group mean (default 10)")
    a = ap.parse_args()
    tpl = TEMPLATES[a.variant]

    names, scalps, skipped = [], [], []
    ref_tri = None
    for name in sorted(os.listdir(a.input_dir)):
        d = os.path.join(a.input_dir, name, a.subdir) if a.subdir \
            else os.path.join(a.input_dir, name)
        if not os.path.isdir(d):
            continue
        missing = [s for s in SHELLS if not os.path.isfile(os.path.join(d, tpl[s]))]
        if missing:
            skipped.append((name, "missing " + ", ".join(missing)))
            continue
        try:
            pos, tri = load_tri(os.path.join(d, tpl["scalp"]))
        except Exception as e:
            skipped.append((name, str(e)[:60]))
            continue
        if ref_tri is None:
            ref_tri, ref_n = tri, pos.shape[0]
        elif pos.shape[0] != ref_n or not np.array_equal(tri, ref_tri):
            skipped.append((name, "triangulation differs from the first subject"))
            continue
        names.append(name)
        scalps.append(pos)

    if len(names) < 3:
        print("Only %d usable subjects, nothing to compare." % len(names))
        return

    X = np.stack(scalps)
    mean = X.mean(0)
    rms = np.sqrt(((X - mean) ** 2).sum(-1).mean(-1))
    rot = np.array([rotation_to(x, mean) for x in X])
    median_rms = float(np.median(rms))
    ratio = rms / (median_rms or 1.0)
    # Share of the total shape variance each subject accounts for.
    share = rms ** 2 / (rms ** 2).sum()

    box = X.reshape(-1, 3).max(0) - X.reshape(-1, 3).min(0)
    print("%d subjects, scalp %d vertices, bounding box %.0f x %.0f x %.0f "
          "(millimetres expected)" % (len(names), ref_n, *box))
    print("RMS to group mean: median %.2f mm    rotation to mean: %.1f +- %.1f deg"
          % (median_rms, rot.mean(), rot.std()))
    print("expected variance share per subject: %.2f%%" % (100.0 / len(names)))

    if skipped:
        print("\nskipped %d:" % len(skipped))
        for n, why in skipped:
            print("  %-14s %s" % (n, why))

    flag = np.flatnonzero((ratio > a.rms_factor) | (rot > a.max_rotation))
    print()
    if flag.size == 0:
        w = int(rms.argmax())
        print("Nothing flagged (RMS < %.1f x median, rotation < %.0f deg)."
              % (a.rms_factor, a.max_rotation))
        print("Worst is %s at %.2f mm (%.2f x median, %.1f deg, %.1f%% of variance)."
              % (names[w], rms[w], ratio[w], rot[w], 100 * share[w]))
        return

    print("%d outlier(s):" % flag.size)
    print("  %-14s %8s %9s %9s %10s"
          % ("subject", "RMS mm", "x median", "rot deg", "variance"))
    for i in flag[np.argsort(-rms[flag])]:
        print("  %-14s %8.2f %9.2f %9.1f %9.1f%%"
              % (names[i], rms[i], ratio[i], rot[i], 100 * share[i]))
    print("\nA subject carrying far more than %.2f%% of the variance distorts the "
          "leading components." % (100.0 / len(names)))
    print("Large rotation = the fiducial warp landed badly; near 90 deg = "
          "nasion/LPA/RPA swapped.")


if __name__ == "__main__":
    main()
