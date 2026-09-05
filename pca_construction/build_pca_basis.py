#!/usr/bin/env python
"""
build_pca_basis.py
==================

Build a PCAwarp head-shape basis from your own set of segmented head meshes.

This is the step that comes AFTER the MRIsegmentation pipeline
(https://github.com/harmening/MRIsegmentation). That pipeline segments each
subject into corrected four-tissue boundary meshes in CTF space and, crucially,
gives every subject the SAME triangulation, so that vertex i is the same
anatomical point in every head. That shared vertex correspondence is a required
prerequisite here: the PCA is built coordinate by coordinate, so it is only
meaningful when all subjects share one triangulation. Given that, this script
stacks the meshes and computes the PCA basis used by PCAwarp.

It writes:
    mean_head.npy   dict {shell: (pos, tri)}  -- the mean head shape
    std_dev.npy     (D,) array                -- per-coordinate std used to normalize
    ALLpcas.npy     (n_subjects-1, n_points, 3) -- the informative components
    mean_<shell>.tri                          -- mean shell meshes (viewable)
    pca_basis.mat   (optional, --save-mat)    -- everything, for MATLAB/FieldTrip

Input layout (default matches the MRIsegmentation / PCAwarp corrected meshes):

    <input_dir>/
        <subject_1>/ctf/bnd4_1922_corrected_scalp.tri
                        bnd4_1922_corrected_skull.tri
                        bnd4_1922_corrected_csf.tri
                        bnd4_1922_corrected_cortex.tri
        <subject_2>/ctf/...
        ...

    --variant hartmut  expects the neck-extended scalp instead:
        bnd4_1922_hartmut_corrected_scalp.tri   (skull/csf/cortex unchanged)

Example
-------
    python build_pca_basis.py --input-dir /data/OASIS_healthy --output-dir ./pca_out
    python build_pca_basis.py --input-dir /data/mine --variant hartmut --save-mat

Only NumPy and scikit-learn are required (+ SciPy for --save-mat).
"""
import argparse
import os

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
from tri_io import load_tri, write_tri
from pca_core import pca_headwise

SHELLS = ["scalp", "skull", "csf", "cortex"]

# Default filenames per variant. {tissue} is filled from SHELLS.
FILENAME_TEMPLATES = {
    "normal": {t: "bnd4_1922_corrected_%s.tri" % t for t in SHELLS},
    "hartmut": {
        "scalp": "bnd4_1922_hartmut_corrected_scalp.tri",
        "skull": "bnd4_1922_corrected_skull.tri",
        "csf": "bnd4_1922_corrected_csf.tri",
        "cortex": "bnd4_1922_corrected_cortex.tri",
    },
}


def find_subjects(input_dir, subdir):
    """Return sorted (subject_name, mesh_dir) for every subject folder."""
    subjects = []
    for name in sorted(os.listdir(input_dir)):
        subj_path = os.path.join(input_dir, name)
        if not os.path.isdir(subj_path):
            continue
        mesh_dir = os.path.join(subj_path, subdir) if subdir else subj_path
        if os.path.isdir(mesh_dir):
            subjects.append((name, mesh_dir))
    return subjects


def load_head(mesh_dir, templates):
    """Load one subject's four shells into {shell: (pos, tri)}."""
    head = {}
    for shell in SHELLS:
        fn = os.path.join(mesh_dir, templates[shell])
        if not os.path.isfile(fn):
            raise FileNotFoundError(fn)
        pos, tri = load_tri(fn)
        head[shell] = (pos, tri)
    return head


def check_correspondence(bnds, subject_names):
    """The whole method assumes identical triangulation across subjects.

    Same number of vertices per shell AND the same face table. If this does
    not hold, PCA is comparing apples to oranges. We fail loudly and name the
    offending subject rather than producing a silently wrong basis.
    """
    ref = bnds[0]
    for shell in SHELLS:
        ref_pos, ref_tri = ref[shell]
        for name, bnd in zip(subject_names[1:], bnds[1:]):
            pos, tri = bnd[shell]
            if pos.shape != ref_pos.shape:
                raise ValueError(
                    "Vertex-count mismatch for shell '%s' in subject '%s' "
                    "(%s vs reference %s). All subjects need the same "
                    "vertex-corresponded meshes." % (shell, name, pos.shape, ref_pos.shape))
            if tri.shape != ref_tri.shape or not np.array_equal(tri, ref_tri):
                raise ValueError(
                    "Triangulation mismatch for shell '%s' in subject '%s'. "
                    "All subjects must share the same triangulation (vertex i "
                    "= same anatomical point everywhere)." % (shell, name))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Folder with one subfolder per subject.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the basis files.")
    p.add_argument("--variant", choices=["normal", "hartmut"], default="normal",
                   help="'normal' 4-tissue meshes, or 'hartmut' neck-extended scalp.")
    p.add_argument("--subdir", default="ctf",
                   help="Subfolder inside each subject holding the meshes "
                        "(default 'ctf'; use '' if meshes sit directly in the subject folder).")
    p.add_argument("--keep-null-component", action="store_true",
                   help="Keep the rank-deficient trailing component(s). Off by "
                        "default: PCA on N centred heads determines at most N-1 "
                        "directions, and the extra one is arbitrary noise that "
                        "differs between LAPACK builds. Only useful for "
                        "byte-reproducing an older basis that shipped it.")
    p.add_argument("--save-mat", action="store_true",
                   help="Also write pca_basis.mat (needs SciPy).")
    args = p.parse_args(argv)

    templates = FILENAME_TEMPLATES[args.variant]
    subjects = find_subjects(args.input_dir, args.subdir)
    if not subjects:
        p.error("No subject folders found under %s (subdir=%r)."
                % (args.input_dir, args.subdir))

    print("Loading %d subjects (variant=%s) ..." % (len(subjects), args.variant))
    bnds, names = [], []
    for name, mesh_dir in subjects:
        try:
            bnds.append(load_head(mesh_dir, templates))
            names.append(name)
        except FileNotFoundError as e:
            print("  skipping %s: missing %s" % (name, e))
    if len(bnds) < 2:
        p.error("Need at least 2 usable subjects, found %d." % len(bnds))

    check_correspondence(bnds, names)
    print("  triangulation OK across %d subjects." % len(bnds))

    pcas, mean_bnd, std_dev, bndsize = pca_headwise(
        bnds, drop_null=not args.keep_null_component)
    n_components, n_points, dim = pcas.shape
    print("Built basis: %d components over %d points x %d dims."
          % (n_components, n_points, dim))
    n_dropped = len(bnds) - n_components
    if n_dropped > 0:
        print("  dropped %d rank-deficient component(s); %d subjects in, "
              "%d informative directions out."
              % (n_dropped, len(bnds), n_components))
    elif args.keep_null_component:
        print("  kept the trailing null component (--keep-null-component).")

    os.makedirs(args.output_dir, exist_ok=True)
    tris = {shell: bnds[0][shell][1] for shell in SHELLS}

    # mean_head as {shell: (pos, tri)} + per-shell .tri for inspection.
    mean_head = {}
    start = 0
    for shell in SHELLS:
        size = bndsize[SHELLS.index(shell)] * dim
        pos = mean_bnd[start:start + size].reshape(-1, dim)
        start += size
        mean_head[shell] = (pos, tris[shell])
        write_tri(pos, tris[shell], os.path.join(args.output_dir, "mean_%s.tri" % shell))
    np.save(os.path.join(args.output_dir, "mean_head.npy"), mean_head)
    np.save(os.path.join(args.output_dir, "std_dev.npy"), std_dev)
    np.save(os.path.join(args.output_dir, "ALLpcas.npy"), pcas)

    if args.save_mat:
        import scipy.io as sio
        sio.savemat(os.path.join(args.output_dir, "pca_basis.mat"), {
            "ALLpcas": pcas,
            "mean_bnd": {s: mean_head[s][0] for s in SHELLS},
            "std_dev_pos": std_dev,
            "tris": tris,
            "shells": SHELLS,
            "wise": "headwise",
            "variant": args.variant,
            "unit": "mm",
            "coordsys": "ctf",
        })

    print("Wrote basis to %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
