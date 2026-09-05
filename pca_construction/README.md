# Building a PCAwarp PCA basis from your own MRIs

This folder turns a set of segmented head meshes into the three files PCAwarp fits against: `ALLpcas.npy`, `mean_head.npy` and `std_dev.npy`.

- `build_pca_basis.py` — build a basis from a folder of subjects. Start here.
- `qc_meshes.py` — report outlier subjects before you build.
- `pca_core.py` — the PCA itself (`pca_headwise`) and a reconstruction helper (`fit_head`).

Needs `numpy` and `scikit-learn`, plus `scipy` for `--save-mat`.

I'd like these head models to cover more populations than the largely Western OASIS sample they start from. If you build a basis for a population that isn't represented yet, I'd be glad to hear about it, and happy to help or to link out put your database here: nils.harmening@tu-berlin.de


## Recipe

1. Get whole-head T1 MRIs. One MRI per folder. They must **not be defaced and not skull-stripped**, because the pipeline segments scalp and skull and warps fiducials onto the face. 
2. Segment every subject with [MRIsegmentation](https://github.com/harmening/MRIsegmentation/) (`start_segmentation.m`). Use the preprocessing, which aligns to RAS with ATRA. This is also what gives every subject the same triangulation.
3. Collect the four `bnd4_1922_corrected_*.tri` files per subject from that subject's `ctf/` folder. Files with the same names also sit one level up, in the pre-CTF frame; using those silently produces a wrong basis.
4. Run `qc_meshes.py`, then `build_pca_basis.py`.


## Vertex correspondence

All subjects must share one triangulation: same vertex count per shell, same face table, and vertex *i* the same anatomical point everywhere. `projectmesh` in `start_segmentation.m` produces exactly that. The PCA is built coordinate by coordinate, so without it the result is meaningless, and `build_pca_basis.py` stops with an error naming the offending subject.

There is no rigid alignment in this code. The meshes arrive already co-registered in CTF space.


## How the basis is built

Per subject, the four shells are concatenated in the order `scalp, skull, csf, cortex` into one vector of length `D = n_points * 3`. Stacking subjects gives a matrix `M`. Each **column** is then mean-centered and divided by its standard deviation, and `sklearn.decomposition.PCA` is fit on the result. The column means are saved as `mean_head`, the column standard deviations as `std_dev`.

- `ALLpcas.npy` — `(n_components, n_points, 3)`. Directions in the **normalised**
  space, not millimeters.
- `mean_head.npy` — `{shell: (pos, tri)}`, in the input units.
- `std_dev.npy` — `(D,)`, in shell order.

`n_components` is `n_subjects - 1`. PCA on N centered observations determines at most N-1 directions, so the extra component scikit-learn returns is arbitrary noise; `build_pca_basis.py` drops it (`--keep-null-component` keeps it).

Reconstructing a head from coefficients `c`:

```python
pcas_flat = pcas[:N].reshape(N, -1)
mean_flat = np.concatenate([mean[s][0].reshape(-1) for s in
                            ["scalp", "skull", "csf", "cortex"]])
head_flat = Pcas_flat.T @ c * std + mean_flat
```

The `* std` is what converts a component into millimeters. It is also why two bases from different cohorts are only comparable after that multiplication.

`fit_head()` in `pca_core.py` fits `c` to a fully known head, which is a useful check on a left-out subject. Fitting from scalp proxies, which is the actual use case, is in the paper.


## Usage

```bash
python qc_meshes.py       --input-dir /path/to/subjects
python build_pca_basis.py --input-dir /path/to/subjects --output-dir data/pcas/mine
```

Expected layout (`--variant normal`):

```
subjects/subj_001/ctf/bnd4_1922_corrected_{scalp,skull,csf,cortex}.tri
```

Flags:

- `--variant hartmut` — neck-extended scalp (`bnd4_1922_hartmut_corrected_scalp.tri`).
- `--subdir` — where the meshes sit inside each subject folder. Default `ctf`, `''` for directly inside, any relative path works (`anat/ctf`).
- `--save-mat` — also write `pca_basis.mat` for MATLAB and FieldTrip.

Meshes are expected in millimetres and in the CTF frame, which is what
`start_segmentation.m` produces. `qc_meshes.py` prints the bounding box so you can
see at a glance whether that holds.

Then drop the three `.npy` files into `data/pcas/<your_name>/` and set `PCA_DIR = '<your_name>'` in `PCAwarp.py`.


## Reproducibility

Rebuilding a basis from the same meshes reproduces `mean_head` and `std_dev` exactly and the components to ~1e-14.

**Component signs are not stable across machines.** A principal component is only defined up to sign, and different BLAS/LAPACK builds resolve it differently. Rebuilding the `bcas` basis elsewhere reproduced it to 3e-14, but 89 of its 177 components came out negated. Fitting is unaffected, since the coefficient absorbs the sign, but an elementwise diff will look alarming. Compare `|dot(a_k, b_k)|` against 1 instead.

If two builds disagree by more than that, check in this order: the per-subject fiducials and their row order (row 1 nasion, row 2 LPA, row 3 RPA), then the CTF transform derived from them, then the tissue segmentation boundaries, and only then the PCA, which is deterministic given identical meshes.
