#!/usr/bin/env python
"""
Core of the PCAwarp head-shape model: build a PCA basis over a set of
vertex-corresponded head meshes, and reconstruct a head from that basis.

The basis is "headwise": all shells of one head (scalp, skull, csf, cortex,
optionally more) are concatenated into a single vector, so the principal
components capture how the shells co-vary across subjects.

Reference:
    Harmening, von Lühmann, Blankertz: "Data-driven head model
    individualization from digitized electrode positions or photogrammetry
    improves M/EEG source localization accuracy", Imaging Neuroscience, 2026.
    https://doi.org/10.1162/IMAG.a.1073

Only NumPy and scikit-learn are required.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def pca_headwise(bnds, drop_null=True, null_tol=1e-12):
    """Build a headwise PCA basis.

    Parameters
    ----------
    bnds : list of dict
        One entry per subject. Each dict maps a shell name to a
        (pos, tri) tuple, where pos is (n_vertices, dim) and tri is the
        triangulation. Every subject MUST use the same shells, in the same
        order, with the same number of vertices and the same triangulation
        (vertex i is the same anatomical point in every subject).

    drop_null : bool
        Discard components that carry no information (default True).

        PCA on n centred observations can explain at most n-1 directions, but
        scikit-learn returns n components regardless, so the last one lies in
        the numerical null space: it is not determined by the data, and
        different LAPACK builds return different vectors there. Dropping it
        keeps a basis reproducible across machines and avoids shipping a row
        that only ever adds noise. Deeper rank deficiency (duplicate or
        linearly dependent subjects) is caught too, via null_tol.
    null_tol : float
        A component counts as null when its explained variance is below
        null_tol times the largest component's. Only used when drop_null.

    Returns
    -------
    pcas : (n_components, n_points, dim) array
        Principal components. n_components = n_subjects - 1 with the default
        drop_null, or n_subjects when drop_null is False.
        n_points = sum of vertices over all shells.
    mean_bnd : (n_points * dim,) array
        Mean shape (flattened), i.e. mean over subjects of the stacked
        coordinate vector. This is what gets stored as mean_head.
    std_dev : (n_points * dim,) array
        Per-coordinate standard deviation used to normalize before PCA.
        Needed to un-normalize during reconstruction.
    bndsize : list of int
        Number of vertices per shell, in shell order.
    """
    shells = list(bnds[0].keys())
    first = bnds[0][shells[0]][0]
    dim = first.shape[1] if first.ndim == 2 else 1
    bndsize = [len(bnds[0][shell][0]) for shell in shells]
    n_subjects = len(bnds)
    n_points = sum(bndsize)

    # Stack every subject into one row: [shell0 verts | shell1 verts | ...].
    pnts = np.zeros((n_subjects, n_points, dim))
    for i, bnd in enumerate(bnds):
        start = 0
        for shell, size in zip(shells, bndsize):
            pnts[i, start:start + size, :] = bnd[shell][0].reshape((size, dim))
            start += size
    para = pnts.reshape((n_subjects, n_points * dim))

    # Normalize: subtract the mean shape, divide by per-coordinate std.
    mean_bnd = para.mean(axis=0)
    para = para - mean_bnd
    std_dev = para.std(axis=0)
    std_dev[std_dev == 0.0] = 1.0          # guard flat coordinates
    para = para / std_dev

    pca = PCA()
    pca.fit(para)
    components = pca.components_

    if drop_null:
        # Analytic bound first: centring costs one degree of freedom.
        n_keep = min(components.shape[0], n_subjects - 1)
        # Then drop anything below the variance floor, which catches duplicate
        # or linearly dependent subjects rather than just the centring.
        variance = pca.explained_variance_[:n_keep]
        if variance.size and variance[0] > 0:
            informative = np.flatnonzero(variance > null_tol * variance[0])
            if informative.size:
                n_keep = int(informative[-1]) + 1
        components = components[:n_keep]

    pcas = components.reshape((-1, n_points, dim))
    return pcas, mean_bnd, std_dev, bndsize


def fit_head(head, pcas, mean_bnd, std_dev):
    """Fit PCA coefficients to a (fully known) head and reconstruct it.

    Least-squares fit in the normalized space, then map back to coordinates.
    Use this to sanity-check a basis (reconstruct a left-out subject) or as
    the starting point for the partial-observation fit described in the paper
    (e.g. fitting from scalp/electrode positions only).

    Parameters
    ----------
    head : dict {shell: (pos, tri)}   same shells/order as the basis.
    pcas : (n_components, n_points, dim) array from pca_headwise.
    mean_bnd, std_dev : arrays from pca_headwise.

    Returns
    -------
    reconstructed : dict {shell: (n_vertices, dim) array}
    coeff : (n_components,) array of PCA coefficients.
    """
    shells = list(head.keys())
    n_components, n_points, dim = pcas.shape
    bndsize = [len(head[shell][0]) for shell in shells]
    assert sum(bndsize) == n_points, "head vertex count != basis vertex count"
    tot = n_points * dim

    known = np.concatenate([np.asarray(head[shell][0]).reshape(-1) for shell in shells])
    y = (known - mean_bnd) / std_dev
    X = pcas.reshape((n_components, tot)).T

    # fit_intercept=True mirrors the original PCAwarp code; the data is already
    # centered, so the intercept is ~0 and only the coefficients are used.
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    coeff = reg.coef_

    recon_flat = (X.dot(coeff) * std_dev) + mean_bnd
    recon = recon_flat.reshape((n_points, dim))
    reconstructed, start = {}, 0
    for shell, size in zip(shells, bndsize):
        reconstructed[shell] = recon[start:start + size, :].reshape((size, dim))
        start += size
    return reconstructed, coeff
