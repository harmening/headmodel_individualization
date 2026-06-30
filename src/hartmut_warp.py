#!/usr/bin/env python
"""Warp HArtMuT artefact sources into an individual head.

Python port of the HArtMuT individualwarp routine (project_points.jl,
https://github.com/harmening/HArtMuT/tree/main/individualwarp). The original
uses Julia + VTK ray casting. This version uses trimesh, which PCAwarp already
depends on, so no extra toolchain is needed.

The idea: existing nonlinear warps only behave for cortical sources (inside the
skull). Muscle and eye sources sit between skull and scalp, or in front of the
face, so they need their own scheme. For every source we shoot a ray from a
fixed interior point through the source, read where it crosses the skull and
the scalp of the template head, and keep the source at the same relative
position when we re-cross the skull and scalp of the individual head.
"""
import numpy as np
import trimesh


SHELLS = ('skull', 'scalp')


def _as_mesh(bnd_shell):
    verts, tris = bnd_shell
    return trimesh.Trimesh(np.asarray(verts, dtype=float),
                           np.asarray(tris), process=False)


def _ray_hits(mesh, origin, direction):
    """All intersection points of the ray origin -> direction with mesh."""
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=origin[None, :], ray_directions=direction[None, :],
        multiple_hits=True)
    return locs


def _pick(hits, ref, fallback_pos):
    """Pick the intersection closest to ref, or fall back to nearest vertex."""
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return hits[np.argmin(np.linalg.norm(hits - ref, axis=1))]
    # no intersection: degenerate, snap to the nearest source vertex
    return fallback_pos[np.argmin(np.linalg.norm(fallback_pos - ref, axis=1))]


def project_points(pos, source_head, target_head, mean_pnt):
    """Project source positions from the source head into the target head.

    pos          (n, 3) source positions, same frame as both heads.
    source_head  dict with 'skull' and 'scalp', each a (verts, tris) tuple,
                 the template the positions are defined in.
    target_head  same shape, the individual head to warp into.
    mean_pnt     (3,) fixed interior point the rays start from.

    Returns (n, 3) warped positions in the target frame.
    """
    pos = np.asarray(pos, dtype=float)
    mean_pnt = np.asarray(mean_pnt, dtype=float)
    src = {s: _as_mesh(source_head[s]) for s in SHELLS}
    tgt = {s: _as_mesh(target_head[s]) for s in SHELLS}

    new_pos = np.zeros_like(pos)
    for i, p in enumerate(pos):
        direction = p - mean_pnt
        n = np.linalg.norm(direction)
        if n == 0:
            new_pos[i] = p
            continue
        direction = direction / n

        # source head: relative position between skull and scalp along the ray
        s_skull = _pick(_ray_hits(src['skull'], mean_pnt, direction), p,
                        np.asarray(source_head['skull'][0], dtype=float))
        s_scalp = _pick(_ray_hits(src['scalp'], mean_pnt, direction), p,
                        np.asarray(source_head['scalp'][0], dtype=float))
        d_skull = np.linalg.norm(p - s_skull)
        d_scalp = np.linalg.norm(s_scalp - p)
        total = d_skull + d_scalp
        ratio = 0.5 if total == 0 else d_skull / total  # 0 at skull, 1 at scalp

        # target head: reproduce that ratio on its own skull -> scalp segment
        t_skull = _pick(_ray_hits(tgt['skull'], mean_pnt, direction), p,
                        np.asarray(target_head['skull'][0], dtype=float))
        t_scalp = _pick(_ray_hits(tgt['scalp'], mean_pnt, direction), p,
                        np.asarray(target_head['scalp'][0], dtype=float))
        new_pos[i] = t_skull + (t_scalp - t_skull) * ratio
    return new_pos


def is_left_eye(label):
    """True for left-eye sources, dropping the bilateral 'leftright' tag."""
    label = str(label)
    return 'Eye' in label and 'left' in label.lower() and 'leftright' not in label.lower()


def warp_hartmut(artefact_pos, labels, source_head, target_head, mean_pnt):
    """Warp a whole HArtMuT artefact model into the individual head.

    All arguments live in one common frame (PCAwarp warps in CTF). Returns the
    warped positions and the subset that belongs to one (left) eye, which the
    FieldTrip export hands to the example as eye.pos.
    """
    new_pos = project_points(artefact_pos, source_head, target_head, mean_pnt)
    labels = np.asarray([str(l) for l in np.ravel(labels)])
    eye_pos = new_pos[np.array([is_left_eye(l) for l in labels])]
    return new_pos, eye_pos
