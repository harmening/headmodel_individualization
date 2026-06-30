#!/usr/bin/env python
import os
import argparse
import numpy as np
from os.path import join as pth

from src.tri_io import write_tri, load_tri
from src.transform_to_ctf import transform_to_ctf, apply_transform
from src.pca_warp import shortest_dist, elec_warp
from src.tri2nii import tri2nii
from src.nii_postprocessing import postprocessing


BASEDIR = os.path.dirname(os.path.realpath(__file__))


# =====================
# Constants
# =====================
NUM_PCAS = 16  # out of 316
HARTMUT = True  # whether to use Hartmut's PCA-based warping (True) or the
                # original one (False)

# HArtMuT artefact warping. Only runs when HARTMUT is True (the HArtMuT PCAs).
# The artefact sources are defined in the template the chosen HArtMuT model
# lives in, so we warp from that template into the individual head. We start
# from the base HArtMuT NYhead coming from the HArtMuT repo (sibling checkout
# by default), see https://github.com/harmening/HArtMuT. Override the paths if
# yours differ.
HARTMUT_REPO = pth(BASEDIR, '..', 'HArtMuT')
HARTMUT_MODEL = pth(HARTMUT_REPO, 'HArtMuTmodels', 'HArtMuT_NYhead_small.mat')
HARTMUT_TEMPLATE = {'scalp': pth(HARTMUT_REPO, 'individualwarp', 'NYhead',
                                 'scalp.stl'),
                    'skull': pth(HARTMUT_REPO, 'individualwarp', 'NYhead',
                                 'skull.stl')}
HARTMUT_MEANPNT = np.array([0.0, -10.0, 0.0])  # fixed ray origin, model frame, mm
ACPC2CTF = pth(BASEDIR, 'src', 'transform_acpc2ctf_icbm.npy')



# Please do not change the following paths
if HARTMUT:
    pca_dir = 'pcas_hartmut'
else:
    pca_dir = 'pcas'
PCAS = pth(BASEDIR, 'data', pca_dir, 'ALLpcas.npy')
MEAN_HEAD = pth(BASEDIR, 'data', pca_dir, 'mean_head.npy')
STD_DEV = pth(BASEDIR, 'data', pca_dir, 'std_dev.npy')
SHELLS = ['scalp', 'skull', 'csf', 'cortex']



# =====================
# Core Functions
# =====================
def pca_surfacemesh_warping(fiducials, optodes):
    """Perform PCA-based surface mesh warping."""
    mean_bnd = np.load(MEAN_HEAD, allow_pickle=True).item()
    std_dev = np.load(STD_DEV, allow_pickle=True)
    pcas = np.load(PCAS, allow_pickle=True)

    if len(pcas) < NUM_PCAS:
        raise ValueError('Too many PCs requested. Check NUM_PCAS.')
    pcas = pcas[:NUM_PCAS]

    mean_pnt = np.mean(mean_bnd['cortex'][0], axis=0)
    _, min_dist = shortest_dist(mean_pnt, mean_bnd['scalp'][0])
    mean_pnt[2] = np.max(mean_bnd['scalp'][0][:, 2]) - min_dist

    return elec_warp(optodes, pcas, mean_bnd, std_dev)


def load_scalp_file(filepath, nas, lpa, rpa):
    """Load scalp data depending on file format."""
    if filepath.endswith('.npy'):
        return np.load(filepath), nas, lpa, rpa
    if filepath.endswith('.txt'):
        return np.loadtxt(filepath), nas, lpa, rpa

    if filepath.endswith('.bvct'):
        return _load_captrak(filepath)
    if filepath.endswith(('.hsp', '.elp', '.eeg')):
        return _load_polhemus(filepath)
    if filepath.endswith('.elc'):
        return _load_elc(filepath, nas, lpa, rpa)

    return _load_mesh(filepath), nas, lpa, rpa


def _load_captrak(filepath):
    import mne
    captrak = mne.channels.read_dig_captrak(filepath)
    channels = np.array([dig['r'] for dig in captrak.dig if dig['kind'] ==
                         mne.io.constants.FIFF.FIFFV_POINT_EEG])
    scalp_proxies = channels * 1000  # m -> mm
    nas, lpa, rpa = _extract_fiducials(captrak.dig)
    return scalp_proxies, nas, lpa, rpa


def _load_polhemus(filepath):
    import mne
    polhemus = mne.channels.read_dig_polhemus_isotrak(filepath)
    channels = np.array([dig['r'] for dig in polhemus.dig if dig['kind'] ==
                         mne.io.constants.FIFF.FIFFV_POINT_EEG])
    scalp_proxies = channels * 1000  # m -> mm
    nas, lpa, rpa = _extract_fiducials(polhemus.dig)
    return scalp_proxies, nas, lpa, rpa


def _extract_fiducials(dig_points):
    import mne
    nas, lpa, rpa = None, None, None
    for dig in dig_points:
        if dig['kind'] != mne.io.constants.FIFF.FIFFV_POINT_CARDINAL:
            continue
        if dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
            nas = dig['r'] * 1000
        elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
            lpa = dig['r'] * 1000
        elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
            rpa = dig['r'] * 1000
    return nas, lpa, rpa


def _load_elc(filepath, nas, lpa, rpa):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            label, xyz = line.strip().split(':')
            xyz = np.array(list(map(float, xyz.split())))
            if label.strip() in ['NAS', 'Nz', 'Nasion']:
                nas = xyz
            elif label.strip() in ['LPA', 'Lpa', 'LeftEar']:
                lpa = xyz
            elif label.strip() in ['RPA', 'Rpa', 'RightEar']:
                rpa = xyz
            else:
                coords.append(xyz)
    return np.array(coords), nas, lpa, rpa


def _load_mesh(filepath):
    try:
        import trimesh
        return np.array(trimesh.load(filepath).vertices)
    except Exception as e:
        raise RuntimeError(f'Could not load scalp file: {e}')



# =====================
# Export Functions
# =====================
def export_openmeeg(bnd_w, output_dir):
    for shell in SHELLS:
        write_tri(bnd_w[shell][0], bnd_w[shell][1],
                  pth(output_dir, f'pca_warped_{shell}.tri'))
        try:
            import trimesh
            mesh = trimesh.Trimesh(bnd_w[shell][0], bnd_w[shell][1])
            mesh.export(pth(output_dir, f'pca_warped_{shell}.stl'),
                        file_type='stl_ascii')
        except ImportError:
            pass


def export_fieldtrip(bnd_w, output_dir, coordsys='digitized', eye_pos=None,
                     fiducials=None, hartmut=None, fname='pcawarp_bnd.mat'):
    """Write the surfaces as bnd.mat to be used by fieldtrip's dipolefitting.

    Standard FieldTrip constructs only: a 4-element mesh array `bnd` (scalp,
    skull, csf, cortex), a parallel `tissue` cell array, an optional `eye`
    struct with one eye's candidate positions for artefact-aware dipole fitting
    a la HArtMuT, and an optional `fiducials` struct (Nz, LPA, RPA, in the same
    digitized frame) so the head can be transformed to a known coordinate system.
    The `coordsys` field is set to 'digitized' by default,

    When HARTMUT is True, `hartmut` carries the warped artefact model (positions,
    source model dict), which is written alongside as the individual HArtMuT
    artefact model.
    """
    try:
        import scipy.io as sio
    except ImportError:
        print('scipy not installed, skipping .mat export.')
        return

    dtype = np.dtype([('pos', 'O'), ('tri', 'O'), ('unit', 'O'),
                      ('coordsys', 'O')])
    bnd = np.empty((1, len(SHELLS)), dtype=dtype)
    for i, shell in enumerate(SHELLS):
        pos, tri = bnd_w[shell]
        bnd[0, i]['pos'] = np.asarray(pos, dtype=float)
        bnd[0, i]['tri'] = np.asarray(tri, dtype=float) + 1  # MATLAB uses 1-based indexing
        bnd[0, i]['unit'] = 'mm'
        bnd[0, i]['coordsys'] = coordsys

    out = {'bnd': bnd, 'tissue': np.array(SHELLS, dtype=object)}
    if eye_pos is not None and len(eye_pos):
        out['eye'] = {'pos': np.asarray(eye_pos, dtype=float)}
    if fiducials is not None and len(fiducials):
        out['fiducials'] = {'pos': np.asarray(fiducials, dtype=float),
                            'label': np.array(['Nz', 'LPA', 'RPA'], dtype=object)}
    sio.savemat(pth(output_dir, fname), out)

    if HARTMUT and hartmut is not None:
        new_pos, model = hartmut
        export_hartmut(new_pos, model, output_dir)


def export_hartmut(new_pos, model, output_dir, fname='pcawarp_hartmut.mat'):
    """Save the warped HArtMuT artefact model, mirroring the Julia individualwarp
    output: same orientation, labels and unit, only the positions are warped."""
    import scipy.io as sio
    artefactmodel = {'pos': np.asarray(new_pos, dtype=float),
                     'orientation': model['orientation'],
                     'labels': model['labels'],
                     'unit': model['unit']}
    sio.savemat(pth(output_dir, fname),
                {'HArtMuT': {'artefactmodel': artefactmodel}})


def warp_artefacts_ctf(bnd_w_ctf):
    """Warp the HArtMuT artefact sources into the individual head, in CTF.

    bnd_w_ctf holds the warped individual surfaces in CTF (before the
    back-transform to the input frame). Returns the warped artefact positions,
    the one-eye candidate positions, and the source model dict, all in CTF.
    """
    import scipy.io as sio
    import trimesh
    from src.hartmut_warp import warp_hartmut

    acpc2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    acpc2ctf[:3, 3] *= 1000.0  # m -> mm

    # source template (HArtMuT model frame) into CTF
    src_head = {}
    for shell in ('scalp', 'skull'):
        mesh = trimesh.load(HARTMUT_TEMPLATE[shell])
        verts = np.asarray(mesh.vertices, dtype=float)
        src_head[shell] = (apply_transform(acpc2ctf, verts),
                           np.asarray(mesh.faces))
    tgt_head = {shell: bnd_w_ctf[shell] for shell in ('scalp', 'skull')}

    H = sio.loadmat(HARTMUT_MODEL, struct_as_record=False,
                    squeeze_me=True)['HArtMuT']
    am = H.artefactmodel
    model = {'pos': np.asarray(am.pos, dtype=float),
             'orientation': am.orientation, 'labels': am.labels,
             'unit': am.unit}
    pos_ctf = apply_transform(acpc2ctf, model['pos'])
    mean_pnt = apply_transform(acpc2ctf, HARTMUT_MEANPNT[None, :])[0]

    new_pos_ctf, eye_ctf = warp_hartmut(pos_ctf, model['labels'],
                                        src_head, tgt_head, mean_pnt)
    return new_pos_ctf, eye_ctf, model


def export_npy(bnd_w, transform, output_dir):
    np.save(pth(output_dir, 'pca_warped_bnd.npy'), bnd_w)
    np.save(pth(output_dir, 'pca_warped_bnd_transform.npy'), transform)


def export_cedalion(bnd_w, transform, fiducials, output_dir):
    print('Start cedalion export...')
    # Transform again into ctf and then into RAS (this is necessary for tri2nii)
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3,3] *= 1000 # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(transform, bnd_w[shell][0]),
                        bnd_w[shell][1])
        bnd_w[shell] = (apply_transform(ctf2ras, bnd_w[shell][0]),
                        bnd_w[shell][1])
    bnds = [(bnd_w[shell][0], bnd_w[shell][1]) for shell in SHELLS]
    cedalion_output_dir = pth(output_dir, 'cedalion')
    os.makedirs(cedalion_output_dir, exist_ok=True)

    # Create segmentation masks
    """
    back_transform = np.linalg.pinv(transform) @ ras2ctf # first back to ctf,
                                                         # then in phtgrammetry
                                                         # coordinate system
    """
    back_transform = np.eye(4) #for RAS
    tri2nii(bnds, output_dir=cedalion_output_dir, transform=back_transform,
            t1_fn='src/template.nii', meshes='all')

    # Postprocessing and clean up
    postprocessing(cedalion_output_dir, num_tissues=4)
    for i in range(5):
        for fn in [f'mask_{i}.nii', f'mask_{i}.nii.gz']:
            if os.path.exists(pth(cedalion_output_dir, fn)):
                os.remove(pth(cedalion_output_dir, fn))

    # Save ditigized2ras transform (photogrammetry to output)
    from_crs, to_crs = "digitized", "ras"
    ditigized2ras = ctf2ras @ transform # first to ctf, then ctf2ras
    fiducials_ras = apply_transform(ditigized2ras, fiducials)
    try:
        import xarray as xr
    except ImportError:
        print('xarray not installed, skipping ditigized2ras export.')
    else:
        ditigized2ras = xr.DataArray(ditigized2ras, dims=[to_crs, from_crs])
        ditigized2ras.to_netcdf(pth(cedalion_output_dir, "t_ditigized2ras.nc"))
  
    # Save landmarks
    import json
    landmarks = [{"id": i,
                  "label": label,
                  "position": list(pos),
                  "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  }
                 for i, (label, pos) in enumerate(zip(['Nz', 'LPA', 'RPA'],
                                                      fiducials_ras))]
    data_dict = {"@schema": str("https://raw.githubusercontent.com/slicer/"
                              "slicer/master/Modules/Loadable/Markups/"
                              "Resources/Schema/markups-schema-v1.0.3.json"),
                 "markups": [{
                     "type": "Fiducial",
                     "coordinateSystem": to_crs,
                     "coordinateUnits": "mm", #landmark.units,
                     "controlPoints": landmarks,
                     }]}
    json.dump(data_dict, open(pth(cedalion_output_dir, "landmarks.mrk.json"),
                              "w"), indent=2)


def export_mne(bnd_w, transform, output_dir):
    print('Start python-MNE export...')
    try:
        import nibabel as nib
        import mne
        from mne.transforms import Transform
    except ImportError:
        print('mne or nibabel not installed, skipping MNE export.')
        return

    mne_output_dir = pth(output_dir, 'mne', 'pcawarp')
    os.makedirs(pth(mne_output_dir, 'bem'), exist_ok=True)
    os.makedirs(pth(mne_output_dir, 'mri'), exist_ok=True)
    mne_names = {'scalp': 'outer_skin.surf',
                 'skull': 'outer_skull.surf',
                 'csf': 'inner_skull.surf',
                 'cortex': 'inner_csf.surf'}
    for shell in SHELLS:
        nib.freesurfer.io.write_geometry(pth(mne_output_dir, 'bem',
                                             mne_names[shell]),
                                         bnd_w[shell][0],
                                         bnd_w[shell][1])

    # Create a fake T1.mgz file from the cedalion output (for MNE plotting)
    data = nib.load(pth(output_dir, 'cedalion', 'mask_skin.nii')).get_fdata()
    new_data = np.zeros(data.shape)
    tissue_color = {'skin': 0.45, # ~0.4–0.6 (depends on fat content)
                    'bone': 0.05, # ~0.0–0.1 (very dark, almost no signal)
                    'csf': 0.1, #CSF ~0.0–0.1 (dark; long T1 -> low signal)
                    'cortex': 0.55} #cortex (0.5–0.7 (wm brighter than gm)
    for lab, tissue in enumerate(['skin', 'bone', 'csf', 'cortex']):
        mask = nib.load(pth(output_dir, 'cedalion', f"mask_{tissue}.nii"))
        new_data[mask.get_fdata() == 1] = tissue_color[tissue]
    new_img = nib.nifti1.Nifti1Image(new_data, mask.affine, mask.header)
    nib.save(new_img, pth(mne_output_dir, 'mri', 'T1.mgz'))

    # Transform object for coregistration
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3,3] *= 1000 # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    ditigized2ras = ctf2ras @ transform # first to ctf, then ctf2ras
    trans = Transform('head', 'mri', ditigized2ras)
    mne.write_trans(pth(mne_output_dir, 'ditigized2ras-trans.fif'), trans)



# =====================
# Main Workflow
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-scalp', type=str, required=True,
                        help='Path to scalp proxy file. Can be a .npy- or a \
                              .txt-file or a surface mesh of typical mesh \
                              formats like .stl, .obj, .ply')
    parser.add_argument('-nas', type=float, nargs=3, default=None,
                        help='Nasion (NAS) fiducial coordinates.')
    parser.add_argument('-lpa', type=float, nargs=3, default=None,
                        help='Left preauricular (LPA) fiducial coordinates.')
    parser.add_argument('-rpa', type=float, nargs=3, default=None,
                        help='Right preauricular (RPA) fiducial coordinates.')
    args = parser.parse_args()

    scalp, nas, lpa, rpa = load_scalp_file(args.scalp, args.nas, args.lpa,
                                           args.rpa)
    if nas is None or lpa is None or rpa is None:
        raise ValueError('Fiducials (nas, lpa, rpa) must be provided.')

    print('Transforming into CTF coordinate system...')
    mean_scalp = np.load(MEAN_HEAD, allow_pickle=True).item()['scalp'][0]
    fiducials = np.array([nas, lpa, rpa])
    scalp, transform = transform_to_ctf(scalp, *fiducials,
                                        mean_scalp=mean_scalp,
                                        return_transform=True)
    
    print('Cut scalp proxy points above the ears...')
    CUT = 30#mm 
    scalp = scalp[scalp[:, 2] > CUT]
    print('Decimate number of scalp proxy points to 100...')
    if len(scalp) > 100:
        scalp = scalp[np.random.choice(len(scalp), 100, replace=False)]

    print('Performing PCA warping...')
    bnd_w_ctf = pca_surfacemesh_warping(fiducials, scalp)

    output_dir = os.path.dirname(args.scalp)
    inv = np.linalg.pinv(transform)  # CTF -> input frame

    # Warp the HArtMuT artefact sources (muscle, eyes) into the individual head.
    # Only for the HArtMuT PCAs, the artefact model lives in their template.
    eye_pos = None
    hartmut = None
    eye_ctf_export = None
    hartmut_ctf = None
    if HARTMUT:
        print('Warping HArtMuT artefact sources into the individual head...')
        new_pos_ctf, eye_ctf, model = warp_artefacts_ctf(bnd_w_ctf)
        new_pos = apply_transform(inv, new_pos_ctf)
        eye_pos = apply_transform(inv, eye_ctf) if len(eye_ctf) else None
        hartmut = (new_pos, model)
        eye_ctf_export = eye_ctf if len(eye_ctf) else None
        hartmut_ctf = (new_pos_ctf, model)

    print('Back-transform warped boundaries...')
    bnd_w = {}
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(inv, bnd_w_ctf[shell][0]),
                        bnd_w_ctf[shell][1])

    # Exports
    print(f"Export (.tri, .stl, .mat, .npy) to {output_dir}/...")
    export_openmeeg(bnd_w, output_dir)
    # export the FieldTrip head in three frames: the input (digitized) frame, CTF, and MNI. CTF and
    # MNI are recognized coordinate systems, so HArtMuT eye fitting works on those two files without
    # any further coordinate transform, and a standard cap aligns with the MNI head
    fiducials_ctf = apply_transform(transform, fiducials)
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3, 3] *= 1000  # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    bnd_w_mni = {shell: (apply_transform(ctf2ras, bnd_w_ctf[shell][0]), bnd_w_ctf[shell][1])
                 for shell in SHELLS}
    eye_mni = apply_transform(ctf2ras, eye_ctf_export) if eye_ctf_export is not None else None
    fiducials_mni = apply_transform(ctf2ras, fiducials_ctf)
    export_fieldtrip(bnd_w, output_dir, coordsys='digitized', eye_pos=eye_pos,
                     fiducials=fiducials, fname='pcawarp_bnd_digitized.mat')
    export_fieldtrip(bnd_w_ctf, output_dir, coordsys='ctf', eye_pos=eye_ctf_export,
                     fiducials=fiducials_ctf, hartmut=hartmut_ctf, fname='pcawarp_bnd_ctf.mat')
    export_fieldtrip(bnd_w_mni, output_dir, coordsys='mni', eye_pos=eye_mni,
                     fiducials=fiducials_mni, fname='pcawarp_bnd_mni.mat')
    export_npy(bnd_w, transform, output_dir)
    export_cedalion(bnd_w, transform, fiducials, output_dir)
    export_mne(bnd_w, transform, output_dir)



if __name__ == '__main__':
    main()

