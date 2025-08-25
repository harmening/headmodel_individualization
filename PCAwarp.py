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


# =====================
# Constants
# =====================
NUM_PCAS = 16  # out of 316
# Please do not change the following paths
BASEDIR = os.path.dirname(os.path.realpath(__file__))
PCAS = pth(BASEDIR, 'data', 'pcas', 'ALLpcas.npy')
MEAN_HEAD = pth(BASEDIR, 'data', 'pcas', 'mean_head.npy')
STD_DEV = pth(BASEDIR, 'data', 'pcas', 'std_dev.npy')
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


def export_fieldtrip(bnd_w, output_dir):
    try:
        import scipy.io as sio
    except ImportError:
        print('scipy not installed, skipping .mat export.')
    else:
        sio.savemat(pth(output_dir, 'pca_warped_bnd.mat'), {'bnd': bnd_w})


def export_npy(bnd_w, transform, output_dir):
    np.save(pth(output_dir, 'pca_warped_bnd.npy'), bnd_w)
    np.save(pth(output_dir, 'pca_warped_bnd_transform.npy'), transform)


def export_cedalion(bnd_w, transform, fiducials, output_dir):
    print('Start cedalion export...')
    # Transform again into ctf and then into RAS (this is necessary for tri2nii)
    ras2ctf = np.load('src/transform_acpc2ctf_icbm.npy', allow_pickle=True)
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
                                         bnd_w[shell][0] / 1000,
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
    trans = Transform('head', 'mri', transform)
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
    bnd_w = pca_surfacemesh_warping(fiducials, scalp)

    print('Back-transform warped boundaries...')
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(np.linalg.pinv(transform),
                                        bnd_w[shell][0]),
                        bnd_w[shell][1])

    # Exports
    output_dir = os.path.dirname(args.scalp)
    print(f"Export (.tri, .stl, .mat, .npy) to {output_dir}/...")
    export_openmeeg(bnd_w, output_dir)
    export_fieldtrip(bnd_w, output_dir)
    export_npy(bnd_w, transform, output_dir)
    export_cedalion(bnd_w, transform, fiducials, output_dir)
    export_mne(bnd_w, transform, output_dir)



if __name__ == '__main__':
    main()

